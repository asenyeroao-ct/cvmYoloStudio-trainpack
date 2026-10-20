# ul_bridge — JSON-lines bridge to Ultralytics.
# Copyright (C) 2026 YOLO Studio authors
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU Affero General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License for more details. You should have received a copy of the GNU Affero
# General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Translate JSON requests into Ultralytics calls and the results back into JSON.

Process modes (stdin: one JSON object per line; stdout: one JSON object per line):

    python -m ul_bridge          one-shot: reads one line ({"ops": [...]} or a single op), runs it, exits
    python -m ul_bridge serve    resident: handles requests until "exit" or end of stdin

The first stdout line is {"ready": true, "protocol_version": N, "bridge": version}. Everything that
Ultralytics prints goes to stderr. Each request carries an "id"; replies use the same id:

    {"id": 1, "ok": true, "result": ...}
    {"id": 1, "ok": false, "error": {"code": ..., "type": ..., "message": ...}}
    {"id": 1, "event": "<callback name>", "data": {...}}      during "call" with "events"
    {"id": 1, "item": ..., "n": k}                            during "stream"

Operations: env, setup, load, call, get, free, stream, feed, drop, cancel, memory, ckpt_read,
ckpt_write, reorder_head, exit. Values that are not plain JSON use one-key tagged objects:
{"$h": handle}, {"$nd": {"dtype", "shape", "data"(base64) | "value"}}, {"$rle": {"shape", "counts"}},
{"$dict": [[key, value], ...]}, {"$tuple": [...]}; in requests also {"$npz": path}, a dict of tensors read
from a .npz file.

"call" and "get" with "ret": "arrays" take a mapping of tensors / arrays (for example a state_dict) and
return [[key, shape, dtype], ...]; with "out" the arrays are also written to that .npz file.
"""
from __future__ import annotations

import base64
import gc
import importlib
import json
import os
import queue
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROTOCOL_VERSION = 2
__version__ = "0.2.0"

# Ultralytics classes that "load" can create: kind -> (module, attribute)
_KINDS = {
    "YOLO": ("ultralytics", "YOLO"),
    "YOLOE": ("ultralytics", "YOLOE"),
    "YOLOWorld": ("ultralytics", "YOLOWorld"),
    "RTDETR": ("ultralytics", "RTDETR"),
    "SAM": ("ultralytics", "SAM"),
    "FastSAM": ("ultralytics", "FastSAM"),
    "NAS": ("ultralytics", "NAS"),
    "SAMPredictor": ("ultralytics.models.sam", "Predictor"),
    "SAM2Predictor": ("ultralytics.models.sam", "SAM2Predictor"),
    "SAM2VideoPredictor": ("ultralytics.models.sam", "SAM2VideoPredictor"),
}
_SMALL = 4096          # arrays up to this many elements are sent as nested lists
_EVENT_SMALL = 64      # inside callback events, larger arrays are reduced to {"$shape": ...}


class BridgeError(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


# ---------------------------------------------------------------------- array helpers
def _np():
    import numpy as np

    return np


def _is_tensor(x) -> bool:
    return type(x).__module__.startswith("torch") and hasattr(x, "detach") and hasattr(x, "cpu")


def rle_encode(arr) -> dict:
    """Boolean array (..., H, W) -> {"shape": [...], "counts": [[run lengths], ...]} (row-major, zeros first)."""
    np = _np()
    a = np.asarray(arr).astype(bool)
    shape = list(a.shape)
    planes = a.reshape(-1, shape[-2] * shape[-1]) if a.ndim >= 2 else a.reshape(1, -1)
    counts = []
    for flat in planes:
        if flat.size == 0:
            counts.append([])
            continue
        change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        runs = np.diff(np.concatenate(([0], change, [flat.size]))).tolist()
        counts.append(([0] + runs) if flat[0] else runs)
    return {"shape": shape, "counts": counts}


def rle_decode(obj: dict):
    np = _np()
    shape = [int(v) for v in obj["shape"]]
    plane = shape[-2] * shape[-1] if len(shape) >= 2 else (shape[0] if shape else 0)
    planes = []
    for runs in obj["counts"]:
        flat = np.zeros(plane, dtype=bool)
        pos, val = 0, False
        for n in runs:
            if val:
                flat[pos:pos + n] = True
            pos += n
            val = not val
        planes.append(flat)
    return np.stack(planes).reshape(shape) if planes else np.zeros(shape, dtype=bool)


def _nd_decode(obj: dict):
    np = _np()
    dtype = obj.get("dtype") or "float32"
    if "data" in obj:
        a = np.frombuffer(base64.b64decode(obj["data"]), dtype=dtype)
        return a.reshape([int(v) for v in obj["shape"]]).copy()
    return np.asarray(obj["value"], dtype=dtype)


def _npz_tensors(path: str) -> dict:
    """.npz file -> {key: torch.Tensor}."""
    import torch

    np = _np()
    with np.load(path, allow_pickle=False) as z:
        return {k: torch.from_numpy(np.array(z[k])) for k in z.files}


def _arrays(value, out: str | None) -> list:
    """Mapping of tensors / arrays -> [[key, shape, dtype], ...]; with out, also saved as .npz."""
    np = _np()
    if not hasattr(value, "items"):
        raise BridgeError("not_a_mapping", type(value).__name__)
    arrays = {}
    for k, v in value.items():
        if _is_tensor(v):
            v = v.detach().cpu()
            if str(v.dtype) == "torch.bfloat16":
                v = v.float()
            v = v.numpy()
        arrays[str(k)] = np.asarray(v)
    if out:
        np.savez(out, **arrays)
    return [[k, list(a.shape), str(a.dtype)] for k, a in arrays.items()]


# ---------------------------------------------------------------------- the bridge
class Bridge:
    def __init__(self, out) -> None:
        self._out = out
        self._out_lock = threading.Lock()
        self.handles: dict[str, object] = {}
        self._next = 0
        self.streams: dict = {}
        self.control: "queue.Queue[dict]" = queue.Queue()
        self.pending: list[dict] = []
        self._active_generators: set = set()

    # -------------------------------------------------- output
    def send(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        with self._out_lock:
            self._out.write(line + "\n")
            self._out.flush()

    def store(self, obj, name: str | None = None) -> str:
        if not name:
            self._next += 1
            name = f"h{self._next}"
        self.handles[name] = obj
        return name

    def obj(self, name) -> object:
        try:
            return self.handles[name]
        except (KeyError, TypeError):
            raise BridgeError("unknown_handle", str(name)) from None

    # -------------------------------------------------- encoding
    def enc(self, x, handles: bool = True, small: int = _SMALL, results: dict | None = None):
        np = _np()
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, Path):
            return str(x)
        name = type(x).__name__
        if name == "Results":
            return self.enc_results(x, results or {})
        if callable(getattr(x, "mean_results", None)) or name == "ClassifyMetrics":
            return self.enc_metrics(x)
        if _is_tensor(x):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            if x.dtype == bool and x.ndim >= 2:
                return {"$rle": rle_encode(x)}
            if x.size <= small:
                return x.tolist()
            if small == _EVENT_SMALL:
                return {"$shape": list(x.shape), "dtype": str(x.dtype)}
            c = np.ascontiguousarray(x)
            return {"$nd": {"dtype": str(c.dtype), "shape": list(c.shape),
                            "data": base64.b64encode(c.tobytes()).decode("ascii")}}
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, (list, tuple, set)):
            return [self.enc(v, handles, small, results) for v in x]
        if isinstance(x, dict):
            if all(isinstance(k, str) for k in x):
                return {k: self.enc(v, handles, small, results) for k, v in x.items()}
            return {"$dict": [[self.enc(k, False), self.enc(v, handles, small, results)] for k, v in x.items()]}
        if name in ("device", "dtype"):
            return str(x)
        if hasattr(x, "__dict__") and name in ("SimpleNamespace", "IterableSimpleNamespace"):
            return self.enc(dict(vars(x)), handles, small, results)
        if handles:
            return {"$h": self.store(x)}
        return repr(x)

    def dec(self, x):
        if isinstance(x, list):
            return [self.dec(v) for v in x]
        if isinstance(x, dict):
            if len(x) == 1:
                (k, v), = x.items()
                if k == "$h":
                    return self.obj(v)
                if k == "$nd":
                    return _nd_decode(v)
                if k == "$npz":
                    return _npz_tensors(v)
                if k == "$rle":
                    return rle_decode(v)
                if k == "$dict":
                    return {self.dec(a): self.dec(b) for a, b in v}
                if k == "$tuple":
                    return tuple(self.dec(i) for i in v)
            return {k: self.dec(v) for k, v in x.items()}
        return x

    @staticmethod
    def _flat(t) -> list:
        if t is None:
            return None
        if _is_tensor(t):
            t = t.detach().cpu().numpy()
        return _np().asarray(t).reshape(-1).tolist()

    def enc_results(self, r, opts: dict) -> dict:
        """Ultralytics Results -> raw arrays. Coordinates are normalized by orig_shape (obb.xywhr is in pixels)."""
        f = self._flat
        out = {"orig_shape": [int(v) for v in r.orig_shape[:2]], "path": str(getattr(r, "path", "") or ""),
               "speed": {k: v for k, v in (r.speed or {}).items()}}
        b = getattr(r, "boxes", None)
        if b is not None:
            out["boxes"] = {"n": len(b), "xyxyn": f(b.xyxyn), "conf": f(b.conf), "cls": f(b.cls),
                            "id": f(b.id) if getattr(b, "id", None) is not None else None}
        o = getattr(r, "obb", None)
        if o is not None:
            out["obb"] = {"n": len(o), "xywhr": f(o.xywhr), "xyxyxyxyn": f(o.xyxyxyxyn), "conf": f(o.conf),
                          "cls": f(o.cls), "id": f(o.id) if getattr(o, "id", None) is not None else None}
        m = getattr(r, "masks", None)
        if m is not None:
            mode = opts.get("masks")
            if mode == "xy":
                out["masks"] = {"n": len(m), "xyn": [f(s) for s in m.xyn]}
            elif mode == "rle":
                out["masks"] = {"n": len(m), "rle": rle_encode(m.data.detach().cpu().numpy() > 0.5)}
        k = getattr(r, "keypoints", None)
        if k is not None and getattr(k, "data", None) is not None:
            out["keypoints"] = {"shape": list(k.data.shape), "xyn": f(k.xyn),
                                "conf": f(k.conf) if getattr(k, "conf", None) is not None else None}
        p = getattr(r, "probs", None)
        if p is not None:
            out["probs"] = {"data": f(p.data)}
        return out

    def enc_metrics(self, m) -> dict:
        """Ultralytics metrics objects -> their numeric outputs (methods called as Ultralytics defines them)."""
        out: dict = {"type": type(m).__name__}

        def safe(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001 - a missing attribute just leaves the field out
                return None

        if callable(getattr(m, "mean_results", None)):
            out["mean_results"] = self.enc(safe(m.mean_results), False)
        idx = getattr(m, "ap_class_index", None)
        if idx is not None and callable(getattr(m, "class_result", None)):
            out["ap_class_index"] = self.enc(idx, False)
            out["class_results"] = [self.enc(safe(lambda i=i: m.class_result(i)), False) for i in range(len(idx))]
        for attr in ("fitness", "speed", "nt_per_class", "nt_per_image", "f1", "results_dict", "top1", "top5",
                     "save_dir", "task"):
            if hasattr(m, attr):
                out[attr] = self.enc(safe(lambda a=attr: getattr(m, a)), False)
        for attr in ("box", "seg", "pose"):
            sub = getattr(m, attr, None)
            if sub is not None and callable(getattr(sub, "mean_results", None)):
                out[attr] = self.enc_metrics(sub)
        return out

    # -------------------------------------------------- field expressions (callback events)
    @staticmethod
    def _path(obj, path: str):
        for part in [p for p in path.split(".") if p]:
            if isinstance(obj, dict):
                obj = obj[int(part) if part.lstrip("-").isdigit() and part not in obj else part]
            elif part.lstrip("-").isdigit() and isinstance(obj, (list, tuple)) or (
                    part.lstrip("-").isdigit() and hasattr(obj, "__getitem__") and not hasattr(obj, part)):
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    def field(self, obj, expr: str):
        """'a.b.c', 'len(a.b)', 'type(a)', 'a.method()' evaluated on a callback argument."""
        try:
            if expr.startswith("len(") and expr.endswith(")"):
                return len(self._path(obj, expr[4:-1]))
            if expr.startswith("type(") and expr.endswith(")"):
                return type(self._path(obj, expr[5:-1])).__name__
            if expr.endswith("()"):
                return self._path(obj, expr[:-2])()
            return self._path(obj, expr)
        except Exception:  # noqa: BLE001 - unavailable fields are null
            return None

    # -------------------------------------------------- dispatch
    def handle(self, msg: dict) -> bool:
        """Runs one request and sends its reply; returns whether it succeeded."""
        rid = msg.get("id")
        op = msg.get("op")
        fn = getattr(self, f"op_{op}", None) if isinstance(op, str) else None
        if fn is None:
            self.send({"id": rid, "ok": False, "error": {"code": "unknown_op", "type": "", "message": str(op)}})
            return False
        try:
            result = fn(msg)
            if op != "stream" or not msg.get("method"):
                self.send({"id": rid, "ok": True, "result": result})
            return True
        except BridgeError as exc:
            self.send({"id": rid, "ok": False, "error": {"code": exc.code, "type": "", "message": str(exc)}})
        except Exception as exc:  # noqa: BLE001 - every failure is reported, the process stays up
            traceback.print_exc()
            self.send({"id": rid, "ok": False, "error": {"code": "exception", "type": type(exc).__name__,
                                                         "message": str(exc)}})
        return False

    # -------------------------------------------------- operations
    def op_exit(self, msg: dict):
        return {}

    def op_env(self, msg: dict):
        import torch
        import ultralytics
        from ultralytics.utils import ASSETS

        devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                devices.append({"index": i, "name": p.name, "capability": [p.major, p.minor],
                                "memory": int(p.total_memory)})
        return {"protocol_version": PROTOCOL_VERSION, "bridge": __version__, "python": sys.version.split()[0],
                "executable": sys.executable, "pid": os.getpid(), "ultralytics": ultralytics.__version__,
                "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": bool(devices),
                "devices": devices, "assets": str(ASSETS)}

    def op_setup(self, msg: dict):
        """Process setup before Ultralytics is imported: extra import paths, DLL folders, blocked modules."""
        for e in msg.get("sys_path") or []:
            p = str(e["path"])
            if p not in sys.path:
                sys.path.insert(0, p) if e.get("prepend") else sys.path.append(p)
        for d in msg.get("dll_dirs") or []:
            if hasattr(os, "add_dll_directory") and os.path.isdir(d):
                os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        for name in msg.get("block_modules") or []:
            sys.modules.setdefault(name, None)  # type: ignore[arg-type]
        for k, v in (msg.get("env") or {}).items():
            os.environ[str(k)] = str(v)
        importlib.invalidate_caches()
        return {"sys_path": len(sys.path)}

    def op_load(self, msg: dict):
        kind = msg.get("kind")
        if kind not in _KINDS:
            raise BridgeError("unknown_kind", str(kind))
        mod, attr = _KINDS[kind]
        cls = getattr(importlib.import_module(mod), attr)
        args = self.dec(msg.get("args") or [])
        if "model" in msg:
            args = [msg["model"], *args]
        obj = cls(*args, **self.dec(msg.get("kwargs") or {}))
        return {"handle": self.store(obj, msg.get("handle"))}

    def _callbacks(self, target, rid, events: dict) -> list:
        added = []
        for name, fields in (events or {}).items():
            def cb(arg, _name=name, _fields=list(fields or [])):
                data = {f: self.enc(self.field(arg, f), False, _EVENT_SMALL) for f in _fields}
                self.send({"id": rid, "event": _name, "data": data})

            target.add_callback(name, cb)
            added.append((name, cb))
        return added

    def _patches(self, patches: dict) -> None:
        rng = (patches or {}).get("multi_scale_range")
        if rng:
            _patch_multi_scale(int(rng[0]), int(rng[1]))

    def op_call(self, msg: dict):
        target = self.obj(msg.get("handle"))
        method = self._path(target, msg["method"])
        self._patches(msg.get("patches"))
        added = self._callbacks(target, msg.get("id"), msg.get("events")) if msg.get("events") else []
        try:
            value = method(*self.dec(msg.get("args") or []), **self.dec(msg.get("kwargs") or {}))
        finally:
            for name, cb in added:
                try:
                    target.callbacks[name].remove(cb)
                except (AttributeError, KeyError, ValueError):
                    pass
        if msg.get("ret") == "handle":
            return {"$h": self.store(value, msg.get("as"))}
        if msg.get("ret") == "none":
            return None
        if msg.get("ret") == "arrays":
            return _arrays(value, msg.get("out"))
        return self.enc(value, True, results=msg.get("results"))

    def op_get(self, msg: dict):
        value = self._path(self.obj(msg.get("handle")), msg.get("attr") or "")
        if msg.get("ret") == "handle":
            return {"$h": self.store(value, msg.get("as"))}
        if msg.get("ret") == "arrays":
            return _arrays(value, msg.get("out"))
        return self.enc(value, msg.get("ret") != "value")

    def op_free(self, msg: dict):
        names = list(self.handles) if msg.get("all") else list(msg.get("handles") or [msg.get("handle")])
        n = 0
        for name in names:
            if self.handles.pop(name, None) is not None:
                n += 1
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"freed": n}

    def op_memory(self, msg: dict):
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_available():
            return {"cuda": False}
        out = {"cuda": True, "allocated": int(torch.cuda.memory_allocated()),
               "reserved": int(torch.cuda.memory_reserved()), "max_allocated": int(torch.cuda.max_memory_allocated())}
        if msg.get("reset_peak"):
            torch.cuda.reset_peak_memory_stats()
        return out

    # -------------------------------------------------- streams
    def _control(self, sid) -> tuple[bool, set]:
        """Reads queued control messages for stream sid; others wait in self.pending."""
        cancel, drops = False, set()
        while True:
            try:
                m = self.control.get_nowait()
            except queue.Empty:
                break
            if m.get("stream") == sid and m.get("op") == "cancel":
                cancel = True
                self.send({"id": m.get("id"), "ok": True, "result": {"cancelled": True}})
            elif m.get("stream") == sid and m.get("op") == "drop":
                drops.add(m.get("obj_id"))
                self.send({"id": m.get("id"), "ok": True, "result": {"dropped": m.get("obj_id")}})
            else:
                self.pending.append(m)
        return cancel, drops

    def op_stream(self, msg: dict):
        """Generator stream ("method": items of an Ultralytics generator) or a feed stream ("feed": true)."""
        if msg.get("feed"):
            return self._feed_open(msg)
        rid = msg.get("id")
        target = self.obj(msg.get("handle"))
        it = self._path(target, msg["method"])(*self.dec(msg.get("args") or []), **self.dec(msg.get("kwargs") or {}))
        n, cancelled, dropped = 0, False, set()
        for item in it:
            c, d = self._control(rid)
            dropped |= d
            if c:
                cancelled = True
                break
            value = self.enc(item, False, results=msg.get("results"))
            if dropped and isinstance(value, dict):
                _drop_ids(value, dropped)
            self.send({"id": rid, "item": value, "n": n})
            n += 1
        close = getattr(it, "close", None)
        if callable(close):
            close()
        self.send({"id": rid, "ok": True, "result": {"done": True, "count": n, "cancelled": cancelled}})
        return None

    def _feed_open(self, msg: dict):
        """Incremental SAM 2 video segmentation: the caller feeds frames one at a time (see op_feed)."""
        import torch

        pred = self.obj(msg.get("handle"))
        sid = msg.get("id")
        with torch.inference_mode():
            pred.inference_state = pred._init_state(int(msg["num_frames"]))
            pred.backbone_out = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if msg.get("reset_peak"):
                torch.cuda.reset_peak_memory_stats()
        self.streams[sid] = {"pred": pred, "local": -1, "n_obj": int(msg["objects"]), "dropped": set(),
                             "pool": ThreadPoolExecutor(max_workers=2), "pre": {}}
        return {"stream": sid, "imgsz": [int(v) for v in pred.imgsz], "device": str(pred.device)}

    @staticmethod
    def _read(path: str):
        import cv2

        np = _np()
        try:
            data = np.fromfile(path, dtype=np.uint8)
        except OSError:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None

    def op_feed(self, msg: dict):
        """One frame: with "prompts" it becomes a conditioning frame, otherwise it is tracked from memory.

        Returns binary masks (N, h, w) at the predictor's letterboxed resolution (no padding), the object
        ids, and the object presence scores (sigmoid of object_score_logits; 1.0 on conditioning frames).
        """
        import torch
        import torch.nn.functional as F

        st = self.streams.get(msg.get("stream"))
        if st is None:
            raise BridgeError("unknown_stream", str(msg.get("stream")))
        pred = st["pred"]
        np = _np()
        frame = msg["frame"]
        if isinstance(frame, str):
            fut = st["pre"].pop(frame, None)
            img = fut.result() if fut is not None else self._read(frame)
        else:
            img = self.dec(frame)
        for p in msg.get("prefetch") or []:
            if p not in st["pre"]:
                st["pre"][p] = st["pool"].submit(self._read, p)
        if img is None:
            raise BridgeError("frame_unreadable", str(frame))
        h, w = img.shape[:2]
        st["local"] += 1
        local = st["local"]
        with torch.inference_mode():
            pred.backbone_out = None
            im = pred.preprocess([img])
            pred.inference_state["im"] = im
            pred.backbone_out = pred.model.forward_image(im)
            prompts = msg.get("prompts")
            if prompts:
                for pr in prompts:
                    obj = int(pr["obj_id"])
                    if pr.get("masks") is not None:
                        mask = self.dec(pr["masks"])
                        _, _, m = pred._prepare_prompts(pred.imgsz, (h, w), masks=np.asarray(mask)[None].astype(np.uint8))
                        pred.add_new_prompts(obj_id=obj, masks=m[[0]][:, None], frame_idx=local)
                        continue
                    bb = np.asarray(pr["bboxes"], dtype=np.float32).reshape(1, 4) if pr.get("bboxes") else None
                    pts = lab = None
                    if pr.get("points"):
                        pts = np.asarray(pr["points"], dtype=np.float32).reshape(1, -1, 2)
                        lab = np.asarray(pr["labels"], dtype=np.int32).reshape(1, -1)
                    p, l, _ = pred._prepare_prompts(pred.imgsz, (h, w), bboxes=bb, points=pts, labels=lab)
                    pred.add_new_prompts(obj_id=obj, points=p, labels=l, frame_idx=local)
                pred.propagate_in_video_preflight()
                out = pred.inference_state["output_dict"]["cond_frame_outputs"][local]
                low = out["pred_masks"]
                scores = torch.full((low.shape[0],), 1.0)
            else:
                state = pred.inference_state
                out = pred._run_single_frame_inference(
                    output_dict=state["output_dict"], frame_idx=local, batch_size=st["n_obj"],
                    is_init_cond_frame=False, point_inputs=None, mask_inputs=None, reverse=False,
                    run_mem_encoder=True)
                out.pop("pred_masks_high_res", None)
                state["output_dict"]["non_cond_frame_outputs"][local] = out
                state["frames_already_tracked"].add(local)
                pred._prune_non_cond_memory(local)
                low = out["pred_masks"]
                scores = out["object_score_logits"].float().sigmoid().flatten().cpu()
            size = int(pred.imgsz[0])
            g = min(size / h, size / w)
            nh, nw = int(round(h * g)), int(round(w * g))
            m = F.interpolate(low.float(), size=(size, size), mode="bilinear", align_corners=False)
            masks = (m[:, 0, :nh, :nw] > 0).cpu().numpy()
        keep = [i for i in range(masks.shape[0]) if i not in st["dropped"]]
        return {"frame_idx": local, "orig_shape": [h, w], "obj_ids": keep,
                "scores": [float(scores[i]) for i in keep],
                "masks": {"$rle": rle_encode(masks[keep] if keep else np.zeros((0, nh, nw), bool))}}

    def op_drop(self, msg: dict):
        st = self.streams.get(msg.get("stream"))
        if st is None:
            raise BridgeError("unknown_stream", str(msg.get("stream")))
        st["dropped"].add(int(msg["obj_id"]))
        return {"dropped": int(msg["obj_id"])}

    def op_cancel(self, msg: dict):
        st = self.streams.pop(msg.get("stream"), None)
        if st is None:
            return {"cancelled": False}
        st["pool"].shutdown(wait=False, cancel_futures=True)
        pred = st["pred"]
        pred.inference_state = {}
        pred.backbone_out = None
        out = {"cancelled": True}
        torch = sys.modules.get("torch")
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            out["max_allocated"] = int(torch.cuda.max_memory_allocated())
            torch.cuda.empty_cache()
        return out

    # -------------------------------------------------- checkpoints
    def op_ckpt_read(self, msg: dict):
        """Fields of a .pt checkpoint as Ultralytics loads it."""
        from ultralytics import YOLO

        model = YOLO(msg["path"])
        inner = getattr(model, "model", None)
        ckpt = getattr(model, "ckpt", None) or {}
        head = None
        try:
            head = inner.model[-1]
        except (AttributeError, IndexError, TypeError):
            pass
        out = {"task": getattr(model, "task", None), "model_class": type(inner).__name__ if inner is not None else "",
               "names": self.enc(getattr(inner, "names", None), False),
               "yaml": self.enc(getattr(inner, "yaml", None), False),
               "yaml_file": str(getattr(inner, "yaml_file", "") or ""),
               "args": self.enc(getattr(inner, "args", None), False),
               "args_type": type(getattr(inner, "args", None)).__name__,
               "train_args": self.enc(ckpt.get("train_args") if isinstance(ckpt, dict) else None, False),
               "ckpt_keys": sorted(str(k) for k in ckpt) if isinstance(ckpt, dict) else [],
               "end2end": self.enc(getattr(inner, "end2end", None), False),
               "head": None}
        if head is not None:
            out["head"] = {"type": type(head).__name__,
                           "modules": [n for n, _ in head.named_children()] if hasattr(head, "named_children") else [],
                           **{a: self.enc(getattr(head, a), False) for a in ("nc", "no", "reg_max", "end2end", "kpt_shape")
                              if hasattr(head, a)}}
        return out

    def op_ckpt_write(self, msg: dict):
        """Loads path, optionally reorders the class outputs of the head, sets fields, clears checkpoint keys, saves."""
        from ultralytics import YOLO

        model = YOLO(msg["path"])
        inner = getattr(model, "model", None)
        if inner is None:
            raise BridgeError("no_model", msg["path"])
        if msg.get("reorder_head") is not None:
            _reorder_head(inner.model[-1], [int(i) for i in msg["reorder_head"]])
        for path, value in (msg.get("set") or {}).items():
            *parents, last = path.split(".")
            target = self._path(inner, ".".join(parents)) if parents else inner
            value = self.dec(value)
            if isinstance(target, dict):
                target[last] = value
            else:
                setattr(target, last, value)
        ckpt = getattr(model, "ckpt", None)
        if isinstance(ckpt, dict):
            for key in msg.get("clear") or []:
                if key in ckpt:
                    ckpt[key] = None
        model.save(msg["out"])
        return {"out": msg["out"]}

    def op_reorder_head(self, msg: dict):
        if msg.get("order") is None:
            raise BridgeError("bad_request", "order")
        return self.op_ckpt_write({**msg, "reorder_head": msg["order"]})


def _drop_ids(value: dict, dropped: set) -> None:
    """Removes tracked objects whose id is in dropped from a serialized Results."""
    for key in ("boxes", "obb"):
        part = value.get(key)
        if not part or not part.get("id"):
            continue
        keep = [i for i, t in enumerate(part["id"]) if int(t) not in dropped]
        per = {"xyxyn": 4, "xywhr": 5, "xyxyxyxyn": 8, "conf": 1, "cls": 1, "id": 1}
        for k, step in per.items():
            if part.get(k) is not None:
                part[k] = [v for i in keep for v in part[k][i * step:(i + 1) * step]]
        part["n"] = len(keep)


def _reorder_head(det, order: list[int]) -> None:
    """Keeps / reorders the class channels of a Detect-style head: order[new] = old.

    The class branch is cv3 (and one2one_cv3 on end-to-end heads); the last Conv2d of each scale has nc
    output channels, which are indexed along dim 0. Box branches are untouched.
    """
    import torch

    nc = getattr(det, "nc", None)
    cv3 = getattr(det, "cv3", None)
    if nc is None or cv3 is None:
        raise BridgeError("unsupported_head", type(det).__name__)
    if not order or len(set(order)) != len(order) or any(i < 0 or i >= nc for i in order):
        raise BridgeError("bad_order", str(order))
    idx = torch.tensor(order, dtype=torch.long)
    new_nc = len(order)
    with torch.no_grad():
        for seq in list(cv3) + list(getattr(det, "one2one_cv3", None) or []):
            conv = seq[-1]
            if conv.weight.shape[0] != nc:
                raise BridgeError("unexpected_head_shape", str(tuple(conv.weight.shape)))
            if new_nc == nc:
                conv.weight.copy_(conv.weight[idx])
                if conv.bias is not None:
                    conv.bias.copy_(conv.bias[idx])
            else:
                conv.out_channels = new_nc
                conv.weight = torch.nn.Parameter(conv.weight[idx].clone())
                if conv.bias is not None:
                    conv.bias = torch.nn.Parameter(conv.bias[idx].clone())
    det.nc = new_nc
    if hasattr(det, "no"):
        det.no = new_nc + int(getattr(det, "reg_max", 0)) * 4


def _patch_multi_scale(min_size: int, max_size: int) -> None:
    """Compatibility patch, Ultralytics 8.4.x: DetectionTrainer.preprocess_batch picks the multi-scale size
    from imgsz +/- a ratio with no upper bound (imgsz=640 jitters 32..1312). This replaces it with a uniform
    pick in [min_size, max_size] (multiples of the stride). Applies to subclasses that inherit the method.
    """
    import math
    import random

    import torch
    from torch import nn
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    def preprocess_batch(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = random.randrange(min_size, max_size + self.stride) // self.stride * self.stride
            sz = max(self.stride, min(sz, max_size))
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    DetectionTrainer.preprocess_batch = preprocess_batch


# ---------------------------------------------------------------------- process
def _protocol_streams():
    """Moves the protocol pipes away from fd 0 / 1. fd 1 (including C-level output) goes to stderr and fd 0
    becomes the null device, so nothing else in the process can read the requests or write into the replies
    (a library touching sys.stdin while the reader thread blocks on it can deadlock)."""
    proto_out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1, newline="\n")
    proto_in = os.fdopen(os.dup(0), "rb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    sys.stdin = open(os.devnull, encoding="utf-8")
    return proto_in, proto_out


def _reader(bridge: Bridge, first: "queue.Queue", stream) -> None:
    """Request pipe -> requests. Control ops (cancel / drop for generator streams) go to the control queue."""
    for line in stream:
        raw = line.decode("utf-8", "replace").strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            bridge.send({"id": None, "ok": False, "error": {"code": "bad_json", "type": "", "message": raw[:200]}})
            continue
        if msg.get("op") in ("cancel", "drop") and msg.get("stream") in bridge._active_generators:
            bridge.control.put(msg)
        else:
            first.put(msg)
    first.put(None)


def main(argv: list[str]) -> int:
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass
    serve = bool(argv) and argv[0] == "serve"
    proto_in, proto_out = _protocol_streams()
    bridge = Bridge(proto_out)
    requests: "queue.Queue" = queue.Queue()
    threading.Thread(target=_reader, args=(bridge, requests, proto_in), daemon=True).start()
    bridge.send({"ready": True, "protocol_version": PROTOCOL_VERSION, "bridge": __version__})

    def run(msg: dict) -> bool:
        gen = msg.get("op") == "stream" and msg.get("method")
        if gen:
            bridge._active_generators.add(msg.get("id"))
        try:
            return bridge.handle(msg)
        finally:
            bridge._active_generators.discard(msg.get("id"))

    if not serve:
        msg = requests.get()
        if msg is None:
            return 2
        for op in (msg.get("ops") if isinstance(msg.get("ops"), list) else [msg]):
            if not run(op):
                return 1
        return 0
    while True:
        msg = bridge.pending.pop(0) if bridge.pending else requests.get()
        if msg is None:
            return 0
        run(msg)
        if msg.get("op") == "exit":
            return 0
