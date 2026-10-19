"""Protocol tests for ul_bridge.

ProtocolTest and EncodingTest need only NumPy. UltralyticsTest needs Ultralytics and PyTorch and is
skipped when they are not installed (as in CI); its checkpoint test also needs UL_BRIDGE_TEST_MODEL,
the path of a local .pt file (the bridge never downloads weights).

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import ul_bridge  # noqa: E402

HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None and importlib.util.find_spec("torch") is not None


def _env() -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", YOLO_AUTOINSTALL="false")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p)
    return env


class Serve:
    """python -m ul_bridge serve, one request per line."""

    def __init__(self) -> None:
        self.p = subprocess.Popen([sys.executable, "-m", "ul_bridge", "serve"], cwd=str(ROOT), env=_env(),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.ready = self.read()

    def read(self) -> dict:
        line = self.p.stdout.readline()
        if not line:
            raise AssertionError("bridge exited: " + self.p.stderr.read().decode("utf-8", "replace")[-2000:])
        return json.loads(line)

    def send(self, obj) -> None:
        self.p.stdin.write((obj if isinstance(obj, str) else json.dumps(obj)).encode("utf-8") + b"\n")
        self.p.stdin.flush()

    def ask(self, obj: dict) -> dict:
        self.send(obj)
        return self.read()

    def close(self) -> int:
        try:
            self.ask({"id": 0, "op": "exit"})
        finally:
            self.p.stdin.close()
            code = self.p.wait(timeout=60)
            self.p.stdout.close()
            self.p.stderr.close()
        return code


class ProtocolTest(unittest.TestCase):
    def test_serve_ready_and_errors(self) -> None:
        s = Serve()
        try:
            self.assertEqual(s.ready, {"ready": True, "protocol_version": ul_bridge.PROTOCOL_VERSION,
                                       "bridge": ul_bridge.__version__})
            r = s.ask({"id": 1, "op": "nope"})
            self.assertEqual((r["id"], r["ok"], r["error"]["code"]), (1, False, "unknown_op"))
            r = s.ask({"id": 2, "op": "load", "kind": "NotAClass"})
            self.assertEqual(r["error"]["code"], "unknown_kind")
            r = s.ask({"id": 3, "op": "get", "handle": "h99", "attr": "x"})
            self.assertEqual(r["error"]["code"], "unknown_handle")
            r = s.ask({"id": 4, "op": "drop", "stream": 123, "obj_id": 1})
            self.assertEqual(r["error"]["code"], "unknown_stream")
            r = s.ask({"id": 5, "op": "reorder_head", "path": "x.pt", "out": "y.pt"})
            self.assertEqual(r["error"]["code"], "bad_request")
            s.send("{not json")
            r = s.read()
            self.assertEqual((r["id"], r["error"]["code"]), (None, "bad_json"))
            r = s.ask({"id": 6, "op": "setup", "env": {"UL_BRIDGE_TEST_VAR": "1"}})
            self.assertTrue(r["ok"])
            r = s.ask({"id": 7, "op": "free", "all": True})
            self.assertEqual(r["result"], {"freed": 0})
            r = s.ask({"id": 8, "op": "memory"})
            self.assertEqual(r["result"], {"cuda": False})
        finally:
            self.assertEqual(s.close(), 0)

    def test_once_mode(self) -> None:
        ok = {"ops": [{"id": 1, "op": "setup"}, {"id": 2, "op": "free", "all": True}]}
        r = subprocess.run([sys.executable, "-m", "ul_bridge"], cwd=str(ROOT), env=_env(), capture_output=True,
                           input=json.dumps(ok).encode() + b"\n", timeout=60)
        lines = [json.loads(x) for x in r.stdout.decode().splitlines()]
        self.assertEqual(r.returncode, 0)
        self.assertTrue(lines[0]["ready"])
        self.assertEqual([x["id"] for x in lines[1:]], [1, 2])
        bad = {"ops": [{"id": 1, "op": "nope"}, {"id": 2, "op": "setup"}]}
        r = subprocess.run([sys.executable, "-m", "ul_bridge"], cwd=str(ROOT), env=_env(), capture_output=True,
                           input=json.dumps(bad).encode() + b"\n", timeout=60)
        lines = [json.loads(x) for x in r.stdout.decode().splitlines()]
        self.assertEqual(r.returncode, 1)
        self.assertEqual([x.get("id") for x in lines[1:]], [1])


class EncodingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.out = io.StringIO()
        self.b = ul_bridge.Bridge(self.out)

    def test_rle_roundtrip(self) -> None:
        m = np.zeros((2, 5, 7), dtype=bool)
        m[0, 1:3, 2:6] = True
        m[1, 0, 0] = True
        m[1, 4, 6] = True
        enc = ul_bridge.rle_encode(m)
        self.assertEqual(enc["shape"], [2, 5, 7])
        self.assertEqual(enc["counts"][1][0], 0)       # starts with a run of ones -> leading zero-length run
        self.assertTrue(np.array_equal(ul_bridge.rle_decode(enc), m))

    def test_values(self) -> None:
        small = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.assertEqual(self.b.enc(small), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        big = np.arange(5000, dtype=np.int32)
        e = self.b.enc(big)
        self.assertEqual(set(e), {"$nd"})
        self.assertTrue(np.array_equal(self.b.dec(e), big))
        mask = np.eye(4, dtype=bool)
        self.assertEqual(set(self.b.enc(mask)), {"$rle"})
        self.assertTrue(np.array_equal(self.b.dec(self.b.enc(mask)), mask))
        self.assertEqual(self.b.enc({0: "a", 2: "b"}), {"$dict": [[0, "a"], [2, "b"]]})
        self.assertEqual(self.b.dec({"$dict": [[0, "a"]]}), {0: "a"})
        self.assertEqual(self.b.dec({"$tuple": [1, 2]}), (1, 2))
        self.assertEqual(self.b.enc(np.float32(1.5)), 1.5)
        self.assertEqual(self.b.dec({"$nd": {"dtype": "int64", "value": [1, 2]}}).tolist(), [1, 2])

    def test_handles(self) -> None:
        obj = object()
        e = self.b.enc(obj)
        self.assertEqual(set(e), {"$h"})
        self.assertIs(self.b.dec(e), obj)
        self.assertEqual(self.b.enc(obj, handles=False), repr(obj))

    def test_fields(self) -> None:
        arg = types.SimpleNamespace(epoch=3, loss=np.zeros(3), metrics={"m": 0.5}, items=[1, 2, 3],
                                    name=lambda: "x")
        self.assertEqual(self.b.field(arg, "epoch"), 3)
        self.assertEqual(self.b.field(arg, "len(items)"), 3)
        self.assertEqual(self.b.field(arg, "type(loss)"), "ndarray")
        self.assertEqual(self.b.field(arg, "metrics.m"), 0.5)
        self.assertEqual(self.b.field(arg, "items.-1"), 3)
        self.assertEqual(self.b.field(arg, "name()"), "x")
        self.assertIsNone(self.b.field(arg, "missing.attr"))
        big = np.zeros(1000)
        self.assertEqual(self.b.enc(big, False, ul_bridge._EVENT_SMALL), {"$shape": [1000], "dtype": "float64"})

    def test_drop_ids(self) -> None:
        value = {"boxes": {"n": 2, "xyxyn": [0, 0, 1, 1, 0, 0, 0.5, 0.5], "conf": [0.9, 0.8], "cls": [0, 1],
                           "id": [7, 8]}}
        ul_bridge._drop_ids(value, {7})
        self.assertEqual(value["boxes"], {"n": 1, "xyxyn": [0, 0, 0.5, 0.5], "conf": [0.8], "cls": [1], "id": [8]})


@unittest.skipUnless(HAS_ULTRALYTICS, "needs ultralytics and torch")
class UltralyticsTest(unittest.TestCase):
    def test_env(self) -> None:
        s = Serve()
        try:
            r = s.ask({"id": 1, "op": "env"})
            self.assertTrue(r["ok"], r)
            self.assertEqual(r["result"]["protocol_version"], ul_bridge.PROTOCOL_VERSION)
            self.assertTrue(r["result"]["ultralytics"])
        finally:
            s.close()

    @unittest.skipUnless(os.environ.get("UL_BRIDGE_TEST_MODEL"), "set UL_BRIDGE_TEST_MODEL to a local .pt file")
    def test_load_and_ckpt_read(self) -> None:
        model = os.environ["UL_BRIDGE_TEST_MODEL"]
        s = Serve()
        try:
            r = s.ask({"id": 1, "op": "load", "kind": "YOLO", "model": model})
            self.assertTrue(r["ok"], r)
            h = r["result"]["handle"]
            r = s.ask({"id": 2, "op": "get", "handle": h, "attr": "task"})
            self.assertIn(r["result"], ("detect", "segment", "pose", "obb", "classify"))
            r = s.ask({"id": 3, "op": "ckpt_read", "path": model})
            self.assertTrue(r["ok"], r)
            self.assertIn("names", r["result"])
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
