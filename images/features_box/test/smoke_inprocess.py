"""Ad-hoc in-process smoke test for the features service (no box build
needed): instantiate the servicer and drive Process() with envelopes.

    cd images/features_box && python test/smoke_inprocess.py

Needs numpy, opencv-python(-headless) and (for the .mat case) scipy locally.
"""
import io
import json
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "protos")

import numpy as np
import cv2
import features_service as svc
import pipeline_pb2
from aux import wrap_value, unwrap_value


def make_image(seed=0, size=(320, 240)):
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 220, size=(*size[::-1], 3), dtype=np.uint8)
    # some structure: blocks so SIFT has corners to find
    for _ in range(12):
        x, y = rng.integers(0, 160, 2)
        w, h = rng.integers(10, 60, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), 2)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def call(srv, config=None, images=None, names=None, method="Process"):
    data = {}
    if images is not None:
        data["images"] = wrap_value(images)
    if names is not None:
        data["names"] = wrap_value(names)
    req = pipeline_pb2.Envelope(
        config_json=json.dumps(config) if config else "",
        data=data)
    return getattr(srv, method)(req, None)


A = make_image(1)
B = make_image(2)

srv = svc.PipelineService()
failures = []


def section(resp):
    return (json.loads(resp.config_json or "{}")).get("features", {})


def check(name, cond, detail=""):
    print(f"  {'ok ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        failures.append(name)


def np_field(resp, field):
    return np.load(io.BytesIO(bytes(unwrap_value(resp.data[field]))))


print("== 1. extract, 2 images ==")
r = call(srv, {"features": {"command": "extract", "parameters": {}}}, [A, B])
s = section(r)
print("   ", {k: v for k, v in s.items() if k != "encoding"})
check("status done", s.get("status") == "done", str(s.get("error", "")))
for f in ("keypoints", "counts", "annotated"):
    check(f"field {f}", f in r.data)
kp = np_field(r, "keypoints")
counts = np_field(r, "counts")
print(f"      keypoints: {kp.shape} {kp.dtype}   counts: {counts.tolist()}")
check("keypoints rank 3", kp.ndim == 3, str(kp.shape))
check("per-image axis", kp.shape[0] == 2, str(kp.shape))
check("130 rows (2 + 128)", kp.shape[1] == 130, str(kp.shape))
check("N_max == max(counts)", kp.shape[2] == int(counts.max()),
      f"{kp.shape[2]} vs {counts.tolist()}")
check("counts length", counts.shape == (2,), str(counts.shape))
check("num_keypoints echoes counts",
      s.get("num_keypoints") == [int(c) for c in counts], str(s.get("num_keypoints")))
check("encoding keypoints", s.get("encoding", {}).get("keypoints") == "numpy")
check("encoding counts", s.get("encoding", {}).get("counts") == "numpy")
check("encoding annotated", s.get("encoding", {}).get("annotated") == "identity")
ann = unwrap_value(r.data["annotated"])
check("annotated count", len(ann) == 2, str(len(ann)))
dec = cv2.imdecode(np.frombuffer(bytes(ann[0]), np.uint8), cv2.IMREAD_COLOR)
check("annotated decodes", dec is not None and dec.shape[:2] == (240, 320),
      str(None if dec is None else dec.shape))
check("no mat by default", "mat" not in r.data)
check("feature reported", s.get("feature") == "SIFT")
check("padding is zero", float(np.abs(kp[int(np.argmin(counts)), :,
                                        int(counts.min()):]).sum()) == 0.0)

print("== 2. extract, 1 image, draw off ==")
r = call(srv, {"features": {"command": "extract",
                            "parameters": {"draw": False, "nfeatures": 50}}}, [A])
s = section(r)
check("status done", s.get("status") == "done", str(s.get("error", "")))
check("no annotated", "annotated" not in r.data)
kp1 = np_field(r, "keypoints")
print("      keypoints:", kp1.shape)
check("single-image axis", kp1.shape[0] == 1, str(kp1.shape))
# OpenCV's retainBest keeps every keypoint tied with the last kept one, so
# the cap is "about nfeatures", not exactly it (51 for 50 is normal).
check("nfeatures caps the count", kp1.shape[2] <= 60, str(kp1.shape))

print("== 2b. legacy call without command ==")
r = call(srv, {"features": {"parameters": {"draw": "false"}}}, [A])
check("no-command defaults to extract", section(r).get("status") == "done")
check("string 'false' parsed as bool", "annotated" not in r.data)

print("== 3. mat payload + names ==")
try:
    import scipy.io as sio
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("  SKIPPED — scipy not installed locally (the box image ships it)")

r = call(srv, {"features": {"command": "extract",
                            "parameters": {"mat": HAVE_SCIPY, "nfeatures": 30}}},
         [A, B], names=["left.png", "right.png"])
s = section(r)
check("status done", s.get("status") == "done", str(s.get("error", "")))
check("mat field", ("mat" in r.data) == HAVE_SCIPY)
check("files echoed", s.get("files") == ["left.mat", "right.mat"], str(s.get("files")))
if "mat" in r.data:
    blobs = unwrap_value(r.data["mat"])
    check("one mat per image", len(blobs) == 2, str(len(blobs)))
    m = sio.loadmat(io.BytesIO(bytes(blobs[0])))
    check("mat has kp key", "kp" in m, str(sorted(k for k in m if not k.startswith("__"))))
    kp_mat = m["kp"]
    print("      mat kp:", kp_mat.shape, kp_mat.dtype)
    check("mat kp is 130 x N", kp_mat.shape[0] == 130, str(kp_mat.shape))
    kp_npy = np_field(r, "keypoints")
    n0 = int(np_field(r, "counts")[0])
    check("mat matches keypoints slice",
          np.allclose(kp_mat, kp_npy[0, :, :n0]))

print("== 4. names/images mismatch -> error ==")
r = call(srv, {"features": {"command": "extract"}}, [A, B], names=["only_one.jpg"])
check("names mismatch error", section(r).get("status") == "error",
      str(section(r).get("error")))

print("== 5. missing images -> empty_request ==")
r = call(srv, {"features": {"command": "extract"}}, None)
check("empty_request", section(r).get("status") == "empty_request", str(section(r)))

print("== 6. bad config -> error ==")
r = call(srv, {"opencv": {"command": "match"}}, [A])
check("wrong namespace error", section(r).get("status") == "error")
r = call(srv, None, [A])
check("no config error", section(r).get("status") == "error")
r = call(srv, {"features": {"command": "explode"}}, [A])
check("unknown command error", section(r).get("status") == "error"
      and "explode" in section(r).get("error", ""))
r = call(srv, {"features": {"command": "extract",
                            "parameters": {"nfeatures": "lots"}}}, [A])
check("bad param error", section(r).get("status") == "error",
      str(section(r).get("error")))
r = call(srv, {"features": {"command": "extract", "parameters": {"draw": "maybe"}}}, [A])
check("bad bool error", section(r).get("status") == "error",
      str(section(r).get("error")))

print("== 7. reset ==")
r = call(srv, {"features": {"command": "reset"}})
s = section(r)
check("reset done", s.get("status") == "done" and s.get("action") == "reset", str(s))

print("== 8. bad frame -> error ==")
r = call(srv, {"features": {"command": "extract"}}, [b"not an image"])
check("error on undecodable", section(r).get("status") == "error",
      str(section(r).get("error")))

print("== 9. featureless image -> empty, well-formed result ==")
blank = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))[1].tobytes()
r = call(srv, {"features": {"command": "extract"}}, [blank])
s = section(r)
check("status done", s.get("status") == "done", str(s.get("error", "")))
kp0 = np_field(r, "keypoints")
check("empty (1, 130, 0)", kp0.shape == (1, 130, 0), str(kp0.shape))
check("counts zero", np_field(r, "counts").tolist() == [0])

print()
print("FAIL -- " + "; ".join(failures) if failures else "PASS -- all in-process cases ok")
sys.exit(1 if failures else 0)
