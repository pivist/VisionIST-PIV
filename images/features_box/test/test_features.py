#!/usr/bin/env python3
"""Test script for the features gRPC service (shared envelope interface).

Connects to a running features box and:
  1. ``extract`` the two bundled test images -> ``keypoints`` (the CLI's
     ``(2 + 128, N)`` layout, padded & stacked), ``counts`` and the red-dot
     ``annotated`` JPEGs
  2. verifies the declared ``encoding`` map in the response config
  3. ``mat`` payload: the ``.mat`` bytes load back under the key ``kp`` and
     agree with the ``keypoints`` array, and ``data.names`` drives the
     reported file names
  4. single image, ``draw: false``, and the legacy no-command form
  5. ``empty_request`` / ``error`` contract (missing images, unknown command,
     bad parameters, name/image mismatch, undecodable frame)
  6. ``command: reset`` as a standard no-op

Run (from the repo or image root, server already up):
    python images/features_box/test/test_features.py
    BOX_HOST=10.0.0.5:8061 python images/features_box/test/test_features.py

Note: no box build required — ``smoke_inprocess.py`` drives the same
service code in-process (needs cv2 locally).
"""

import io
import json
import os
import sys

# Make the protos/ folder importable (same pattern as the opencv test).
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_TEST_DIR, "..", "protos"))

import grpc  # noqa: E402
import aux  # noqa: E402
import pipeline_pb2  # noqa: E402
import pipeline_pb2_grpc  # noqa: E402

import numpy as np  # noqa: E402

_DESCRIPTOR_ROWS = 2 + 128  # x, y + SIFT descriptor


def load_local_image(path: str) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    print(f"loaded {os.path.basename(path)}: {len(data) / (1024*1024):.2f} MB")
    return data


def make_stub(target: str):
    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    return pipeline_pb2_grpc.PipelineServiceStub(channel), channel


def check_status(response):
    cfg = json.loads(response.config_json or "{}")
    section = cfg.get("features")
    if not isinstance(section, dict):
        print(f"  ERROR: response config not namespaced under 'features': {cfg}")
        return None
    if section.get("status") == "error":
        print(f"  ERROR: {section.get('error')}")
    return section


def decode_np(value) -> np.ndarray:
    """The box declares these ``numpy`` (np.save blobs): decode back to
    an array, shape and dtype intact."""
    return np.load(io.BytesIO(bytes(value)), allow_pickle=False)


def main():
    target = os.getenv("BOX_HOST", "localhost:8061")
    print(f"Target: {target}")
    stub, channel = make_stub(target)

    failures = []
    img_00 = load_local_image(os.path.join(_TEST_DIR, "00.jpg"))
    img_01 = load_local_image(os.path.join(_TEST_DIR, "01.jpg"))

    # ------------------------------------------------------------------ #
    # Case 1: extract — two images -> keypoints + counts + annotated     #
    # ------------------------------------------------------------------ #
    print("\n== case 1: extract, 2 images ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"features": {"command": "extract",
                                             "parameters": {}}}),
        data={"images": aux.wrap_value([img_00, img_01])},
    ))
    section = check_status(response)
    if section is None:
        return 1
    if section.get("status") != "done":
        print(f"  status not done: {section}")
        failures.append("case1 status")
    for f in ("keypoints", "counts", "annotated"):
        if f not in response.data:
            print(f"  missing field: {f}")
            failures.append(f"case1 {f} missing")
    enc = section.get("encoding", {})
    for f, codec in (("keypoints", "numpy"), ("counts", "numpy"),
                     ("annotated", "identity")):
        if enc.get(f) != codec:
            print(f"  {f} not declared {codec}: {enc.get(f)!r}")
            failures.append(f"case1 encoding {f}")

    keypoints = counts = None
    if "keypoints" in response.data and "counts" in response.data:
        keypoints = decode_np(aux.unwrap_value(response.data["keypoints"]))
        counts = decode_np(aux.unwrap_value(response.data["counts"]))
        print(f"  keypoints : shape={keypoints.shape} dtype={keypoints.dtype}")
        print(f"  counts    : {counts.tolist()}")
        if keypoints.ndim != 3 or keypoints.shape[0] != 2:
            failures.append("case1 keypoints per-image axis")
        if keypoints.shape[1] != _DESCRIPTOR_ROWS:
            print(f"  keypoints rows {keypoints.shape[1]} != {_DESCRIPTOR_ROWS}")
            failures.append("case1 keypoints rows")
        if counts.shape != (2,) or keypoints.shape[2] != int(counts.max()):
            failures.append("case1 counts/padding")
        if section.get("num_keypoints") != [int(c) for c in counts]:
            failures.append("case1 num_keypoints")

    if "annotated" in response.data:
        ann = aux.unwrap_value(response.data["annotated"])
        print(f"  annotated : {len(ann)} JPEGs, "
              f"{[round(len(a)/1024) for a in ann]} kB")
        if len(ann) != 2:
            failures.append("case1 annotated count")
    print(f"  runtime={section.get('runtime'):.3f}s feature={section.get('feature')}")

    # ------------------------------------------------------------------ #
    # Case 2: mat payload + names                                        #
    # ------------------------------------------------------------------ #
    print("\n== case 2: mat payload + names ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"features": {
            "command": "extract",
            "parameters": {"mat": True, "draw": False, "nfeatures": 200}}}),
        data={"images": aux.wrap_value([img_00, img_01]),
              "names": aux.wrap_value(["00.jpg", "01.jpg"])},
    ))
    section = check_status(response)
    if section is None or section.get("status") != "done":
        failures.append("case2 status")
    else:
        if section.get("files") != ["00.mat", "01.mat"]:
            print(f"  files not echoed from names: {section.get('files')}")
            failures.append("case2 files")
        if "annotated" in response.data:
            failures.append("case2 draw:false still annotated")
        if "mat" not in response.data:
            failures.append("case2 mat missing")
        else:
            blobs = aux.unwrap_value(response.data["mat"])
            print(f"  mat       : {len(blobs)} files, "
                  f"{[round(len(b)/1024) for b in blobs]} kB")
            try:
                import scipy.io as sio
                m = sio.loadmat(io.BytesIO(bytes(blobs[0])))
                kp_mat = m.get("kp")
                print(f"  mat['kp'] : shape={None if kp_mat is None else kp_mat.shape}")
                if kp_mat is None or kp_mat.shape[0] != _DESCRIPTOR_ROWS:
                    failures.append("case2 mat kp layout")
                else:
                    kp = decode_np(aux.unwrap_value(response.data["keypoints"]))
                    n0 = int(decode_np(aux.unwrap_value(response.data["counts"]))[0])
                    if not np.allclose(kp_mat, kp[0, :, :n0]):
                        failures.append("case2 mat vs keypoints")
            except ImportError:
                print("  (scipy not installed locally — .mat content not checked)")

    # ------------------------------------------------------------------ #
    # Case 3: single image, draw off + legacy no-command form            #
    # ------------------------------------------------------------------ #
    print("\n== case 3: extract, 1 image ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"features": {"command": "extract",
                                             "parameters": {"nfeatures": 100}}}),
        data={"images": aux.wrap_value([img_00])},
    ))
    section = check_status(response)
    if section is None or section.get("status") != "done":
        failures.append("case3 status")
    else:
        arr = decode_np(aux.unwrap_value(response.data["keypoints"]))
        print(f"  keypoints : shape={arr.shape}")
        if arr.shape[0] != 1:
            failures.append("case3 keypoints shape")
        if section.get("files") != ["image_000.mat"]:
            failures.append("case3 default file name")

    print("\n== case 3b: legacy call without command (compat) ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"features": {"parameters": {"draw": "false"}}}),
        data={"images": aux.wrap_value([img_00])},
    ))
    section = check_status(response)
    if section is None or section.get("status") != "done":
        failures.append("case3b legacy")
    elif "annotated" in response.data:
        failures.append("case3b draw string flag")

    # ------------------------------------------------------------------ #
    # Case 4: error / empty_request contract                              #
    # ------------------------------------------------------------------ #
    print("\n== case 4: empty_request + errors ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"features": {"command": "extract"}})))
    section = check_status(response)
    if section is None or section.get("status") != "empty_request":
        failures.append("case4 empty_request")

    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"features": {"command": "explode"}}),
        data={"images": aux.wrap_value([img_00])},
    ))
    section = check_status(response)
    if section is None or section.get("status") != "error" \
            or "explode" not in str(section.get("error")):
        failures.append("case4 unknown command")

    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"features": {"command": "extract",
                                             "parameters": {"nfeatures": "lots"}}}),
        data={"images": aux.wrap_value([img_00])},
    ))
    section = check_status(response)
    if section is None or section.get("status") != "error":
        failures.append("case4 bad parameter")

    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"features": {"command": "extract"}}),
        data={"images": aux.wrap_value([img_00, img_01]),
              "names": aux.wrap_value(["only_one.jpg"])},
    ))
    section = check_status(response)
    if section is None or section.get("status") != "error":
        failures.append("case4 names mismatch")

    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"features": {"command": "extract"}}),
        data={"images": aux.wrap_value([b"not an image"])},
    ))
    section = check_status(response)
    if section is None or section.get("status") != "error":
        failures.append("case4 undecodable")

    # ------------------------------------------------------------------ #
    # Case 5: reset (standard no-op on this fully stateless box)          #
    # ------------------------------------------------------------------ #
    print("\n== case 5: reset command ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"features": {"command": "reset"}})))
    section = check_status(response)
    if section is None or section.get("status") != "done" \
            or section.get("action") != "reset":
        failures.append("case5 reset")
    else:
        print(f"  ok: {section}")

    channel.close()

    print("\n" + ("FAIL -- " + "; ".join(failures) if failures
                  else "PASS -- all features test cases."))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
