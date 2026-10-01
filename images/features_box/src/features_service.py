"""features box — SIFT keypoint extraction behind the shared envelope.

A port of the **SIFT Extractor** CLI (https://github.com/pivist/features) to
the fleet contract: same detector, same ``kp`` layout, same red-dot annotated
image — but driven over gRPC instead of a mounted input/output folder.

A standard shared-envelope box (one RPC: ``Process``) with two commands in
the ``config_json`` section ``{"features": {"command": ..., "parameters": {...}}}``:

* **``extract``** (the default when no command is given) — run SIFT on each
  input image and return, per image, the original tool's combined
  ``(2 + 128, N)`` array (row 0 = x, row 1 = y, rows 2..129 = the descriptor),
  optionally the annotated JPEG and the byte-identical ``.mat`` file.
  Stateless.
* **``reset``** — accepted by every standard box; here a plain no-op, since
  ``extract`` is stateless.

Contract (see docs/gRPC_Services_Reference.md):

* response ``config_json`` is **namespaced**: ``{"features": {"status": …}}``
  with ``status`` in ``done | empty_request | error`` (the human reason in
  ``"error"`` on failure), plus ``runtime`` and box-specific fields.
* the response declares its payload encoding (``"encoding": {field: codec}``):
  ``keypoints`` / ``counts`` are ``np.save`` (``.npy``) blobs declared
  ``numpy`` — ``visionist_client`` decodes them to ``np.ndarray`` keeping
  their shape — while ``annotated`` and ``mat`` are opaque file bytes
  declared ``identity``.
* pure CPU: OpenCV's SIFT needs no GPU, so this box has no device parameter
  and no idle watchdog (nothing to park).

Fidelity notes (kept deliberately identical to the CLI):

* the detector is fed ``cv2.cvtColor(img, cv2.COLOR_BGR2RGB)``, exactly as
  ``main.py`` does.  SIFT greys the frame internally with the BGR weights,
  so feeding it RGB swaps the red and blue weights — a quirk, but changing
  it would move the keypoints relative to the original tool's output.
* ``keypoints`` stacks ``np.concatenate((pts.T, des.T), axis=0)`` — the very
  array the CLI writes to ``<name>.mat`` under the key ``kp``.
* the annotated image draws a filled radius-2 red circle at each rounded
  keypoint centre, on the *BGR* frame, like ``main.py``.
"""

import concurrent.futures as futures
import io
import json
import logging
import os
import sys
import time

sys.path.append("./protos")
import pipeline_pb2 as folder_wd_pb2  # noqa: E402
import pipeline_pb2_grpc as folder_wd_pb2_grpc  # noqa: E402
from aux import wrap_value, unwrap_value  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT_ENV_VAR = 'PORT'

_DESCRIPTOR_LENGTH = 128  # SIFT
_MAT_KEY = "kp"           # the key main.py writes into the .mat

# Per-command parameter defaults (``parameters`` in the config section).
# The SIFT ones mirror ``cv2.SIFT_create()``'s own defaults, which is what
# the original ``main.py`` used (it passed no arguments at all).
_EXTRACT_DEFAULTS = {
    "nfeatures": 0,               # 0 = keep every keypoint SIFT finds
    "n_octave_layers": 3,
    "contrast_threshold": 0.04,
    "edge_threshold": 10.0,
    "sigma": 1.6,
    "draw": True,                 # return the red-dot annotated image
    "radius": 2,                  # annotation dot radius, as in main.py
    "mat": False,                 # also return the .mat file bytes
}


def _done(extra, data=None, encoding=None):
    """Standard success envelope: namespaced status + declared encoding."""
    section = {"status": "done", **extra}
    if encoding:
        section["encoding"] = encoding
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({"features": section}),
        data=data or {},
    )


def _empty_request():
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({"features": {"status": "empty_request"}}))


def _error(message):
    return folder_wd_pb2.Envelope(
        config_json=json.dumps({"features": {"status": "error", "error": message}}))


# ---------------------------------------------
# Helpers
# ---------------------------------------------
def np_to_bytes(arr: np.ndarray) -> bytes:
    """Serialize with np.save — keeps the array's dtype and shape in the
    blob, so the client's ``numpy`` codec (``np.load``) restores it exactly."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr))
    return buf.getvalue()


def pad_and_stack(arrays, pad_value=0.0):
    """Pad a list of 2-D arrays to the same shape and stack on axis 0."""
    if not arrays:
        return np.zeros((0, 0), dtype=np.float32)
    max_shape = np.max([np.array(a.shape) for a in arrays], axis=0)
    padded = []
    for a in arrays:
        pad_width = [(0, int(m - s)) for s, m in zip(a.shape, max_shape)]
        padded.append(np.pad(a, pad_width, mode='constant', constant_values=pad_value))
    return np.stack(padded, axis=0)


def _num(parameters, key, default, cast, non_negative=True):
    """Coerce ``parameters[key]`` (falling back to ``default``) to a number."""
    v = parameters.get(key, default)
    try:
        v = cast(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"parameters.{key} must be a number, got {v!r}") from e
    if non_negative and v < 0:
        raise ValueError(f"parameters.{key} must be >= 0, got {v!r}")
    return v


def _flag(parameters, key, default):
    """Coerce ``parameters[key]`` to a bool, accepting JSON's ``true`` and the
    string forms the webui's select widgets send (``"true"`` / ``"false"``)."""
    v = parameters.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
    raise ValueError(f"parameters.{key} must be a boolean, got {v!r}")


def _parse_extract_parameters(parameters):
    d = _EXTRACT_DEFAULTS
    spec = {
        "nfeatures": _num(parameters, "nfeatures", d["nfeatures"], int),
        "n_octave_layers": _num(parameters, "n_octave_layers",
                                d["n_octave_layers"], int),
        "contrast_threshold": _num(parameters, "contrast_threshold",
                                   d["contrast_threshold"], float),
        "edge_threshold": _num(parameters, "edge_threshold",
                               d["edge_threshold"], float),
        "sigma": _num(parameters, "sigma", d["sigma"], float),
        "draw": _flag(parameters, "draw", d["draw"]),
        "radius": _num(parameters, "radius", d["radius"], int),
        "mat": _flag(parameters, "mat", d["mat"]),
    }
    if spec["n_octave_layers"] < 1:
        raise ValueError("parameters.n_octave_layers must be >= 1")
    if spec["sigma"] <= 0:
        raise ValueError("parameters.sigma must be > 0")
    return spec


def _stem(name, index):
    """``dog.jpg`` -> ``dog``; a missing/blank name -> ``image_000``."""
    base = os.path.basename(str(name or "")).strip()
    stem = os.path.splitext(base)[0]
    return stem or f"image_{index:03d}"


def _savemat_bytes(arr: np.ndarray) -> bytes:
    """The CLI's ``scipy.io.savemat(path, {'kp': combined})``, to memory."""
    import scipy.io as sio  # local: only needed when parameters.mat is on

    buf = io.BytesIO()
    sio.savemat(buf, {_MAT_KEY: arr})
    return buf.getvalue()


# ---------------------------------------------
# Service Definition
# ---------------------------------------------
class PipelineService(folder_wd_pb2_grpc.PipelineServiceServicer):
    """Stateless: no model to load, no device to hold, no session state."""

    # ------------------------------------------------------------- inputs
    @staticmethod
    def _decode_images(image_bytes_list):
        """JPEG/PNG bytes -> BGR ndarray list; raises on any bad frame."""
        frames = []
        for i, raw in enumerate(image_bytes_list):
            arr = np.frombuffer(bytes(raw), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"could not decode image #{i + 1} of "
                                 f"{len(image_bytes_list)} (not a supported raster?)")
            frames.append(img)
        return frames

    # ---------------------------------------------------------- extraction
    @staticmethod
    def _sift(spec):
        return cv2.SIFT_create(
            nfeatures=spec["nfeatures"],
            nOctaveLayers=spec["n_octave_layers"],
            contrastThreshold=spec["contrast_threshold"],
            edgeThreshold=spec["edge_threshold"],
            sigma=spec["sigma"],
        )

    @classmethod
    def _extract_one(cls, img_bgr, detector, spec):
        """One image -> (combined (130, N) float32, annotated JPEG or None).

        ``combined`` is exactly what the CLI writes to ``<name>.mat``:
        ``np.concatenate((pts.T, des.T), axis=0)``."""
        # Faithful to main.py: the detector is handed the RGB frame.
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        kps, desc = detector.detectAndCompute(img_rgb, None)

        if kps and desc is not None and len(desc):
            pts = np.array([k.pt for k in kps], dtype=np.float32).T      # (2, N)
            combined = np.concatenate((pts, desc.T.astype(np.float32)),  # (130, N)
                                      axis=0)
        else:
            # main.py would have raised here (des is None); the box answers
            # with a well-formed empty result instead.
            kps = []
            combined = np.zeros((2 + _DESCRIPTOR_LENGTH, 0), dtype=np.float32)

        annotated = None
        if spec["draw"]:
            canvas = img_bgr.copy()
            for k in kps:
                cx, cy = int(k.pt[0]), int(k.pt[1])
                cv2.circle(canvas, (cx, cy), spec["radius"], (0, 0, 255), -1)
            ok, buf = cv2.imencode(".jpg", canvas)
            if not ok:
                raise ValueError("failed to encode the annotated image")
            annotated = buf.tobytes()

        return combined, annotated

    # ---------------------------------------------------------------- Process
    def Process(self, request, context):
        start_time = time.time()

        try:
            if not request.config_json:
                return _error("No config JSON")

            config = json.loads(request.config_json)
            if not isinstance(config, dict) or not isinstance(config.get("features"), dict):
                return _error("config section 'features' missing or not an object")
            section = config["features"]
            command = section.get("command") or "extract"
            parameters = section.get("parameters", {})
            if not isinstance(parameters, dict):
                return _error("parameters must be an object")

            # --- reset: plain no-op (this box is fully stateless) ---------
            if command == "reset":
                return _done({"action": "reset"})

            if command != "extract":
                return _error(
                    f"unknown command {command!r} (expected 'extract' or 'reset')")

            images = unwrap_value(request.data["images"]) if "images" in request.data else None
            if isinstance(images, (bytes, bytearray)):
                images = [images]
            if not images:
                return _empty_request()

            names = unwrap_value(request.data["names"]) if "names" in request.data else None
            if isinstance(names, str):
                names = [names]
            if names is not None and len(names) != len(images):
                return _error(f"data.names has {len(names)} entries for "
                              f"{len(images)} images")

            spec = _parse_extract_parameters(parameters)
            imgs_in = self._decode_images(images)

            # ---------------------------------------------------------- run
            detector = self._sift(spec)
            combined_list, annotated_list = [], []
            for img in imgs_in:
                combined, annotated = self._extract_one(img, detector, spec)
                combined_list.append(combined)
                if annotated is not None:
                    annotated_list.append(annotated)

            counts = np.array([c.shape[1] for c in combined_list], dtype=np.int32)
            stems = [_stem(names[i] if names else None, i)
                     for i in range(len(imgs_in))]

            data = {
                # (num_images, 130, N_max) — zero-padded to the widest image
                "keypoints": wrap_value(np_to_bytes(pad_and_stack(combined_list))),
                # (num_images,) — the true N per image, before padding
                "counts": wrap_value(np_to_bytes(counts)),
            }
            encoding = {"keypoints": "numpy", "counts": "numpy"}

            if annotated_list:
                data["annotated"] = wrap_value(annotated_list)
                encoding["annotated"] = "identity"

            if spec["mat"]:
                data["mat"] = wrap_value([_savemat_bytes(c) for c in combined_list])
                encoding["mat"] = "identity"

            return _done(
                {
                    "feature": "SIFT",
                    "descriptor_length": _DESCRIPTOR_LENGTH,
                    "mat_key": _MAT_KEY,
                    "num_images": len(imgs_in),
                    "num_keypoints": [int(n) for n in counts],
                    "files": [f"{s}.mat" for s in stems],
                    "runtime": time.time() - start_time,
                },
                data=data, encoding=encoding)
        except Exception as e:
            logging.exception(f"Error in Process: {e}")
            return _error(str(e))


# ---------------------------------------------
# Server setup
# ---------------------------------------------
def get_port():
    try:
        port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        if port <= 0:
            logging.error("Port must be positive")
            return None
        return port
    except ValueError:
        logging.exception("Invalid port value")
        return None


def run_server(server):
    port = get_port()
    if not port:
        return
    target = f"[::]:{port}"
    server.add_insecure_port(target)
    server.start()
    logging.info(f"Server started at {target}")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    import grpc
    import grpc_reflection.v1alpha.reflection as grpc_reflection

    logging.basicConfig(
        format="[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(),
        options=[
            ('grpc.max_send_message_length', -1),
            ('grpc.max_receive_message_length', -1),
        ],
    )
    folder_wd_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)

    service_names = (
        folder_wd_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
        grpc_reflection.SERVICE_NAME,
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
