# Features Box

SIFT keypoint extraction behind the **shared envelope** interface — a port of
the [pivist/features](https://github.com/pivist/features) "SIFT Extractor"
command-line tool, which read a mounted folder of images and wrote one
annotated JPEG plus one MATLAB `.mat` per image. Same detector, same output
layout; the folder becomes an `Envelope`. Same one-RPC contract as every
standard box:

```protobuf
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

| Command | What it does | State |
|---|---|---|
| `extract` (default) | SIFT on each input image → `keypoints` (the CLI's `(2 + 128, N)` layout), `counts`, and optionally the red-dot `annotated` JPEGs and the `.mat` files. Stateless | none |
| `reset` | standard no-op (this box is stateless) | — |

Pure CPU — OpenCV's SIFT needs no GPU, so this box has no `device` parameter
and holds no VRAM.

## Directory structure

```
features_box/
├── docker/
│   └── Dockerfile
├── protos/
│   ├── pipeline.proto         # shared proto (same as every other box)
│   ├── pipeline_pb2.py        # generated
│   ├── pipeline_pb2_grpc.py   # generated
│   └── aux.py                 # wrap_value / unwrap_value helpers
├── src/
│   └── features_service.py    # PipelineService.Process(Envelope)
├── test/
│   ├── test_features.py       # smoke test against a running box
│   ├── smoke_inprocess.py     # drives the service in-process (no box build)
│   ├── 00.jpg                 # test fixture
│   └── 01.jpg                 # test fixture
├── requirements.txt
└── README.md
```

## Build

```bash
cd images/features_box
docker build --tag sipgisr/featuresbox --build-arg SERVICE_NAME=features -f docker/Dockerfile .
```

Small image: `opencv-contrib-python-headless` + `scipy` + grpc, no torch and
no model weights.

## Run

```bash
docker run --rm -p 8061:8061 -e PORT=8061 sipgisr/featuresbox
```

## Service usage

### Request

```json
// config_json — namespaced under the box key
{
  "features": {
    "command": "extract",            // "extract" (default) | "reset"
    "parameters": {
      "nfeatures": 0,                // 0 = keep every keypoint SIFT finds
      "n_octave_layers": 3,          // the cv2.SIFT_create() defaults —
      "contrast_threshold": 0.04,    //   i.e. exactly what the original
      "edge_threshold": 10,          //   CLI used (it passed no arguments)
      "sigma": 1.6,
      "draw": true,                  // return the red-dot annotated JPEGs
      "radius": 2,                   // annotation dot radius, as in main.py
      "mat": false                   // also return one .mat per image
    }
  }
}
```

`draw` / `mat` accept JSON booleans and the strings `"true"` / `"false"`
(what the webui's select widgets send).

`data`:

| field | kind | meaning |
|---|---|---|
| `images` | `bb` | one or more images (jpg/png/bmp/tiff); each is extracted independently |
| `names` | `ss` | *optional*: one original file name per image, in order — they name the `.mat` artifacts in the response's `files` (default: `image_000`, `image_001`, …) |

### Response

`config_json` carries a **namespaced status** (never flat):

```json
// extract, two images
{
  "features": {
    "status": "done",
    "feature": "SIFT",
    "descriptor_length": 128,
    "mat_key": "kp",
    "num_images": 2,
    "num_keypoints": [2518, 8514],
    "files": ["00.mat", "01.mat"],
    "runtime": 0.42,
    "encoding": {
      "keypoints": "numpy", "counts": "numpy",
      "annotated": "identity", "mat": "identity"
    }
  }
}

// reset
{ "features": { "status": "done", "action": "reset" } }
```

Status vocabulary per the shared contract: `done` (success),
`empty_request` (no `data.images`), `error` (reason in `"error"` — bad
command, bad parameter, `names`/`images` length mismatch, undecodable
frame, …).

`data`:

| field | kind | description |
|---|---|---|
| `keypoints` | `b` (`numpy`) | `np.save` blob, `(num_images, 2 + 128, N)`: **row 0 = x, row 1 = y, rows 2..129 = the SIFT descriptor** — the CLI's `kp` matrix, zero-padded to the widest image and stacked |
| `counts` | `b` (`numpy`) | `(num_images,)` `int32`: the true keypoint count per image (everything past it in `keypoints` is padding) |
| `annotated` | `bb` (`identity`) | `parameters.draw` (default on): one JPEG per image, a filled red dot at each keypoint — byte-for-byte the CLI's annotated output |
| `mat` | `bb` (`identity`) | `parameters.mat`: one MATLAB `.mat` per image, key `kp` — the CLI's `<name>.mat`, unchanged |

The declared `numpy` fields decode to `np.ndarray` in `visionist_client` (the
codec restores the `np.save` array — see `docs/CODECS.md`); the `identity`
fields stay raw file bytes, ready to write to disk.

### Semantics worth knowing

- **Padding.** Images rarely yield the same number of keypoints, so
  `keypoints` is padded with zeros on the last axis. **Always slice with
  `counts`** before using an image's points:
  `kp[i, :2, :counts[i]].T` → `(N_i, 2)` `(x, y)`,
  `kp[i, 2:, :counts[i]].T` → `(N_i, 128)` descriptors.
- **`nfeatures` is approximate.** OpenCV keeps every keypoint tied with the
  last one it retains, so a cap of 500 can return 501.
- **A featureless image is not an error.** It answers `done` with a
  `(1, 130, 0)` array and `counts == [0]` (the original CLI raised and
  skipped the file).
- **RGB quirk, kept on purpose.** `main.py` feeds SIFT
  `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`; SIFT then greys the frame with the
  BGR weights, so red and blue are swapped relative to plain greyscale. The
  box does the same, so its output matches the CLI's bit for bit — if you
  want the textbook behaviour, grey the image yourself before sending it.
- **`extract`** is stateless and idempotent; **`reset`** is the standard box
  reset, a plain no-op here.

## Call with visionist_client

```python
from visionist_client import Visionist
import pathlib

b = Visionist("localhost:8061")          # 9071 for the fleet's compose mapping

res = b.run(
    data   = {"images": [pathlib.Path("images/features_box/test/00.jpg"),
                         pathlib.Path("images/features_box/test/01.jpg")],
              "names":  ["00.jpg", "01.jpg"]},
    config = {"features": {"command": "extract",
                           "parameters": {"nfeatures": 500, "mat": True}}},
)

print(res.config["features"])      # status / num_keypoints / files / runtime
kp     = res.keypoints             # decoded: np.ndarray (2, 130, N)
counts = res.counts                # (2,) int32

xy   = kp[0, :2, :counts[0]].T     # (N_0, 2)   keypoint coordinates
desc = kp[0, 2:, :counts[0]].T     # (N_0, 128) SIFT descriptors

pathlib.Path("00_annotated.jpg").write_bytes(res.annotated[0])
pathlib.Path("00.mat").write_bytes(res.mat[0])   # loads in MATLAB as `kp`
```

In MATLAB:

```matlab
load('00.mat')        % kp: 130 x N
xy   = kp(1:2, :);    % coordinates
desc = kp(3:end, :);  % descriptors
```

## Relation to `opencv_box`

Both boxes run SIFT, but they answer different questions:

| | `features_box` | `opencv_box` |
|---|---|---|
| Question | "give me this image's SIFT features, in the SIFT-Extractor layout" | "match these two images" |
| Layout | one `(2 + 128, N)` matrix (`kp`), MATLAB-compatible | separate `keypoints` `(N, 2)` + `descriptors` `(N, 128)` |
| Matching | none | FLANN / LightGlue + RANSAC fundamental matrix |
| Extras | annotated JPEGs, `.mat` files | `matches_inliers_*`, `fundamental_matrix` |
| Extractors | SIFT | SIFT, ORB, SuperPoint, DISK |

## Testing

```bash
# running box (standard smoke test, like the rest of the fleet)
python images/features_box/test/test_features.py
BOX_HOST=10.0.0.5:8061 python images/features_box/test/test_features.py

# no box build needed — drives src/features_service.py in-process
# (needs numpy, opencv-python(-headless) and scipy locally)
cd images/features_box && python test/smoke_inprocess.py
```

Both were checked against the original CLI: for the bundled fixtures the
box's `keypoints`, `.mat` and annotated JPEG are **identical** to what
`python main.py images --output results` produces.
