# VisionIST

A fleet of independent, Dockerized AI inference services — "boxes" — speaking
one shared gRPC envelope, plus a thin, box-agnostic Python client that drives
any of them.

One box = one container = one gRPC service:

```protobuf
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

Every box listens on port **8061** (AI4EU spec; overridable with the `PORT`
env var), dispatches work from a `config` section namespaced under its own key
(`{"clip": {...}}`, `{"lang_sam": {...}}`, …), and answers with a namespaced
status plus `data` payloads. There is no per-box SDK and no central
orchestrator — boxes stay independent, addressable, composable units, and the
*client* does the calling.

## Calling a box

```python
from visionist_client import Visionist
import pathlib

b = Visionist("localhost:8061")                      # any box, by IP:port
res = b.run(data   ={"images": [pathlib.Path("dog.jpg")]},
            config ={"lang_sam": {"command": "segment",
                                  "parameters": {"box_threshold": 0.3,
                                                 "text_threshold": 0.25},
                                  "text_prompt": ["a dog"]}})
print(res)          # decoded fields + parsed config + status
print(res.results)  # decoded payload (the box declared "encoding": "zstd_pickle")
```

That's the entire end-user surface — zero box knowledge in the call.
Details: [visionist_client/README.md](visionist_client/README.md).

## Quick start

Launch the **whole stack** — every box **plus** the webui — in one command.
(Images build from source; the first build is slow — the `vggt` box pulls ~5 GB
of weights.)

```bash
cd fleet
docker compose build --parallel     # first run only, or after changing a box
docker compose up -d                 # starts the boxes + the webui (+ docktail)
docker compose ps                    # watch them come up

python hello.py                     # one generic tour call per box
python supervisor_demo.py           # guided demo with decoded output per box
```

**Access**
- webui (local): **http://localhost:8080** (SPA at `/`, API at `/api/*`)
- boxes (direct): host ports **9061–9069**, or by service name on the
  `visionist-fleet` network — see the per-box READMEs

**Stop it all:** `cd fleet && docker compose down`
(or `docker compose stop` to keep containers for a faster re-`up`).

Or run a single box yourself (full instructions in
[docs/Quick_Start_Guide.md](docs/Quick_Start_Guide.md)):

```bash
cd images/lang_segm
docker build --tag my_lang_segm -f docker/Dockerfile .
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 --ipc=host my_lang_segm
```

## Boxes in this repo

| Box | Type | What it does |
|-----|------|--------------|
| clip | GPU | CLIP image/text embeddings |
| tapnext_tracker | GPU | Point tracking with TAPNext (stateful; multi-session via `session_id`) |
| lang_segm | GPU | Text-guided segmentation (LangSAM) |
| textEmbedding | GPU / CPU | Sentence-BERT text embeddings |
| vggt | GPU | 3D reconstruction from image sequences |
| yolo | GPU / CPU | Object detection **+ tracking** (YOLOv8n via ultralytics) on images and videos — per-session track ids (multi-session, like tapnext) |
| lightglue_box | GPU / CPU | Feature matching with LightGlue — SuperPoint/DISK features per image; for pairs, the matcher's `matches` (+ `confidence`) |
| opencv_box | GPU / CPU | Feature extraction & matching (SIFT / ORB / LightGlue) |
| features_box | CPU | SIFT keypoints in the SIFT-Extractor layout — `(2 + 128, N)` array, annotated JPEGs, MATLAB `.mat` |
| moge_box | GPU | MoGe-3 monocular 3D geometry (depth / points / normals) — CUDA-only |

**The per-box README is the authoritative source for that box's request shape**
(config keys, fields, status, how to decode `results`):
[`images/<name>/README.md`](images/).

## Key contracts

- **Shared envelope** — [`protos/pipeline.proto`](protos/) is the one
  interface, vendored into every box; `aux.py` is the `wrap_value` /
  `unwrap_value` helper.
- **Config dispatch** — the request's `config_json` carries a section
  namespaced under the box's key; `{"command": "reset"}` is accepted by every
  standard box.
- **Declared payload encoding** — boxes declare `"encoding"` (a codec name, or
  a `{field: codec}` map) in their *response* config; the client decodes with
  the named codec (`identity` / `json` / `torch` / `numpy` / `zstd_pickle`)
  and defaults to raw `bytes`. Design + rationale:
  [docs/CODECS.md](docs/CODECS.md).

## Documentation

Start in [docs/index.md](docs/index.md):

- [Architecture overview](docs/Architecture_Overview.md) — what a box is,
  conventions, GPU memory lifecycle
- [Quick start guide](docs/Quick_Start_Guide.md) — run a box, call it, build
  your own, test it
- [gRPC services reference](docs/gRPC_Services_Reference.md) — the envelope
  contract: `config_json` shape per box, `data` fields, `results` encoding
- [Docker image template guide](docs/Docker_Image_Template_Guide.md) — the
  Dockerfile templates used by the boxes in this repo
- [CODECS](docs/CODECS.md) — self-describing payload decoding
- [webui](webui/README.md) — declarative web layer over the fleet (HTTP API on
  top of `visionist_client`; box knowledge lives in YAML definitions)

## Tests

- **Client** — `visionist_client/tests/`: `fake_box_smoke.py` (in-process fake
  boxes, no GPU) and `codec_smoke.py` (the declared-encoding contract), plus
  `live_tapnext.py` against a real box.
- **Boxes** — each box ships `images/<name>/test/test_<name>.py`, pointable at
  a running box:
  `python images/lang_segm/test/test_lang_sam.py`
