# VisionIST: A Plain-Language Guide

Oct 11, 2026 · @Someone

## What VisionIST is

VisionIST is a set of computer-vision tools, each packaged as its own self-contained program called a **box**. One box detects objects in a photo, another matches two photos to each other, another estimates depth. A box does one job and nothing else.

Every box answers requests the same way, so you never learn a second interface. You send it some data — images, text, numbers — together with a short note saying what you want done, and it sends results back. The thing that sends that request is the **client**: a small Python package called `visionist-client`.

A box is simply a program listening at an address, so it makes no difference whether it runs on a lab server down the corridor or inside your own laptop. The call is identical; only the address changes. That one idea covers everything below.

There are 15 boxes today. They are listed, with what each one does, in the [https://github.com/sipg-isr/VisionIST\_Library ](https://github.com/sipg-isr/VisionIST_Library) registry.

&#91;embedded content: how the client reaches a box · 3 situations\]

The three situations below are the same call with a different address in it.

## What you need before you start

If someone else runs the boxes for you, one command is the whole installation.

1. **Python 3.10 or newer.** Check with `python3 --version`.
2. **The client.**

   ```bash
   pip install visionist-client
   ```
3. **Docker Desktop** — only if you want to run boxes on your own machine (sections 2, 3 and 4). Download it from [docker.com](https://www.docker.com/products/docker-desktop/) and leave it running in the background.
4. **A graphics card** — optional. Several boxes are happy on an ordinary laptop; the heavier ones want an NVIDIA GPU. Each box says which it needs in the fleet table.

That is all. There is no account to create, no server to register with, and nothing to configure.

## 1. The boxes run on someone else's server

You need two things: the server's address and the box's port number. Nothing is installed on your side beyond the client.

```python
from visionist_client import Visionist
import pathlib

b = Visionist("10.0.0.5:9065")        # the server's address, then the box's port
print(b.info())                        # a quick "are you there?" check

res = b.run(
    data   = {"images": [pathlib.Path("dog.jpg")]},
    config = {"opencv": {"command": "match"}},
)
print(res.config)                      # what the box says it did
```

Three things to know:

- **Each box has its own port, and it never changes.** `opencv` always answers on 9065, `yolo` on 9068, `features` on 9071. The full list is the [fleet table](https://github.com/sipg-isr/VisionIST/blob/main/fleet/README.md); adding new boxes never shifts the existing ones.
- **The word in `config` is the box's name.** It tells the box which instructions are meant for it. Each box's README gives the exact words it understands — `command`, and whatever `parameters` it accepts.
- **Your images never touch disk on the server.** They travel with the request and the answer comes straight back.

If your group also runs the browser console, you can skip Python entirely: open `http://<server>:8080`, pick a box from the menu, drag in an image and press run. It is the same boxes, with forms instead of code.

## 2. Running the published boxes on your own computer

The boxes are published ready-made on Docker Hub under the user **`sipgisr`**. You never build them; you download them and start them.

### One box

```bash
docker run --rm -p 9065:8061 -e PORT=8061 sipgisr/visionist-opencv
```

That downloads the box the first time and starts it. `-p 9065:8061` means *"let me reach it on my port 9065"* — 9065 is the number the registry assigned to `opencv`, so using it keeps your code the same whether the box is local or remote. Add `--gpus all` if the box wants a graphics card and you have one.

The image name is always `sipgisr/visionist-<box name>`, always with no version number — there is only ever one current image per box, so `docker pull` is the whole upgrade procedure.

### Several boxes at once

For more than one or two, let the generator write the configuration for you:

```bash
git clone https://github.com/sipg-isr/VisionIST
cd VisionIST
python3 tools/make_fleet.py --boxes opencv,features,yolo --out my-fleet
cd my-fleet
docker compose pull
docker compose up -d
```

Use `--all` instead of `--boxes ...` for everything, and add `--gpu` if you have a graphics card. This also starts the browser console at `http://localhost:8080`.

### How the client sees it

Exactly as before — that is the point.

```python
b = Visionist("localhost:9065")       # the only line that changed
```

The client has no idea whether the box is on your laptop, in the next room or in another country. Stop a box with `docker compose down`, or `Ctrl-C` for a single one.

## 3. Making a box of your own and running it locally

You do not write a box from scratch. A generator writes a complete, working one for you, and you replace the part that does the actual work.

```bash
git clone https://github.com/sipg-isr/VisionIST_Library
cd VisionIST_Library
pip install -r tools/requirements.txt

python3 tools/new_box.py my_box \
    --summary "What it does, in one line" \
    --runtime cpu --tags demo --github your-handle
```

What you get in `boxes/my_box/` already runs and already answers correctly — out of the box it reports the size of any image you send it. So the first thing you ever debug is your own method, never the plumbing.

Then:

1. **Open `src/my_box_service.py`** and replace the one clearly marked section with your own code. Everything around it — receiving the images, sending the answer back — is already written and should be left alone.
2. **List your ingredients** in `requirements.txt` (the Python libraries your code needs).
3. **Describe your box** in `box.yaml`: what it is called, what options it takes, what it returns. This file is the box's identity card, and almost everything else is generated from it.
4. **Package and start it:**

   ```bash
   cd boxes/my_box
   docker build --tag my_box --build-arg SERVICE_NAME=my_box -f docker/Dockerfile .
   docker run --rm -p 8061:8061 -e PORT=8061 my_box
   ```
5. **Call it** from another terminal, exactly like any other box:

   ```python
   from visionist_client import Visionist
   b = Visionist("localhost:8061")
   print(b.info())
   ```

At this point the box is yours alone, living on your machine. Nobody else can see it, and nothing you have done affects anyone. Section 4 is what changes that.

## 4. Publishing your box so everyone can use it

Publishing means two things happen: your box's description joins the shared registry on GitHub, and its ready-made image appears on Docker Hub under `sipgisr`. You do not upload the image yourself — that is automatic once your work is accepted.

### Step 1 — run the checks

From the top of your `VisionIST_Library` copy:

```bash
python3 tools/validate_boxes.py my_box    # is the description complete and correct?
python3 tools/sync_contract.py            # refresh the shared plumbing
python3 tools/build_index.py              # update the catalogue, the port list, the README table
python3 tools/check_docs.py my_box        # does your README do what it claims?
```

`build_index.py` also hands your box its permanent port number — the next free one, never an alphabetical slot. Commit the file it changes (`registry/ports.json`) with the rest, and that port is yours for good.

Do not hand-edit the catalogue, the README table or the port list. They are all written from your `box.yaml`, and editing them by hand is undone on the next run.

### Step 2 — propose it

Push your branch and open a pull request against [VisionIST\_Library](https://github.com/sipg-isr/VisionIST_Library). An automated check rebuilds and tests only the box you touched. A maintainer reviews it; from then on you are listed as the box's owner, so later changes to it come to you.

### Step 3 — release it

Once merged, set a version in your `box.yaml` and mark the release:

```bash
git tag my_box-v0.1.0 && git push origin my_box-v0.1.0
```

That is what builds the image and pushes it to Docker Hub as `sipgisr/visionist-my_box`. Anyone can now run it with the `docker run` line from section 2 — nothing else needed.

### Updating it later

Same route, smaller: change the code, raise the version in `box.yaml`, open a pull request, tag the new version. Boxes are released independently, so your update never disturbs anyone else's box. Because images carry no version number, everyone else picks up your change with `docker compose pull`.

### Two courtesies

- **Keep test images small** (under 256 KB). Anything bigger is referenced by a web link in `box.yaml` rather than committed, because a repository keeps every version of every file forever, for everyone who downloads it.
- **Never edit the `protos/` folder.** It is the shared language every box speaks, and it is copied in automatically. If you think it needs to change, open an issue first — that change touches every box at once.

## Worked example: matching two photos with SIFT

The task: two photographs of the same scene from different viewpoints, and the question *which point in the first photo is which point in the second?* SIFT is the classical answer — it finds distinctive little patches and describes each one so it can be recognised again. The `opencv` box does the whole job.

### Start the box

```bash
docker run --rm -p 9065:8061 -e PORT=8061 sipgisr/visionist-opencv
```

Skip this if the box already runs on a server — use that address below instead of `localhost`.

### Ask for the match

```python
from visionist_client import Visionist
import pathlib

b = Visionist("localhost:9065")

res = b.run(
    data   = {"images": [pathlib.Path("left.jpg"), pathlib.Path("right.jpg")]},
    config = {"opencv": {"command": "match",
                         "parameters": {"feature_extractor": "SIFT",
                                        "max_keypoints": 1000,
                                        "ratio_thresh": 0.75}}},
)

print(res.config["opencv"])      # {'status': 'done', 'num_inliers': 517, ...}

pts_a = res.matches_inliers_a    # points in the first photo   (K, 2)
pts_b = res.matches_inliers_b    # the same points in the second (K, 2)
F     = res.fundamental_matrix   # the 3x3 geometry relating the two views
```

Send exactly two images and the box does three things in one go: it finds SIFT points in each photo, pairs them up, then throws out the pairs that are geometrically impossible. `pts_a[i]` and `pts_b[i]` are the same physical point seen twice, and `num_inliers` tells you how many survived — a few hundred means a confident match, a handful means the photos probably do not overlap.

### Look at the result

```python
import cv2, numpy as np

left, right = cv2.imread("left.jpg"), cv2.imread("right.jpg")
canvas = np.hstack([left, right])
offset = left.shape[1]

for (xa, ya), (xb, yb) in zip(pts_a, pts_b):
    cv2.line(canvas, (int(xa), int(ya)), (int(xb) + offset, int(yb)),
             (0, 255, 0), 1)

cv2.imwrite("matches.jpg", canvas)
```

The photos side by side with a green line across every matched pair. Lines that all run roughly parallel mean the match is sound.

### If you only want the keypoints

Send a single image and the `opencv` box returns just the points and their descriptions, no matching. For the SIFT layout that MATLAB users expect — one `(2+128) × N` matrix per image, plus annotated JPEGs — use the `features` box on port 9071 instead.

## When something goes wrong

| What you see | What it usually means |
| --- | --- |
| `b.info()` cannot connect | The box is not running, or you have the wrong port. Check with `docker ps`. |
| `status: "empty_request"` | The box got no images. Check the field name — most boxes expect `images`. |
| `status: "error"` | The box ran but refused the request; the reason is in the `error` field next to it. Usually an option it does not recognise. |
| An empty reply, no error | The word in your `config` does not match the box's name, so it never saw your instructions. |
| Results come back as raw bytes | An optional library is missing: `pip install "visionist-client[torch]"`. |
| The first call is slow, later ones fast | Normal. A box loads its model on first use and keeps it warm. |

## Where to read more

- [VisionIST](https://github.com/sipg-isr/VisionIST) — the client, the browser console, the documentation
- [VisionIST\_Library](https://github.com/sipg-isr/VisionIST_Library) — every box, each with its own README giving the exact request it accepts
- [VisionIST\_matlab](https://github.com/sipg-isr/VisionIST_matlab) — the same boxes from MATLAB or Octave
