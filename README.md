# Teabag QC

Few-shot anomaly detection for teabags. A fixed USB camera looks at a teabag, a DINOv2 model turns the image into a vector, Qdrant finds the most similar *known-good* teabags, and a myCobot 280 Pi arm sorts the bag into a **good** or **reject** bin.

You only need photos of good teabags. Anything that doesn't look like them is flagged.

---

## Architecture

Three jobs, three places, plus a memory:

| Role | Where it runs | Responsibility |
|---|---|---|
| **Eyes** | USB camera, plugged into the laptop | Fixed overhead view of the pick position. Identical framing every time. |
| **Brain** | Laptop (GPU or decent CPU) | Grab frame, embed, query Qdrant, decide, command the arm. |
| **Memory** | Qdrant in Docker, on the laptop | Stores embeddings plus payload (label, image path, timestamp). Dashboard for the demo video. |
| **Hands** | myCobot 280 Pi (Raspberry Pi) | Moves the arm and works the gripper. No camera, no heavy compute. |

```
   USB camera ──frame──►  LAPTOP (teabag-qc)                      myCobot 280 Pi
                          ┌──────────────────────────┐            ┌──────────────────┐
                          │ look()   crop to ROI     │            │ pi_arm_server.py │
                          │   │                      │            │                  │
                          │   ▼                      │            │  "good" → pick   │
                          │ embed()  DINOv2 → vector │            │     → good bin   │
                          │   │                      │   TCP :9000│  "bad"  → pick   │
                          │   ▼                      │───────────►│     → reject bin │
                          │ ask_qdrant() ◄──────────┐│  "good" /  │  "home"          │
                          │   │                     ││   "bad"    └──────────────────┘
                          │   ▼                     ││
                          │ sort()  score > thresh? ││
                          └─────────────────────────┼┘
                                                    │
                                            Qdrant (Docker)
                                         localhost:6333/dashboard
```

### The three verbs

- **`look()`** grabs a frame, drops stale buffered frames, and crops to a fixed region of interest (ROI) so one teabag fills the image.
- **`ask_qdrant()`** embeds the crop and retrieves the K nearest vectors labelled `good`. The **anomaly score** is `1 - mean(cosine similarity)`. Low means it looks like known-good bags; high means it doesn't.
- **`sort()`** compares the score with the saved threshold and sends `good` or `bad` to the arm.

### How the decision works

1. **Enrol** 20-50 good teabags. Each is embedded and stored in Qdrant with `label=good`.
2. **Calibrate** with ~10 *different* good teabags that weren't enrolled. The threshold is the worst (highest) score among them, times a safety margin of 1.15.
3. **Run.** Any teabag scoring above the threshold is rejected.
4. **Optional:** enrol known-bad examples with `label=bad`. If a new bag's nearest neighbour is a bad example rather than a good one, it's rejected even if its score is under the threshold.

### Embedding

`facebook/dinov2-small` by default. The vector is the CLS token concatenated with the mean patch token, L2-normalised (768 dimensions for `small`, 1536 for `base`). DINOv2 is used rather than CLIP because it is better at texture and fine visual detail like stains and tears.

---

## Why Qdrant? (and when you wouldn't need it)

For this demo, a vector database isn't strictly required. With 30-50 good teabags, the whole "database" fits in a small NumPy matrix:

```python
import numpy as np

good = np.load("good_embeddings.npy")      # shape (50, 768), L2-normalised

def score(vec):
    sims = good @ vec                       # cosine similarity to every good bag
    return 1 - np.sort(sims)[-3:].mean()    # 1 - mean of the 3 nearest
```

That is the same algorithm as `ask_qdrant()`. It runs in microseconds with nothing to install, and at this scale brute force is faster than a network call to Qdrant, with identical results.

### What Qdrant adds for this project

- **Payload and filtering.** Each vector carries its label, image path and timestamp. "Nearest *good* neighbours only" is a filter on the query, and when a bag is rejected you can pull up the matching reference image. With NumPy you would keep parallel arrays or a metadata file and keep them in sync by hand.
- **Persistence.** Embeddings survive restarts, and you can add or delete samples while the system runs. With NumPy you save and reload files yourself.
- **The dashboard.** You can browse the collection, inspect points and run searches visually, which makes a good shot for the demo video.
- **A production-shaped architecture.** A separate memory service next to the brain reads like a real production line, and suits what the demo is showing.

### What it adds if the project grows

- **Scale.** Patch-level embeddings store about 256 vectors per image. That is still only ~13k vectors for 50 bags, which NumPy handles fine. At thousands of bags you are into millions of vectors, and brute force starts to hurt. Qdrant uses an approximate nearest-neighbour index (HNSW) to stay fast.
- **Multiple clients.** Several machines or processes (a second camera station, a dashboard, a retraining script) can share one memory over the network.
- **Concurrent updates.** You can enrol new samples while another process is querying, without managing file locks.
- **Quantization and on-disk storage.** These keep memory use down as the collection grows.

### Which to use

| Situation | Better choice |
|---|---|
| Quick experiment, one script, under ~10k vectors | NumPy (or FAISS) |
| Metadata, filtering, persistence, a visual dashboard | Qdrant |
| Patch-level embeddings at larger scale | Qdrant |
| Multiple stations or services sharing the same memory | Qdrant |

Starting with NumPy to prove the idea works is a perfectly reasonable approach. Qdrant is used here because the dashboard and the production-line story are part of what the demo shows, and because the design carries over unchanged if you scale up.

---

## Project layout

```
teabag/
├── pyproject.toml
├── README.md
├── src/teabag/
│   └── main.py            # laptop script: camera, embedding, Qdrant, CLI
├── pi/
│   └── pi_arm_server.py   # copy this to the Raspberry Pi
├── samples/               # created at runtime: cropped images of enrolled bags
├── threshold.json         # created by `score --set-threshold`
└── qdrant_storage/        # created by Docker: Qdrant's data
```

`samples/` and `threshold.json` are relative paths, so **run commands from the project root**.

---

## Setup

### Requirements

- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- Docker (for Qdrant)
- A USB camera on the laptop, mounted fixed above the pick position
- myCobot 280 Pi with a gripper, on the same network as the laptop

### Install the laptop side

```bash
uv sync
```

The first run downloads the DINOv2 weights from Hugging Face (a one-off).

> **GPU on Windows:** the default PyPI `torch` is CPU-only there. Install the CUDA build from pytorch.org first. On Linux the default wheel already supports CUDA.

### Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

Dashboard: <http://localhost:6333/dashboard>

---

## Usage

Every command is `uv run src/teabag/main.py <command>` (or `uv run teabag-qc <command>`).

`enroll`, `score` and `run` all open a **live camera window** with the ROI drawn as a green box, so you can see exactly what will be captured. Controls (click the video window first, because the keys go to the window, not the terminal):

| Key | Action |
|---|---|
| `Enter` or `Space` | Capture the current frame |
| `q` or `Esc` | Quit the command |

### 1. Frame the shot

```bash
uv run teabag-qc preview
```

Shows the live camera with the ROI as a green box. Edit `ROI = (x, y, w, h)` in `main.py` until one teabag fills the box. Keep it square. Press `q` to quit. The other commands show the same live view, so this is mainly for the initial framing.

### 2. Enrol good teabags

```bash
uv run teabag-qc enroll --label good --n 30
```

Place a bag, check it in the live view, press `Enter` in the window to capture, and repeat. Vary the string position and rotation slightly each time, so normal variation doesn't trigger false alarms. Press `q` to stop early; whatever you captured so far is kept.

### 3. Set the threshold

```bash
uv run teabag-qc score --n 10 --set-threshold
```

Use 10 good bags that you **didn't** enrol. Press `Enter` for each one; scores are printed in the terminal, and the threshold is saved to `threshold.json`. If you quit early, the threshold is calculated from the bags scored so far.

### 4. Try some faulty bags

```bash
uv run teabag-qc score --n 5
```

Nothing is stored, so run this as often as you like. Work through the difficulty ladder below and note which faults score above the threshold.

### 5. (Optional) Enrol known-bad examples

```bash
uv run teabag-qc enroll --label bad --n 5
```

### 6. Run the line

```bash
uv run teabag-qc run
```

Place a bag and press `Enter` in the window. The arm sorts it, and the last verdict and score stay on screen as an overlay on the live view until the next bag. Press `q` to stop.

### Command reference

| Command | Options | What it does |
|---|---|---|
| `preview` | | Live camera view with the ROI box (`q` to quit) |
| `enroll` | `--label good\|bad` (default `good`), `--n` (default 30) | Capture and store samples in Qdrant |
| `score` | `--n` (default 10), `--set-threshold` | Score bags without storing; optionally save the threshold |
| `run` | | Main loop: look, ask Qdrant, sort |

### Configuration

Constants at the top of `main.py`:

| Name | Default | Meaning |
|---|---|---|
| `CAMERA_INDEX` | `0` | Which camera OpenCV opens |
| `ROI` | `(400, 200, 400, 400)` | Crop box: x, y, width, height |
| `MODEL_NAME` | `facebook/dinov2-small` | Try `dinov2-base` for more detail |
| `K` | `3` | Neighbours averaged for the score |
| `THRESHOLD_MARGIN` | `1.15` | Safety margin over the worst good score |
| `PI_HOST`, `PI_PORT` | `192.168.1.50`, `9000` | Where the arm server listens |
| `DRY_RUN` | `True` | Print arm commands instead of sending them |

---

## The arm (Raspberry Pi)

`pi/pi_arm_server.py` listens on TCP port 9000 for one-word commands: `good`, `bad`, `home`. It replies `DONE` or `ERR ...`.

1. Copy it to the Pi and run `pip install pymycobot`.
2. **Teach the poses.** Every pose in the file is a placeholder (`[0, 0, 0, 0, 0, 0]`). For each one, release the servos, move the arm by hand, and read the angles:
   ```python
   from pymycobot import MyCobot280
   mc = MyCobot280("/dev/ttyAMA0", 1000000)
   mc.release_all_servos()
   print(mc.get_angles())
   ```
   Poses needed: `home`, `above_tray`, `tray`, `above_good`, `good`, `above_bad`, `bad`.
3. Run `python3 pi_arm_server.py`, then set `PI_HOST` in `main.py` and `DRY_RUN = False`.

**Safety:**
- Test with the gripper empty and your hand clear of the workspace.
- The server has no authentication, so keep it on a trusted network.
- pymycobot's API (`MyCobot280`, `is_moving`, `set_gripper_state`) varies between versions, and the gripper open/close values depend on your gripper. Check them against your installed version.

---

## Making faulty teabags

Build a difficulty ladder and show where the threshold catches each rung.

| Difficulty | Fault | How |
|---|---|---|
| Easy | Wrong bag | Different flavour or brand |
| Easy | Coloured tape | Strip across the bag |
| Easy | Torn bag | Rip a corner so leaves spill out |
| Medium | Loose tea on the bag | Sprinkle leaves from another bag |
| Medium | Water or coffee stain | Dab with a wet teabag or coffee, let it dry |
| Medium | Missing string or tag | Snip it off |
| Medium | Crushed or folded bag | Crumple it |
| Hard | Felt-tip dot | 2-3 mm dot |
| Hard | Pinprick hole or tiny tear | Pin or scissors |
| Hard | Faint smudge | Rub in a little cocoa or cinnamon |

Whole-image embeddings tend to catch faults covering roughly 5-10% of the crop. For smaller faults, see "Extending" below.

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `error: the following arguments are required: cmd` | You ran the script with no command. Add one: `preview`, `enroll`, `score` or `run`. |
| Pressing Enter in the terminal does nothing | The keys go to the video window. Click the window first, then press `Enter` or `Space`. |
| Video window is blank, or OpenCV reports Qt/display errors | You may have `opencv-python-headless` installed. Use `opencv-python` (what `pyproject.toml` lists): `uv pip uninstall opencv-python-headless`, then `uv sync`. |
| `Camera read failed` | Wrong `CAMERA_INDEX`, or another program has the camera. Try `1`, `2`. |
| `Connection refused` on port 6333 | Qdrant isn't running. Start the Docker container. |
| `No good samples enrolled yet` | Run `enroll --label good` first. |
| `FileNotFoundError: threshold.json` | Run `score --n 10 --set-threshold`, from the project root. |
| Good bags get rejected | Inconsistent lighting, arm shadows in the shot, or too little variety in enrolled samples. Use diffuse light and enrol more varied placements, then recalibrate. |
| Faulty bags get through | Defect too small for whole-image scoring. Lower `THRESHOLD_MARGIN`, use `dinov2-base`, tighten the ROI, or try patch-level scoring. |
| Scores drift between sessions | Camera or lighting moved. Keep both fixed, and recalibrate if they change. |
| Arm command times out | Check `PI_HOST`, that the server is running on the Pi, and that both are on the same network. |

### Tips

- Use diffuse, glare-free light and a matte dark tray. Lighting changes look like anomalies.
- Floppy strings are the biggest source of false alarms. Enrol many string arrangements, or tighten the ROI to the bag body.
- Re-run `score --set-threshold` after changing the model, ROI, camera position or lighting. Old embeddings from a different ROI or model aren't comparable, so clear the collection and re-enrol.

---

## Extending

- **Patch-level scoring (PatchCore-style):** store every DINOv2 patch token as its own point and score by the worst-matching patch, so a small stain can't hide in the average.
- **CLIP comparison:** swap in a CLIP vision model and compare how well good and bad scores separate. A score histogram makes a good demo segment.
- **Hands-free loop:** replace the Enter/Space key press in `wait_for_capture()` with motion detection or a timer.
- **Multiple tray slots:** loop over several ROIs and pick positions.
- **Arm moves freeze the window:** with `DRY_RUN = False` the video window pauses while the arm moves, because the script waits for the Pi's `DONE` reply. Run the arm in a background thread if you want live video during moves.
- **Logging runs:** write verdicts to a separate Qdrant collection (not the reference one) so the dashboard shows history.
