"""
Few-shot teabag anomaly detection: DINOv2 + Qdrant + myCobot 280 Pi.
Runs on the LAPTOP. The Pi runs pi_arm_server.py.

Setup:
    pip install torch transformers pillow opencv-python qdrant-client numpy
    docker run -p 6333:6333 -p 6334:6334 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
    Dashboard: http://localhost:6333/dashboard

Controls (click the video window first): Enter/Space = capture, q/Esc = quit.

Workflow:
    uv run src/teabag/main.py preview                      # tune ROI so one bag fills the box
    uv run src/teabag/main.py enroll --label good --n 30   # build the "normal" memory
    uv run src/teabag/main.py score --n 10 --set-threshold # held-out good bags -> threshold
    uv run src/teabag/main.py score --n 5                  # try faulty bags, see their scores
    uv run src/teabag/main.py enroll --label bad --n 5     # optional: store known-bad examples
    uv run src/teabag/main.py run                          # the demo loop
"""
import argparse
import json
import socket
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from qdrant_client import QdrantClient, models
from transformers import AutoImageProcessor, AutoModel

# ----------------------------- CONFIG -----------------------------------
CAMERA_INDEX = 0
ROI = (400, 200, 400, 400)            # x, y, w, h of the pick position (keep it square)
COLLECTION = "teabags"
MODEL_NAME = "facebook/dinov2-small"  # try dinov2-base for more detail
K = 3                                 # neighbours averaged for the score
THRESHOLD_MARGIN = 1.15               # threshold = worst held-out good score * this
THRESHOLD_FILE = Path("threshold.json")
SAMPLES_DIR = Path("samples")

PI_HOST, PI_PORT = "192.168.1.50", 9000   # your myCobot's IP
DRY_RUN = True                            # True = print arm commands instead of sending

client = QdrantClient(url="http://localhost:6333")

# ----------------------------- EMBEDDING --------------------------------
_device = "cuda" if torch.cuda.is_available() else "cpu"
_processor = _model = None


def _load_model():
    global _processor, _model
    if _model is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME).to(_device).eval()
    return _processor, _model


@torch.no_grad()
def embed(img: Image.Image) -> list[float]:
    """CLS token + mean patch token, L2-normalised. One vector per crop."""
    processor, model = _load_model()
    img = img.convert("RGB").resize((224, 224))  # we resize; no centre-crop so edges survive
    inputs = processor(images=img, return_tensors="pt",
                       do_resize=False, do_center_crop=False).to(_device)
    tokens = model(**inputs).last_hidden_state
    vec = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=-1)
    vec = torch.nn.functional.normalize(vec, dim=-1)
    return vec[0].cpu().tolist()


def vector_size() -> int:
    _, model = _load_model()
    return model.config.hidden_size * 2


# ----------------------------- QDRANT -----------------------------------
def ensure_collection():
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=vector_size(),
                                               distance=models.Distance.COSINE),
        )
        client.create_payload_index(COLLECTION, "label", models.PayloadSchemaType.KEYWORD)


def _label_filter(label: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(
        key="label", match=models.MatchValue(value=label))])


def add_sample(img: Image.Image, label: str):
    SAMPLES_DIR.mkdir(exist_ok=True)
    pid = str(uuid.uuid4())
    path = SAMPLES_DIR / f"{label}_{pid[:8]}.jpg"
    img.save(path)
    client.upsert(COLLECTION, points=[models.PointStruct(
        id=pid, vector=embed(img),
        payload={"label": label, "path": str(path), "ts": time.time()})])


# ----------------------------- THE THREE VERBS --------------------------
_cap = None


def grab(flush: int = 5) -> np.ndarray:
    """Full BGR frame from the USB camera."""
    global _cap
    if _cap is None:
        _cap = cv2.VideoCapture(CAMERA_INDEX)
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        time.sleep(1.0)  # let auto-exposure settle
    for _ in range(flush):  # drop stale buffered frames
        _cap.read()
    ok, frame = _cap.read()
    if not ok:
        raise RuntimeError("Camera read failed")
    return frame


def look() -> Image.Image:
    """Grab a frame and crop to the pick position."""
    x, y, w, h = ROI
    crop = grab()[y:y + h, x:x + w]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def ask_qdrant(img: Image.Image) -> dict:
    """Anomaly score = 1 - mean cosine similarity to the K nearest GOOD bags."""
    vec = embed(img)
    good = client.query_points(COLLECTION, query=vec, limit=K,
                               query_filter=_label_filter("good"),
                               with_payload=True).points
    if not good:
        raise RuntimeError("No good samples enrolled yet")
    bad = client.query_points(COLLECTION, query=vec, limit=1,
                              query_filter=_label_filter("bad"),
                              with_payload=True).points
    return {
        "score": 1.0 - float(np.mean([p.score for p in good])),
        "neighbours": [p.payload["path"] for p in good],
        # optional: a known-bad example is closer than any good one
        "closest_is_bad": bool(bad) and bad[0].score > good[0].score,
    }


def sort(result: dict, threshold: float) -> str:
    """Decide good/bad and command the arm."""
    verdict = "bad" if (result["score"] > threshold or result["closest_is_bad"]) else "good"
    arm_command(verdict)
    return verdict


# ----------------------------- ARM (talks to the Pi) --------------------
def arm_command(cmd: str):
    """cmd: 'good' (pick -> good bin), 'bad' (pick -> reject bin), 'home'."""
    if DRY_RUN:
        print(f"   [dry run] arm <- {cmd}")
        return
    with socket.create_connection((PI_HOST, PI_PORT), timeout=90) as s:
        s.sendall((cmd + "\n").encode())
        reply = s.makefile().readline().strip()
    if reply != "DONE":
        raise RuntimeError(f"Arm replied: {reply!r}")


# ----------------------------- COMMANDS ---------------------------------
def wait_for_capture(prompt: str, status: str = "", colour=(0, 255, 0)) -> bool:
    """Live camera + ROI box. Enter/Space -> True (capture), q/Esc -> False (quit)."""
    x, y, w, h = ROI
    while True:
        frame = grab(flush=0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if status:
            cv2.putText(frame, status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        cv2.imshow("teabag", frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):        # Enter or Space
            return True
        if key in (ord("q"), 27):  # q or Esc
            return False


def cmd_preview():
    x, y, w, h = ROI
    while True:
        frame = grab(flush=0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("preview (q to quit)", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def cmd_enroll(label: str, n: int):
    ensure_collection()
    print("Vary string position and rotation a little between shots.")
    count = 0
    while count < n:
        if not wait_for_capture(f"[{count + 1}/{n}] {label.upper()} - Enter to capture, q to quit"):
            break
        add_sample(look(), label)
        count += 1
    cv2.destroyAllWindows()
    print(f"Enrolled {count} '{label}' samples.")


def cmd_score(n: int, set_threshold: bool):
    scores = []
    while len(scores) < n:
        if not wait_for_capture(f"[{len(scores) + 1}/{n}] Enter to score, q to quit"):
            break
        r = ask_qdrant(look())
        scores.append(r["score"])
        print(f"   score = {r['score']:.4f}")
    cv2.destroyAllWindows()
    if not scores:
        return
    print(f"min {min(scores):.4f}  mean {np.mean(scores):.4f}  max {max(scores):.4f}")
    if set_threshold:
        thr = max(scores) * THRESHOLD_MARGIN
        THRESHOLD_FILE.write_text(json.dumps({"threshold": thr}))
        print(f"Threshold saved: {thr:.4f}")


def cmd_run():
    threshold = json.loads(THRESHOLD_FILE.read_text())["threshold"]
    print(f"Threshold {threshold:.4f}. Press q in the window to stop.")
    status, colour = "", (0, 255, 0)
    while wait_for_capture("Enter to inspect, q to quit", status, colour):
        img = look()                       # EYES
        result = ask_qdrant(img)           # MEMORY
        verdict = sort(result, threshold)  # HANDS
        colour = (0, 200, 0) if verdict == "good" else (0, 0, 255)
        status = f"Last: {verdict.upper()}  score {result['score']:.3f}"
        print(f"{verdict.upper()}  score={result['score']:.4f}  nearest: {result['neighbours'][0]}")
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preview")
    e = sub.add_parser("enroll")
    e.add_argument("--label", default="good", choices=["good", "bad"])
    e.add_argument("--n", type=int, default=30)
    s = sub.add_parser("score")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--set-threshold", action="store_true")
    sub.add_parser("run")
    a = p.parse_args()

    if a.cmd == "preview":
        cmd_preview()
    elif a.cmd == "enroll":
        cmd_enroll(a.label, a.n)
    elif a.cmd == "score":
        cmd_score(a.n, a.set_threshold)
    elif a.cmd == "run":
        cmd_run()


if __name__ == "__main__":
    main()
