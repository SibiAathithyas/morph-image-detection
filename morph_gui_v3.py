# morph_gui_v3.py
# Single-image or multi-image viewer with persistent cards (no batch table).
# Run:
#   streamlit run "D:\Morph image detection project\morph_gui_v3.py"

import io
import time
import hashlib
from pathlib import Path

import streamlit as st
from ultralytics import YOLO
import torch
from PIL import Image, UnidentifiedImageError
import pandas as pd

# ------------------ CONFIG ------------------
MODEL_PATH = r"D:\Morph image detection project\yolo_cv_runs\production\v3\morphdet_v3.pt"
UNCERTAIN_THRESHOLD = 0.80   # if max prob < threshold -> "uncertain"
AUTO_SAVE_DIR = Path(r"D:\Morph image detection project\yolo_cv_runs\test_eval")

# ------------------ HELPERS ------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

def pil_from_bytes(b: bytes) -> Image.Image:
    # robust PIL open
    img = Image.open(io.BytesIO(b))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def to_probs(yolo_probs):
    if yolo_probs is None:
        return None
    if hasattr(yolo_probs, "data") and isinstance(yolo_probs.data, torch.Tensor):
        t = yolo_probs.data
    else:
        t = torch.as_tensor(yolo_probs)
    return t.float().cpu().view(-1)

def predict_image(model: YOLO, image_pil: Image.Image):
    results = model.predict(
        image_pil,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False
    )
    probs = to_probs(getattr(results[0], "probs", None))
    if probs is None or probs.numel() == 0:
        return None, None, None
    idx = int(torch.argmax(probs))
    conf = float(probs[idx])
    raw_label = model.names[idx] if hasattr(model, "names") else str(idx)
    decision = raw_label if conf >= UNCERTAIN_THRESHOLD else "uncertain"
    return raw_label, conf, decision

def file_fingerprint(name: str, b: bytes) -> str:
    # dedupe across uploads: filename + first 1MB hash + size
    h = hashlib.md5(b[:1024 * 1024]).hexdigest()
    return f"{name}|{h}|{len(b)}"

def badge_html(text: str, color_hex: str) -> str:
    return f"""
    <span style="
      display:inline-block;padding:4px 10px;border-radius:999px;
      font-weight:600;background:{color_hex}20;color:{color_hex};
      border:1px solid {color_hex}60;">
      {text}
    </span>"""

# ------------------ UI ------------------
st.set_page_config(page_title="Morph Detector v3", layout="wide")
st.title("Morph Image Detector — v3 (Local)")
st.caption(
    f"Model: {MODEL_PATH} · "
    f"'Uncertain' if confidence < {UNCERTAIN_THRESHOLD:.2f} · "
    f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
)

model = load_model()

# Session state setup
if "cards" not in st.session_state:
    # each card: {"name": str, "bytes": bytes, "pred": str, "label": str|None, "conf": float}
    st.session_state.cards = []
if "seen_keys" not in st.session_state:
    st.session_state.seen_keys = set()
if "autosave_path" not in st.session_state:
    st.session_state.autosave_path = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0  # to fully reset uploader on "Clear all"

uploaded = st.file_uploader(
    "Upload image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
)

# Process uploads immediately; keep prior cards visible
if uploaded:
    added = 0
    t0 = time.time()
    for f in uploaded:
        try:
            content = f.read()
            if not content:
                continue
            fp = file_fingerprint(f.name, content)
            if fp in st.session_state.seen_keys:
                continue  # skip duplicates within UI session
            st.session_state.seen_keys.add(fp)

            try:
                img = pil_from_bytes(content)
            except (UnidentifiedImageError, OSError):
                # store error card so user sees the filename
                st.session_state.cards.append({
                    "name": f.name,
                    "bytes": content,
                    "pred": "error",
                    "label": None,
                    "conf": -1.0
                })
                added += 1
                continue

            raw_label, conf, decision = predict_image(model, img)

            st.session_state.cards.append({
                "name": f.name,
                "bytes": content,
                "pred": decision if decision else "error",
                "label": raw_label,
                "conf": conf if conf is not None else -1.0
            })
            added += 1

        except Exception:
            st.session_state.cards.append({
                "name": f.name,
                "bytes": b"",
                "pred": "error",
                "label": None,
                "conf": -1.0
            })
            added += 1

    if added:
        # autosave a cumulative CSV snapshot
        out_dir = AUTO_SAVE_DIR / f"gui_cumulative_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([{
            "image_name": c["name"],
            "pred_label": c["pred"],
            "raw_class": c["label"],
            "confidence": c["conf"],
        } for c in st.session_state.cards])
        csv_path = out_dir / "results.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        st.session_state.autosave_path = str(csv_path)
        st.success(f"Processed {added} image(s) in {time.time() - t0:.1f}s · Autosaved: {csv_path}")

# Controls
colA, colB = st.columns([1, 1])
with colA:
    if st.button("Clear all"):
        st.session_state.cards.clear()
        st.session_state.seen_keys.clear()
        st.session_state.autosave_path = None
        st.session_state.uploader_key += 1  # force clear the uploader widget
        st.rerun()

with colB:
    if st.session_state.cards:
        df = pd.DataFrame([{
            "image_name": c["name"],
            "pred_label": c["pred"],
            "raw_class": c["label"],
            "confidence": c["conf"],
        } for c in st.session_state.cards])
        st.download_button(
            "Download results.csv",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="results.csv",
            mime="text/csv"
        )

st.write(f"Total images detected: {len(st.session_state.cards)}")

# Card gallery (3 columns)
if st.session_state.cards:
    cols = st.columns(3)
    for i, c in enumerate(st.session_state.cards):
        with cols[i % 3]:
            # image
            try:
                img = pil_from_bytes(c["bytes"])
                st.image(img, caption=c["name"], use_column_width=True)
            except Exception:
                st.warning(f"{c['name']}: could not render image")

            # badge
            if c["pred"] == "real":
                b = badge_html("REAL", "#2ea043")
            elif c["pred"] == "morph":
                b = badge_html("MORPH", "#d73a49")
            elif c["pred"] == "uncertain":
                b = badge_html("UNCERTAIN", "#d29922")
            else:
                b = badge_html("ERROR", "#6a737d")
            st.markdown(b, unsafe_allow_html=True)

            # details
            if c["conf"] >= 0:
                st.caption(f"Confidence: {c['conf']:.4f} · Raw class: {c['label']}")
            else:
                st.caption("No confidence available.")

    if st.session_state.autosave_path:
        st.caption(f"Autosaved results to: {st.session_state.autosave_path}")
else:
    st.info("No results yet. Upload image(s) to begin.")
