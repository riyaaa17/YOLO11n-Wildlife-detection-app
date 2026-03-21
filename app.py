"""
Wildlife Object Detection App — Streamlit UI
Based on YOLO11n fine-tuned on Wildlife Dataset
"""

import streamlit as st
import os
import glob
import yaml
import time
import random
import shutil
import tempfile
from pathlib import Path
from collections import Counter
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WildEye — Wildlife Detection",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --surface2:  #21262d;
    --border:    #30363d;
    --accent:    #e6a817;
    --accent2:   #c77b2e;
    --green:     #3fb950;
    --red:       #f85149;
    --blue:      #58a6ff;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --font-head: 'Playfair Display', Georgia, serif;
    --font-mono: 'DM Mono', 'Fira Code', monospace;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Header */
.wildeye-header {
    background: linear-gradient(135deg, #0d1117 0%, #1a1f2e 50%, #0d1117 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.wildeye-header::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -5%;
    width: 340px;
    height: 340px;
    background: radial-gradient(circle, rgba(230,168,23,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.wildeye-title {
    font-family: var(--font-head) !important;
    font-size: 3.2rem;
    font-weight: 900;
    color: var(--accent) !important;
    letter-spacing: -1px;
    line-height: 1;
    margin: 0;
}
.wildeye-sub {
    font-family: var(--font-mono);
    color: var(--muted);
    font-size: 0.85rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.wildeye-badge {
    display: inline-block;
    background: rgba(230,168,23,0.15);
    border: 1px solid rgba(230,168,23,0.4);
    color: var(--accent);
    font-size: 0.72rem;
    letter-spacing: 2px;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 1rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    flex: 1;
    min-width: 130px;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-val {
    font-family: var(--font-head);
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-label {
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Section headings */
.section-head {
    font-family: var(--font-head) !important;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text) !important;
    border-left: 3px solid var(--accent);
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* Status pills */
.pill-ok  { display:inline-block; background:#1a3a1a; border:1px solid var(--green); color:var(--green); border-radius:20px; padding:2px 12px; font-size:0.76rem; }
.pill-err { display:inline-block; background:#3a1a1a; border:1px solid var(--red);   color:var(--red);   border-radius:20px; padding:2px 12px; font-size:0.76rem; }
.pill-warn{ display:inline-block; background:#3a2a0a; border:1px solid var(--accent);color:var(--accent);border-radius:20px; padding:2px 12px; font-size:0.76rem; }

/* Info panel */
.info-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--blue);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.85rem;
    color: var(--muted);
}

/* Detection card */
.det-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.det-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.det-name { font-weight: 500; font-size: 0.9rem; }
.det-conf { margin-left: auto; color: var(--accent); font-size: 0.85rem; }

/* Streamlit overrides */
div[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 10px !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    border: none !important;
    font-family: var(--font-mono) !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stSlider > div > div { accent-color: var(--accent) !important; }
div[data-testid="stSelectbox"] > div { background: var(--surface2) !important; border-color: var(--border) !important; }
div[data-testid="stMetric"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; padding: 1rem !important; }
div[data-testid="stMetric"] label { color: var(--muted) !important; font-size: 0.75rem !important; letter-spacing: 2px !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: var(--font-head) !important; }
.stTabs [data-baseweb="tab"] { background: var(--surface2) !important; color: var(--muted) !important; border-radius: 6px 6px 0 0 !important; font-family: var(--font-mono) !important; font-size: 0.8rem !important; letter-spacing: 1px !important; }
.stTabs [aria-selected="true"] { background: var(--surface) !important; color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }
.stProgress > div > div { background: var(--accent) !important; }
div[data-testid="stExpander"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
CLASS_COLORS = [
    "#e6a817", "#3fb950", "#58a6ff", "#f85149", "#bc8cff",
    "#ffa657", "#39d353", "#79c0ff", "#ff7b72", "#d2a8ff",
    "#56d364", "#4ac3ff", "#ff6e6e", "#e8b4fb", "#ffd700",
]

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

def fig_to_pil(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none', dpi=130)
    buf.seek(0)
    return Image.open(buf).copy()

def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    ax.spines[:].set_color('#30363d')
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    ax.title.set_color('#e6edf3')
    return fig, ax

@st.cache_resource
def load_model(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)

def draw_boxes(image: Image.Image, boxes, class_names, conf_thresh=0.25):
    img = np.array(image.convert("RGB"))
    fig, ax = plt.subplots(1, figsize=(10, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.imshow(img)
    ax.axis('off')

    detections = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
            color = hex_to_rgb(CLASS_COLORS[cls_id % len(CLASS_COLORS)])

            rect = patches.FancyBboxPatch(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2.5, edgecolor=color,
                facecolor=(*color, 0.08),
                boxstyle="round,pad=2"
            )
            ax.add_patch(rect)
            label = f"{cls_name}  {conf:.0%}"
            ax.text(x1+4, y1-6, label, fontsize=9, color='white', fontweight='bold',
                    bbox=dict(facecolor=color, alpha=0.88, pad=3, edgecolor='none',
                              boxstyle='round,pad=0.3'))
            detections.append((cls_name, conf, cls_id))

    plt.tight_layout(pad=0)
    return fig_to_pil(fig), detections

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
        <span style='font-family:"Playfair Display",serif;font-size:1.4rem;
                     color:#e6a817;font-weight:900;'>WildEye</span>
        <br>
        <span style='font-size:0.7rem;color:#8b949e;letter-spacing:3px;
                     text-transform:uppercase;'>Detection Studio</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🔍 Detect", "🏋️ Train", "📊 Evaluate", "📈 Analytics"],
        label_visibility="collapsed"
    )
    st.divider()

    st.markdown('<p style="font-size:0.72rem;color:#8b949e;letter-spacing:2px;text-transform:uppercase;">Model Config</p>', unsafe_allow_html=True)
    model_source = st.selectbox("Model", ["YOLO11n (pretrained)", "Load custom .pt file"], label_visibility="collapsed")

    custom_model_path = None
    if model_source == "Load custom .pt file":
        uploaded_model = st.file_uploader("Upload .pt file", type=["pt"], label_visibility="collapsed")
        if uploaded_model:
            tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
            tmp.write(uploaded_model.read())
            tmp.flush()
            custom_model_path = tmp.name

    st.divider()
    st.markdown('<p style="font-size:0.72rem;color:#8b949e;letter-spacing:2px;text-transform:uppercase;">Inference Settings</p>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou_thresh  = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.45, 0.05)
    img_size    = st.select_slider("Image size", [320, 416, 512, 640, 768, 1024], value=640)

    st.divider()
    st.markdown("""
    <div style='font-size:0.72rem;color:#8b949e;line-height:1.7;'>
        <span style='color:#e6a817;'>●</span> YOLO11n · Ultralytics<br>
        <span style='color:#3fb950;'>●</span> Wildlife Dataset · Kaggle<br>
        <span style='color:#58a6ff;'>●</span> PyTorch backend
    </div>
    """, unsafe_allow_html=True)

# ── Page: Overview ────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("""
    <div class="wildeye-header">
        <div class="wildeye-title">WildEye</div>
        <div class="wildeye-sub">Wildlife Object Detection · YOLO11n</div>
        <div class="wildeye-badge">YOLO11n · Fine-tuned · Ultralytics</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "YOLO11n")
    c2.metric("Parameters", "2.6 M")
    c3.metric("Input Size", "640 × 640")
    c4.metric("Format", "YOLO .txt")

    st.markdown('<div class="section-head">About This App</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    WildEye is an end-to-end wildlife object detection studio built on <strong>YOLO11n</strong> — 
    Ultralytics' latest nano-scale model fine-tuned on the Kaggle Wildlife Dataset. 
    Upload images, run batch inference, train your own model, and inspect metrics — all from one interface.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-head">Pipeline</div>', unsafe_allow_html=True)
        steps = [
            ("1", "Dataset Download", "Kaggle API integration"),
            ("2", "Visualization", "Class distribution & sample images"),
            ("3", "Training", "AdamW · Mosaic · Mixup augmentation"),
            ("4", "Evaluation", "mAP@50 · Precision · Recall · F1"),
            ("5", "Inference", "Real-time detection with NMS"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:1rem;
                        background:#161b22;border:1px solid #30363d;
                        border-radius:8px;padding:0.8rem 1rem;margin:0.4rem 0;">
                <span style="background:#e6a817;color:#0d1117;font-weight:700;
                             width:24px;height:24px;border-radius:50%;
                             display:flex;align-items:center;justify-content:center;
                             font-size:0.75rem;flex-shrink:0;">{num}</span>
                <div>
                    <div style="font-weight:600;font-size:0.9rem;">{title}</div>
                    <div style="color:#8b949e;font-size:0.78rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-head">Model Comparison</div>', unsafe_allow_html=True)
        df_models = pd.DataFrame({
            "Model":   ["YOLO11n ★", "YOLO11s", "YOLO11m", "YOLO11l"],
            "Params":  ["2.6 M",     "9.4 M",   "20.1 M",  "25.3 M"],
            "mAP@50":  ["~39.5",     "~47.0",   "~51.5",   "~53.4"],
            "Speed":   ["~1.5 ms",   "~2.5 ms", "~4.7 ms", "~6.1 ms"],
        })
        st.dataframe(df_models, hide_index=True, use_container_width=True)

        st.markdown('<div class="section-head">Training Hyperparameters</div>', unsafe_allow_html=True)
        df_hp = pd.DataFrame({
            "Param":   ["epochs", "batch", "optimizer", "lr0", "freeze", "mosaic"],
            "Value":   ["50",     "16",    "AdamW",     "0.001","10",    "1.0"],
        })
        st.dataframe(df_hp, hide_index=True, use_container_width=True)

# ── Page: Detect ──────────────────────────────────────────────────────────────
elif page == "🔍 Detect":
    st.markdown('<div class="section-head">🔍 Run Detection</div>', unsafe_allow_html=True)

    tab_single, tab_batch, tab_webcam = st.tabs(["Single Image", "Batch Upload", "Demo Gallery"])

    # ── Single Image ──────────────────────────────────────────────────────────
    with tab_single:
        uploaded_img = st.file_uploader(
            "Drop an image here", type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed"
        )

        if uploaded_img:
            image = Image.open(uploaded_img).convert("RGB")
            w, h = image.size

            col_orig, col_res = st.columns(2)
            with col_orig:
                st.markdown("**Original**")
                st.image(image, use_container_width=True)
                st.caption(f"{w} × {h} px · {uploaded_img.size/1024:.1f} KB")

            with col_res:
                st.markdown("**Detection Result**")
                with st.spinner("Running YOLO11n inference…"):
                    try:
                        model_name = custom_model_path if custom_model_path else "yolo11n.pt"
                        model = load_model(model_name)
                        results = model.predict(image, imgsz=img_size, conf=conf_thresh,
                                                iou=iou_thresh, verbose=False)
                        result = results[0]

                        # Default class names
                        class_names = model.names if hasattr(model, 'names') else \
                                      {i: f"cls_{i}" for i in range(80)}
                        if isinstance(class_names, dict):
                            class_names = [class_names.get(i, f"cls_{i}") for i in range(max(class_names)+1)]

                        t_start = time.perf_counter()
                        model.predict(image, imgsz=img_size, conf=conf_thresh,
                                      iou=iou_thresh, verbose=False)
                        latency = (time.perf_counter() - t_start) * 1000

                        result_img, detections = draw_boxes(image, result.boxes, class_names, conf_thresh)
                        st.image(result_img, use_container_width=True)
                        st.caption(f"Latency: {latency:.1f} ms · {len(detections)} detection(s)")

                    except Exception as e:
                        st.error(f"Inference error: {e}")
                        detections = []

            if 'detections' in dir() and detections:
                st.markdown('<div class="section-head">Detections</div>', unsafe_allow_html=True)
                dcols = st.columns(min(len(detections), 4))
                for i, (name, conf, cls_id) in enumerate(detections):
                    color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                    dcols[i % 4].markdown(f"""
                    <div class="det-card">
                        <div class="det-dot" style="background:{color};"></div>
                        <span class="det-name">{name}</span>
                        <span class="det-conf">{conf:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Batch Upload ──────────────────────────────────────────────────────────
    with tab_batch:
        batch_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if batch_files:
            if st.button("🚀 Run Batch Inference"):
                try:
                    model_name = custom_model_path if custom_model_path else "yolo11n.pt"
                    model = load_model(model_name)
                    class_names = model.names if hasattr(model, 'names') else {}
                    if isinstance(class_names, dict):
                        class_names = [class_names.get(i, f"cls_{i}") for i in range(max(class_names)+1)]

                    progress = st.progress(0, text="Processing images…")
                    all_detections = []

                    cols = st.columns(3)
                    for idx, f in enumerate(batch_files):
                        img = Image.open(f).convert("RGB")
                        res = model.predict(img, imgsz=img_size, conf=conf_thresh,
                                            iou=iou_thresh, verbose=False)[0]
                        result_img, dets = draw_boxes(img, res.boxes, class_names, conf_thresh)
                        all_detections.extend(dets)

                        with cols[idx % 3]:
                            st.image(result_img, caption=f"{f.name} — {len(dets)} det.", use_container_width=True)

                        progress.progress((idx+1)/len(batch_files),
                                          text=f"Processed {idx+1}/{len(batch_files)}")

                    progress.empty()

                    if all_detections:
                        st.markdown('<div class="section-head">Batch Summary</div>', unsafe_allow_html=True)
                        counts = Counter(name for name, _, _ in all_detections)
                        df_sum = pd.DataFrame(list(counts.items()), columns=["Class", "Detections"])
                        df_sum = df_sum.sort_values("Detections", ascending=False)
                        st.dataframe(df_sum, hide_index=True, use_container_width=True)

                except Exception as e:
                    st.error(f"Batch inference error: {e}")

    # ── Demo Gallery ──────────────────────────────────────────────────────────
    with tab_webcam:
        st.markdown("""
        <div class="info-box">
            Demo gallery requires a trained model and test images from the wildlife dataset.
            Set your dataset root below to browse and run inference on test images.
        </div>
        """, unsafe_allow_html=True)

        dataset_root = st.text_input("Dataset root path", "/content/wildlife_data")
        if st.button("Load Demo Images"):
            test_dirs = [
                os.path.join(dataset_root, "test", "images"),
                os.path.join(dataset_root, "valid", "images"),
            ]
            found_imgs = []
            for d in test_dirs:
                if os.path.exists(d):
                    found_imgs = glob.glob(f"{d}/**/*.*", recursive=True)[:20]
                    break

            if found_imgs:
                sample = random.sample(found_imgs, min(6, len(found_imgs)))
                try:
                    model_name = custom_model_path if custom_model_path else "yolo11n.pt"
                    model = load_model(model_name)
                    class_names = model.names if hasattr(model, 'names') else {}
                    if isinstance(class_names, dict):
                        class_names = [class_names.get(i, f"cls_{i}") for i in range(max(class_names)+1)]

                    cols = st.columns(3)
                    for idx, img_path in enumerate(sample):
                        img = Image.open(img_path).convert("RGB")
                        res = model.predict(img, imgsz=img_size, conf=conf_thresh,
                                            iou=iou_thresh, verbose=False)[0]
                        result_img, dets = draw_boxes(img, res.boxes, class_names, conf_thresh)
                        with cols[idx % 3]:
                            st.image(result_img,
                                     caption=f"{os.path.basename(img_path)} · {len(dets)} det.",
                                     use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("No images found. Check the dataset path.")

# ── Page: Train ───────────────────────────────────────────────────────────────
elif page == "🏋️ Train":
    st.markdown('<div class="section-head">🏋️ Training Configuration</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Configure and launch model training directly from this UI.
        Requires a valid <code>data.yaml</code> and GPU runtime (CUDA recommended).
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Dataset**")
        data_yaml_path = st.text_input("data.yaml path", "/content/data.yaml")
        dataset_root_t = st.text_input("Dataset root", "/content/wildlife_data")

        st.markdown("**Base Model**")
        base_model = st.selectbox("Pretrained model",
                                  ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"])

    with col2:
        st.markdown("**Hyperparameters**")
        t_epochs  = st.number_input("Epochs",        10, 300, 50, 5)
        t_batch   = st.number_input("Batch size",    4, 64,   16, 4)
        t_imgsz   = st.select_slider("Image size",   [320, 416, 512, 640, 768], value=640)
        t_lr      = st.number_input("Learning rate (lr0)", 1e-5, 0.1, 0.001, format="%.4f")
        t_patience= st.number_input("Early stop patience", 3, 50, 10, 1)
        t_freeze  = st.number_input("Freeze layers", 0, 24, 10, 1)

    st.markdown("**Augmentation**")
    a1, a2, a3, a4 = st.columns(4)
    aug_mosaic = a1.slider("Mosaic",     0.0, 1.0, 1.0, 0.1)
    aug_mixup  = a2.slider("Mixup",      0.0, 0.5, 0.1, 0.05)
    aug_cp     = a3.slider("Copy-Paste", 0.0, 0.5, 0.1, 0.05)
    aug_flip   = a4.slider("Flip LR",   0.0, 1.0, 0.5, 0.1)

    st.divider()

    if st.button("🚀 Start Training"):
        if not os.path.exists(data_yaml_path):
            st.error(f"data.yaml not found at: {data_yaml_path}")
        else:
            try:
                from ultralytics import YOLO
                import torch

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                st.info(f"Training on: **{device.upper()}**")

                model = YOLO(base_model)

                progress_bar = st.progress(0, text="Initializing training…")
                status_box = st.empty()

                # Minimal training call — results streamed to terminal
                with st.spinner("Training in progress… (check terminal for live logs)"):
                    results = model.train(
                        data=data_yaml_path,
                        epochs=t_epochs,
                        imgsz=t_imgsz,
                        batch=t_batch,
                        device=device,
                        optimizer='AdamW',
                        lr0=t_lr,
                        patience=t_patience,
                        freeze=t_freeze,
                        mosaic=aug_mosaic,
                        mixup=aug_mixup,
                        copy_paste=aug_cp,
                        fliplr=aug_flip,
                        project='/content/runs/wildlife_detection',
                        name='wildeye_run',
                        exist_ok=True,
                        verbose=False,
                        plots=True,
                    )

                progress_bar.progress(1.0, text="Training complete!")
                TRAIN_DIR = str(results.save_dir)
                BEST_MODEL = os.path.join(TRAIN_DIR, 'weights', 'best.pt')

                st.success(f"Training complete! Best model: `{BEST_MODEL}`")
                st.session_state['train_dir'] = TRAIN_DIR
                st.session_state['best_model'] = BEST_MODEL

                # Show final metrics
                csv_path = os.path.join(TRAIN_DIR, 'results.csv')
                if os.path.exists(csv_path):
                    df_r = pd.read_csv(csv_path)
                    df_r.columns = df_r.columns.str.strip()
                    last = df_r.iloc[-1]
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    for col, key, label in [
                        (mc1, 'metrics/mAP50(B)',    'mAP@50'),
                        (mc2, 'metrics/mAP50-95(B)', 'mAP@50-95'),
                        (mc3, 'metrics/precision(B)','Precision'),
                        (mc4, 'metrics/recall(B)',   'Recall'),
                    ]:
                        if key in df_r.columns:
                            col.metric(label, f"{float(last[key]):.4f}")

            except Exception as e:
                st.error(f"Training failed: {e}")
                st.exception(e)

    # Show previous training results if available
    if 'train_dir' in st.session_state:
        train_dir = st.session_state['train_dir']
        st.markdown('<div class="section-head">Previous Training Results</div>', unsafe_allow_html=True)
        csv_path = os.path.join(train_dir, 'results.csv')
        if os.path.exists(csv_path):
            df_r = pd.read_csv(csv_path)
            df_r.columns = df_r.columns.str.strip()
            epochs_col = df_r['epoch'] if 'epoch' in df_r.columns else range(len(df_r))

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.patch.set_facecolor('#0d1117')
            for ax in axes:
                ax.set_facecolor('#161b22')
                ax.tick_params(colors='#8b949e')
                ax.spines[:].set_color('#30363d')

            for col in ['train/box_loss', 'val/box_loss']:
                if col in df_r.columns:
                    axes[0].plot(epochs_col, df_r[col], label=col.split('/')[-1], linewidth=2)
            axes[0].set_title('Box Loss', color='#e6edf3', fontweight='bold')
            axes[0].legend(labelcolor='#8b949e', facecolor='#161b22', edgecolor='#30363d')

            for col in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)']:
                if col in df_r.columns:
                    axes[1].plot(epochs_col, df_r[col], label=col.replace('metrics/', ''), linewidth=2)
            axes[1].set_title('mAP Scores', color='#e6edf3', fontweight='bold')
            axes[1].legend(labelcolor='#8b949e', facecolor='#161b22', edgecolor='#30363d')

            for col in ['metrics/precision(B)', 'metrics/recall(B)']:
                if col in df_r.columns:
                    axes[2].plot(epochs_col, df_r[col], label=col.replace('metrics/', ''), linewidth=2)
            axes[2].set_title('Precision & Recall', color='#e6edf3', fontweight='bold')
            axes[2].legend(labelcolor='#8b949e', facecolor='#161b22', edgecolor='#30363d')

            plt.tight_layout()
            st.image(fig_to_pil(fig), use_container_width=True)

# ── Page: Evaluate ────────────────────────────────────────────────────────────
elif page == "📊 Evaluate":
    st.markdown('<div class="section-head">📊 Model Evaluation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Run validation on the test split and inspect per-class metrics.
        Requires a trained model and a valid <code>data.yaml</code>.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        eval_model_path = st.text_input(
            "Model path (.pt)",
            st.session_state.get('best_model', 'yolo11n.pt')
        )
    with col2:
        eval_yaml = st.text_input("data.yaml path", "/content/data.yaml")

    e1, e2, e3 = st.columns(3)
    eval_split = e1.selectbox("Evaluation split", ["test", "val"])
    eval_conf  = e2.slider("Confidence", 0.05, 0.95, 0.25, 0.05, key="eval_conf")
    eval_iou   = e3.slider("IoU", 0.1, 0.9, 0.5, 0.05, key="eval_iou")

    if st.button("▶ Run Evaluation"):
        if not os.path.exists(eval_model_path) and eval_model_path not in ['yolo11n.pt']:
            st.error(f"Model not found: {eval_model_path}")
        elif not os.path.exists(eval_yaml):
            st.error(f"data.yaml not found: {eval_yaml}")
        else:
            try:
                from ultralytics import YOLO
                import torch

                with st.spinner("Evaluating model…"):
                    model = YOLO(eval_model_path)
                    val_results = model.val(
                        data=eval_yaml,
                        split=eval_split,
                        imgsz=640,
                        conf=eval_conf,
                        iou=eval_iou,
                        plots=True,
                        verbose=False,
                    )

                map50   = val_results.box.map50
                map5095 = val_results.box.map
                prec    = float(val_results.box.p.mean())
                rec     = float(val_results.box.r.mean())
                f1      = 2*prec*rec/(prec+rec+1e-9)

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("mAP@50",    f"{map50:.4f}")
                m2.metric("mAP@50-95", f"{map5095:.4f}")
                m3.metric("Precision", f"{prec:.4f}")
                m4.metric("Recall",    f"{rec:.4f}")
                m5.metric("F1-Score",  f"{f1:.4f}")

                # Per-class table
                class_names = model.names
                if isinstance(class_names, dict):
                    class_names = [class_names.get(i, f"cls_{i}") for i in range(len(class_names))]

                per_class = []
                for i, cname in enumerate(class_names):
                    try:
                        p   = float(val_results.box.p[i])    if i < len(val_results.box.p)    else 0
                        r   = float(val_results.box.r[i])    if i < len(val_results.box.r)    else 0
                        a50 = float(val_results.box.ap50[i]) if i < len(val_results.box.ap50) else 0
                        ap  = float(val_results.box.ap[i])   if i < len(val_results.box.ap)   else 0
                    except Exception:
                        p = r = a50 = ap = 0
                    per_class.append({"Class": cname, "Precision": round(p,4),
                                      "Recall": round(r,4), "mAP@50": round(a50,4),
                                      "mAP@50-95": round(ap,4)})

                st.markdown('<div class="section-head">Per-Class Metrics</div>', unsafe_allow_html=True)
                df_pc = pd.DataFrame(per_class)
                st.dataframe(df_pc, hide_index=True, use_container_width=True)

                # Per-class mAP chart
                fig, ax = dark_fig(12, 4)
                colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(df_pc))]
                bars = ax.barh(df_pc['Class'], df_pc['mAP@50'], color=colors, edgecolor='none')
                ax.axvline(df_pc['mAP@50'].mean(), color='#e6a817', linestyle='--', linewidth=1.5,
                           label=f"Mean {df_pc['mAP@50'].mean():.3f}")
                ax.set_title('mAP@50 per Class', fontweight='bold', color='#e6edf3')
                ax.set_xlabel('mAP@50')
                ax.legend(labelcolor='#8b949e', facecolor='#161b22', edgecolor='#30363d')
                plt.tight_layout()
                st.image(fig_to_pil(fig), use_container_width=True)

                # Auto plots from Ultralytics
                val_dir = str(val_results.save_dir)
                for plot_name in ['confusion_matrix_normalized.png', 'PR_curve.png', 'F1_curve.png']:
                    pp = os.path.join(val_dir, plot_name)
                    if os.path.exists(pp):
                        st.image(pp, caption=plot_name, use_container_width=True)

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.exception(e)

# ── Page: Analytics ───────────────────────────────────────────────────────────
elif page == "📈 Analytics":
    st.markdown('<div class="section-head">📈 Dataset Analytics</div>', unsafe_allow_html=True)

    dataset_root_a = st.text_input("Dataset root path", "C:/Users/riyad/Downloads/files", key="analytics_root")
    yaml_path_a    = st.text_input("data.yaml path", "C:/Users/riyad/Downloads/files/data.yaml", key="analytics_yaml")

    if st.button("📊 Load Analytics"):
        if not os.path.exists(dataset_root_a):
            st.error(f"Dataset root not found: {dataset_root_a}")
        else:
            try:
                # Load class names
                class_names_a = []
                if os.path.exists(yaml_path_a):
                    with open(yaml_path_a) as f:
                        cfg = yaml.safe_load(f)
                    class_names_a = cfg.get('names', [])

                # Count images per split
                split_counts = {}
                for split in ['train', 'valid', 'test']:
                    img_dir = os.path.join(dataset_root_a, split, 'images')
                    if os.path.exists(img_dir):
                        split_counts[split] = len(glob.glob(f"{img_dir}/**/*.*", recursive=True))

                total_imgs = sum(split_counts.values())
                mc = st.columns(len(split_counts) + 1)
                for i, (split, cnt) in enumerate(split_counts.items()):
                    mc[i].metric(split.capitalize(), cnt)
                mc[-1].metric("Total Images", total_imgs)

                # Class distribution per split
                st.markdown('<div class="section-head">Class Distribution</div>', unsafe_allow_html=True)

                def count_classes_a(label_dir, names):
                    counter = Counter()
                    for lf in glob.glob(f'{label_dir}/**/*.txt', recursive=True):
                        with open(lf) as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    cid = int(parts[0])
                                    name = names[cid] if cid < len(names) else f"cls_{cid}"
                                    counter[name] += 1
                    return counter

                all_split_data = {}
                for split in ['train', 'valid', 'test']:
                    ld = os.path.join(dataset_root_a, split, 'labels')
                    if os.path.exists(ld):
                        all_split_data[split] = count_classes_a(ld, class_names_a)

                if all_split_data:
                    n = len(all_split_data)
                    fig, axes = plt.subplots(1, n, figsize=(7*n, 5))
                    if n == 1:
                        axes = [axes]
                    fig.patch.set_facecolor('#0d1117')
                    for ax, (split, counter) in zip(axes, all_split_data.items()):
                        ax.set_facecolor('#161b22')
                        ax.tick_params(colors='#8b949e'); ax.spines[:].set_color('#30363d')
                        clrs = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(counter))]
                        bars = ax.bar(list(counter.keys()), list(counter.values()),
                                      color=clrs, edgecolor='none')
                        ax.set_title(f'{split.upper()}', color='#e6edf3', fontweight='bold')
                        ax.set_xlabel('Class', color='#8b949e')
                        ax.set_ylabel('Count', color='#8b949e')
                        ax.tick_params(axis='x', rotation=45)
                        for bar, cnt in zip(bars, counter.values()):
                            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                                    str(cnt), ha='center', va='bottom', fontsize=8, color='#8b949e')
                    plt.tight_layout()
                    st.image(fig_to_pil(fig), use_container_width=True)

                # BBox statistics
                st.markdown('<div class="section-head">Bounding Box Statistics</div>', unsafe_allow_html=True)
                widths, heights, areas = [], [], []
                for lf in glob.glob(f'{dataset_root_a}/train/labels/**/*.txt', recursive=True):
                    with open(lf) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                bw, bh = float(parts[3]), float(parts[4])
                                widths.append(bw); heights.append(bh); areas.append(bw*bh)

                if widths:
                    sm1, sm2, sm3 = st.columns(3)
                    sm1.metric("Avg Width",  f"{np.mean(widths):.3f}")
                    sm2.metric("Avg Height", f"{np.mean(heights):.3f}")
                    sm3.metric("Avg Area",   f"{np.mean(areas):.5f}")

                    fig2, axs = plt.subplots(1, 3, figsize=(15, 4))
                    fig2.patch.set_facecolor('#0d1117')
                    for ax in axs:
                        ax.set_facecolor('#161b22')
                        ax.tick_params(colors='#8b949e')
                        ax.spines[:].set_color('#30363d')

                    axs[0].hist(widths,  bins=40, color='#e6a817', edgecolor='none', alpha=0.85)
                    axs[0].set_title('Width Distribution',  color='#e6edf3', fontweight='bold')
                    axs[0].set_xlabel('Width (norm.)', color='#8b949e')

                    axs[1].hist(heights, bins=40, color='#3fb950', edgecolor='none', alpha=0.85)
                    axs[1].set_title('Height Distribution', color='#e6edf3', fontweight='bold')
                    axs[1].set_xlabel('Height (norm.)', color='#8b949e')

                    axs[2].scatter(widths, heights, alpha=0.15, s=4, color='#58a6ff')
                    axs[2].set_title('Width vs Height',    color='#e6edf3', fontweight='bold')
                    axs[2].set_xlabel('Width',  color='#8b949e')
                    axs[2].set_ylabel('Height', color='#8b949e')

                    plt.tight_layout()
                    st.image(fig_to_pil(fig2), use_container_width=True)

                # Split pie chart
                if split_counts:
                    st.markdown('<div class="section-head">Split Distribution</div>', unsafe_allow_html=True)
                    fig3, ax3 = plt.subplots(figsize=(5, 5))
                    fig3.patch.set_facecolor('#0d1117')
                    ax3.set_facecolor('#0d1117')
                    pie_colors = ['#3fb950', '#58a6ff', '#e6a817']
                    wedges, texts, autotexts = ax3.pie(
                        list(split_counts.values()),
                        labels=list(split_counts.keys()),
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=pie_colors[:len(split_counts)],
                        textprops={'color': '#e6edf3', 'fontsize': 11}
                    )
                    for at in autotexts:
                        at.set_color('#0d1117')
                        at.set_fontweight('bold')
                    ax3.set_title('Dataset Split', color='#e6edf3', fontweight='bold', fontsize=13)
                    st.image(fig_to_pil(fig3), width=350)

            except Exception as e:
                st.error(f"Analytics error: {e}")
                st.exception(e)
