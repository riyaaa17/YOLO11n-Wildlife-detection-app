# WildEye — Wildlife Detection Streamlit App

A production-grade Streamlit UI for YOLO11n wildlife object detection.

## Features

| Page | What it does |
|---|---|
| **Overview** | Model info, pipeline steps, hyperparameter table |
| **Detect** | Single image / batch inference / demo gallery |
| **Train** | Configure & launch training from the UI |
| **Evaluate** | Run validation, per-class metrics, confusion matrix |
| **Analytics** | Dataset stats, class distribution, bbox analysis |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. (Optional) Use a trained model
- Train via the **Train** page inside the app, OR
- Load an existing `.pt` file via the sidebar → "Load custom .pt file"

## Running on Google Colab

```python
# Install
!pip install streamlit ultralytics pyngrok -q

# Start tunnel
from pyngrok import ngrok
import subprocess, threading

def run():
    subprocess.run(["streamlit", "run", "app.py",
                    "--server.port=8501", "--server.headless=true"])

t = threading.Thread(target=run, daemon=True)
t.start()

import time; time.sleep(4)
public_url = ngrok.connect(8501)
print("WildEye running at:", public_url)
```

## Sidebar Controls

- **Model**: Choose pretrained YOLO11n or upload your own `.pt`
- **Confidence threshold**: Minimum detection confidence (default 0.25)
- **IoU threshold**: NMS overlap threshold (default 0.45)
- **Image size**: Inference resolution (default 640)

## Dataset Expected Structure

```
/content/wildlife_data/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```
