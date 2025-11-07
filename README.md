# Chest X-ray Detection API + UI

A lightweight FastAPI service and web UI to run chest X‑ray classification with optional Grad‑CAM heatmaps.

The app loads a Keras 3 `.keras` model (`ConvMixer_WEIGHTS_BIASES.keras`) and exposes:
- A browser UI at `/` to upload an image and see predictions + heatmaps
- A JSON API with `/health`, `/predict`, and `/predict_visualize`

If the provided `.keras` file contains only weights, please share the model architecture code to load via `model.load_weights(...)`.

## Features

- FastAPI-based inference server
- Simple web UI (drag & drop image upload)
- Top‑k predictions with class names and scores
- Grad‑CAM heatmap + overlay for visual explanation
- Configurable image size, normalization, labels, and thresholds
- Streamlit app for interactive exploration
- Disease class descriptions (see below)

---

## Quick start (Windows PowerShell)

1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) (Optional) Set labels and image size

```powershell
$env:CLASS_LABELS = "NORMAL,PNEUMONIA"  # update to your labels
$env:IMG_SIZE = "224,224"               # or leave unset to infer from model
```

4) Run the server

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

5) Open the UI

Visit http://127.0.0.1:8000/ in your browser. Upload an image and click “Run prediction.”

---

## API Endpoints

### GET `/health`
Returns model status and metadata.

Example response:
```json
{
	"status": "ok",
	"model_loaded": true,
	"input_size": { "width": 224, "height": 224 },
	"num_classes": 2,
	"labels": ["NORMAL", "PNEUMONIA"]
}
```

### POST `/predict`
Accepts `multipart/form-data` with `file` field (image). Returns top predictions and raw probabilities.

PowerShell example:
```powershell
$FilePath = "C:\\path\\to\\xray.jpg"
$File = Get-Item $FilePath
Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict?top_k=3" -Method Post -Form @{ file = $File }
```

### POST `/predict_visualize`
Same as `/predict` but also returns base64‐encoded images: input, heatmap, and overlay.

Python example:
```powershell
python .\client_example.py "C:\\path\\to\\xray.jpg"
```

Example JSON (truncated):
```json
{
	"predicted": { "label": "PNEUMONIA", "score": 0.97 },
	"top": [ { "label": "PNEUMONIA", "score": 0.97 }, { "label": "NORMAL", "score": 0.03 } ],
	"raw": [0.03, 0.97],
	"images": {
		"input": "data:image/png;base64,...",
		"heatmap": "data:image/png;base64,...",
		"overlay": "data:image/png;base64,..."
	}
}
```

---

## Configuration

Set via environment variables (defaults in parentheses):

- `MODEL_PATH` (ConvMixer_WEIGHTS_BIASES.keras) — Path to the Keras 3 model file
- `IMG_SIZE` (inferred or `224,224`) — `W,H` or a single int for square
- `NORMALIZE` (`none`) — `none` or `imagenet`
- `CLASS_LABELS` (unset) — Comma‑separated labels, e.g. `NORMAL,PNEUMONIA`
- `SCORE_THRESHOLD` (`0.0`) — Optional threshold for top prediction
- `FORCE_RGB` (`1`) — Force convert images to RGB

Example (single line):
```powershell
$env:CLASS_LABELS = "NORMAL,PNEUMONIA"; $env:IMG_SIZE = "224,224"; uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## Docker (optional)

```powershell
# Build
docker build -t xray-api .

# Run
docker run --rm -p 8000:8000 xray-api
```

---

## Troubleshooting

- Import errors for `keras`/`tensorflow`: ensure `pip install -r requirements.txt` completed successfully (Keras 3 + TF 2.17).
- Form upload errors: `python-multipart` is required (already in requirements).
- Model fails to load with “weights only”: supply the architecture code to load weights via `model.load_weights(...)`.
- Input size mismatch: set `IMG_SIZE` to the model’s expected size or leave unset to infer.
- Grad‑CAM issues: the app auto‑selects a convolutional layer; if your model is non‑convolutional, Grad‑CAM may not apply.

---

## Notes
-## Disease Class Descriptions

| Class | Description |
|-------|-------------|
| Normal | Healthy chest X-rays showing clear lung fields, no signs of infection or disease. |
| Pneumonia-Bacterial | Localized infection from bacteria; appears as dense, lobar consolidations. |
| Pneumonia-Viral | Non-COVID viral infections (e.g., Influenza, RSV) usually exhibit diffuse, patchy interstitial infiltrates. |
| COVID-19 | Caused by SARS-CoV-2; often presents with bilateral ground-glass opacities, consolidations, and peripheral lung involvement. |
| Tuberculosis | Chronic bacterial infection by Mycobacterium tuberculosis; often affects upper lobes with fibrotic scarring or cavitations. |
| Emphysema | Chronic lung disease under COPD; visible through hyperinflation, flattened diaphragms, and reduced vascular markings. |


- Performance: use CPU by default; enable GPU TensorFlow for acceleration if available.
- Security: this demo doesn’t implement auth/rate limiting; don’t expose publicly without hardening.
- Medical disclaimer: this tool is for research/education only and not a diagnostic device.
