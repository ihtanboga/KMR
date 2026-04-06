# KMR — Kaplan-Meier Curve Reconstructor

**[Live Demo App](https://ihtanboga.github.io/KMR/)**

A web-based tool for digitizing Kaplan-Meier survival curves from published figures. Upload an image (PNG/JPG) or PDF, calibrate the axes, trace the curves — manually or with auto-trace — and export precise (x, y) coordinates as CSV and JSON.

Built for clinical researchers who need to extract individual patient data from published KM plots for meta-analysis, validation, or reanalysis (Guyot et al. 2012 workflow).

---

## Features

- **Image & PDF support** — Upload PNG, JPG, or multi-page PDF. PDF pages are rendered at configurable DPI with optional region cropping.
- **Axis calibration** — Click two reference points per axis and enter their known values. Supports any scale (years, months, days; proportion or percentage).
- **Three tracing modes:**
  - **Freehand (Pen)** — Draw continuously along the curve with the mouse held down. Auto-simplified via Douglas-Peucker.
  - **Click** — Place individual points with precision. Ideal for step-functions and fine adjustments.
  - **Auto-trace** — Click on a curve to pick its color; OpenCV extracts the curve automatically using color segmentation, skeletonization, and seed-based curve following.
- **Editing tools** — Eraser, drag-to-reposition, undo/redo (Ctrl+Z / Ctrl+Shift+Z), clear arm.
- **Multi-arm support** — Trace multiple treatment arms independently with distinct colors.
- **Loupe (magnifier)** — A live zoom panel tracks the cursor position for pixel-level precision. Zoom level adjustable via scroll wheel (2×–15×).
- **Structured metadata** — Enter study name, population (ITT/PP/mITT), endpoint, arm names, and optional Number at Risk (NAR) tables before tracing.
- **Dual export:**
  - **CSV** — `arm, x, y` with a comment header containing study metadata.
  - **JSON** — Full metadata envelope: study, population, endpoint, calibration, NAR, coordinates, and the companion CSV filename.
- **Smart file naming** — Export files are named `Study_Population_Endpoint_Arm1-Arm2.csv/.json`.
- **Neon trace overlay** — Traced curves are displayed in high-contrast neon colors (green, cyan, yellow) with dark outlines, making them easily distinguishable from the underlying figure.
- **Reset** — Full reset button to start over without reloading.

---

## Auto-trace Pipeline

When you click on a curve in auto-trace mode, the backend runs:

1. **Color masking** — HSV-space thresholding around the picked color (adjustable tolerance).
2. **Morphological cleanup** — Close gaps, remove noise.
3. **Skeletonization** — Thin the mask to 1px (scikit-image or OpenCV contrib).
4. **Seed-based curve following** — Starting from the clicked point, follows the nearest skeleton pixel left and right, tolerating gaps up to 60px with linear interpolation.
5. **Douglas-Peucker simplification** — Reduces point count while preserving shape (adjustable epsilon).
6. **Image masking** (optional) — Paint over unwanted regions (legends, annotations) to exclude them from detection.

---

## Installation

```bash
git clone https://github.com/ihtanboga/KMR.git
cd KMR
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5555` in your browser.

### Requirements

- Python 3.9+
- Flask, NumPy, Pandas, OpenCV, scikit-image, Matplotlib, SciPy, lifelines, PyMuPDF

All listed in `requirements.txt`. No frontend build step — pure vanilla HTML/JS/Canvas.

---

## Usage

### 1. Study Info
Enter study metadata (name, population, endpoint), define treatment arms, and optionally fill in Number at Risk tables. Upload the KM curve image or PDF.

### 2. X-Axis Calibration
Click two points on the X-axis and enter their known values (e.g., 0 and 6 years).

### 3. Y-Axis Calibration
Click two points on the Y-axis and enter their known values (e.g., 0 and 100%).

### 4. Trace & Export
Select an arm, choose a tool (Pen / Click / Auto-trace), and trace the curve. Use the loupe panel for precision. When done, click **Download CSV + JSON**.

---

## Export Format

### CSV
```csv
# study: CLOSURE-AF | population: ITT | endpoint: CV Death + Stroke + SE
arm,x,y
LAA Closure,0.000000,0.000000
LAA Closure,0.082353,0.012500
...
```

### JSON
```json
{
  "study": "CLOSURE-AF",
  "population": "ITT",
  "endpoint": "CV Death + Stroke + SE",
  "csv_file": "CLOSUREAF_ITT_CVDeathStrokeSE_LAAClosure-MedicalTherapy.csv",
  "calibration": {
    "x": { "pointA": 0, "pointB": 6 },
    "y": { "pointA": 0, "pointB": 100 }
  },
  "arms": [
    {
      "name": "LAA Closure",
      "n_points": 142,
      "nar": [
        { "time": 0, "n": 446 },
        { "time": 1, "n": 304 }
      ],
      "coordinates": [
        { "x": 0.0, "y": 0.0 },
        { "x": 0.082, "y": 0.0125 }
      ]
    }
  ]
}
```

---

## IPD Reconstruction (Optional)

The exported coordinates can be fed into the Guyot et al. (2012) algorithm to reconstruct Individual Patient Data. A Python implementation is included in `guyot.py`:

```python
from guyot import reconstruct_arm
import pandas as pd

clicks = pd.read_csv("coordinates.csv")
nar = pd.DataFrame({"time": [0,1,2,3], "NAR": [446,304,202,117]})
ipd = reconstruct_arm(clicks[["time","survival"]], nar, arm_name="Treatment")
# ipd has columns: arm, time, status (1=event, 0=censored)
```

---

## Project Structure

```
KMR/
├── app.py              # Flask server with upload, image serving, auto-trace, export APIs
├── autotrace.py        # OpenCV pipeline: color mask → skeleton → curve following
├── guyot.py            # Guyot et al. 2012 IPD reconstruction (Python port of R reconstructKM)
├── templates/
│   └── index.html      # Full UI: canvas, loupe, step wizard, tools, export
└── requirements.txt
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `F` | Freehand tool |
| `C` | Click tool |
| `A` | Auto-trace tool |
| `E` | Eraser tool |
| `D` | Drag tool |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |
| Scroll wheel | Adjust loupe zoom |

---

## References

- Guyot P, Ades AE, Ouwens MJ, Welton NJ. Enhanced secondary analysis of survival data: reconstructing the data from published Kaplan-Meier survival curves. *BMC Med Res Methodol*. 2012;12:9.
- Wei Y, Royston P. Reconstructing time-to-event data from published Kaplan-Meier curves. *Stata J*. 2017;17(4):786–802.

---

## License

MIT
