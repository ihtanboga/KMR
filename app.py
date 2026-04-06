"""
KM: Kaplan-Meier Curve Reconstructor
Flask web application - digitize KM curves and export (x, y) coordinates
"""

import os
import uuid
import io
from urllib.parse import quote

from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

SESSIONS = {}
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
PDF_DPI = 170


def build_session(filepath, filename):
    ext = os.path.splitext(filename)[1].lower()
    source_type = "pdf" if ext == ".pdf" else "image"
    session = {
        "file_path": filepath,
        "filename": filename,
        "source_type": source_type,
    }

    if source_type == "image":
        session["image_path"] = filepath
        return session

    import fitz

    with fitz.open(filepath) as doc:
        session["page_count"] = len(doc)

    return session


def parse_crop(value):
    """Parse crop coordinates as [x, y, w, h]."""
    if not value:
        return None

    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 4:
            return None
        try:
            nums = [int(round(float(p))) for p in parts]
        except ValueError:
            return None
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            nums = [int(round(float(v))) for v in value]
        except (TypeError, ValueError):
            return None
    else:
        return None

    x, y, w, h = nums
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def crop_image(image, crop):
    """Crop a BGR image using [x, y, w, h] coordinates."""
    if image is None or crop is None:
        return image

    height, width = image.shape[:2]
    x, y, w, h = crop
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    x2 = max(x + 1, min(width, x + w))
    y2 = max(y + 1, min(height, y + h))
    return image[y:y2, x:x2].copy()


def render_pdf_page(filepath, page_number=1, crop=None, dpi=PDF_DPI):
    """Render a PDF page to a BGR image."""
    import fitz
    import cv2
    import numpy as np

    with fitz.open(filepath) as doc:
        index = max(0, min(len(doc) - 1, int(page_number) - 1))
        page = doc[index]
        scale = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)

    image = cv2.imdecode(np.frombuffer(pix.tobytes("png"), dtype=np.uint8), cv2.IMREAD_COLOR)
    return crop_image(image, crop)


def get_session_image(session, page_number=1, crop=None):
    """Return the session source as a BGR image, rendering PDFs as needed."""
    import cv2

    if session["source_type"] == "pdf":
        return render_pdf_page(session["file_path"], page_number=page_number, crop=crop)

    image = cv2.imread(session["file_path"])
    return crop_image(image, crop)


def get_session(session_id):
    """Return an upload session, recovering it from disk if memory was reset."""
    session = SESSIONS.get(session_id)
    if session:
        return session

    prefix = f"{session_id}."
    for filename in os.listdir(UPLOAD_DIR):
        if not filename.startswith(prefix):
            continue

        filepath = os.path.join(UPLOAD_DIR, filename)
        if not os.path.isfile(filepath):
            continue

        session = build_session(filepath, filename)
        SESSIONS[session_id] = session
        return session

    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400

    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    session_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(f.filename)[1] or ".png"
    filename = f"{session_id}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    f.save(filepath)

    session = build_session(filepath, filename)
    SESSIONS[session_id] = session

    image_url = f"/api/image/{session_id}"
    if session["source_type"] == "pdf":
        image_url += "?page=1"

    payload = {
        "session_id": session_id,
        "image_url": image_url,
        "source_type": session["source_type"],
        "filename": filename,
    }

    if session["source_type"] == "pdf":
        payload["page_count"] = session["page_count"]

    return jsonify(payload)


@app.route("/api/image/<session_id>")
def serve_image(session_id):
    import cv2

    session = get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    page_number = request.args.get("page", default=1, type=int)
    crop = parse_crop(request.args.get("crop"))
    image = get_session_image(session, page_number=page_number, crop=crop)
    if image is None:
        return jsonify({"error": "Could not render image"}), 500

    if session["source_type"] == "image" and crop is None:
        return send_file(session["file_path"])

    ok, png = cv2.imencode(".png", image)
    if not ok:
        return jsonify({"error": "Could not encode image"}), 500

    return send_file(
        io.BytesIO(png.tobytes()),
        mimetype="image/png",
        download_name=f"{session_id}_{quote(session['filename'])}.png",
    )


@app.route("/api/autotrace", methods=["POST"])
def api_autotrace():
    """Auto-trace a curve by color."""
    import cv2
    from autotrace import autotrace

    data = request.get_json()
    session_id = data.get("session_id")
    target_rgb = tuple(data.get("color", [255, 0, 0]))
    tolerance = data.get("tolerance", 30)
    epsilon = data.get("epsilon", 2.0)
    calibration = data.get("calibration")
    x_pixel_range = data.get("x_pixel_range")
    image_masks = data.get("image_masks", [])
    seed_point = data.get("seed_point")
    page_number = data.get("page_number", 1)
    crop = parse_crop(data.get("crop"))

    if not isinstance(calibration, dict) or "x" not in calibration or "y" not in calibration:
        return jsonify({"error": "Calibration is required before auto-trace"}), 400

    session = get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    image = get_session_image(session, page_number=page_number, crop=crop)
    if image is None:
        return jsonify({"error": "Could not read image"}), 500

    if image_masks:
        for stroke in image_masks:
            points = stroke.get("points", [])
            size = max(1, int(round(stroke.get("size", 24))))
            if not points:
                continue

            pts = [
                (int(round(p[0])), int(round(p[1])))
                for p in points
                if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
            if not pts:
                continue

            radius = max(1, size // 2)
            if len(pts) == 1:
                cv2.circle(image, pts[0], radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                continue

            for p1, p2 in zip(pts, pts[1:]):
                cv2.line(image, p1, p2, (255, 255, 255), thickness=size, lineType=cv2.LINE_AA)
            cv2.circle(image, pts[0], radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(image, pts[-1], radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    result = autotrace(
        image, target_rgb, calibration,
        tolerance=tolerance, epsilon=epsilon,
        x_pixel_range=tuple(x_pixel_range) if x_pixel_range else None,
        seed_point=tuple(seed_point) if seed_point else None,
    )

    return jsonify(result)


@app.route("/api/export", methods=["POST"])
def api_export():
    """Export calibrated (x, y) coordinates as CSV."""
    data = request.get_json()
    arms = data.get("arms", [])

    # Build CSV in memory
    import csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["arm", "x", "y"])

    for arm in arms:
        name = arm.get("name", "Arm")
        points = arm.get("points", [])  # already calibrated [[x, y], ...]
        for p in points:
            writer.writerow([name, round(p[0], 6), round(p[1], 6)])

    buf.seek(0)
    result_id = str(uuid.uuid4())[:8]

    return jsonify({
        "result_id": result_id,
        "csv_data": buf.getvalue(),
        "n_points": sum(len(a.get("points", [])) for a in arms),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5555)
