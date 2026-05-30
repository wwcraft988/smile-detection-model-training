"""
app.py  –  Flask web server for the smile detection project.
"""

from flask import Flask, request, jsonify, render_template

from detector import detect_smile, add_feedback, retrain, get_feedback_count

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Detection ─────────────────────────────────────────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    if "photo" not in request.files:
        return jsonify({"error": "No file uploaded. Use field name 'photo'."}), 400

    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not _allowed(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    image_bytes = file.read()
    result = detect_smile(image_bytes)
    return jsonify(result), 200


# ── Feedback ──────────────────────────────────────────────────────────────────
@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Save a user correction for a specific face.

    JSON body:
        {
            "keypoints": [[x,y], ...],   // 15 pairs from the face's keypoints
            "correct_label": true|false, // what the correct answer actually is
            "face_index": 0              // which face (optional, default 0)
        }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "JSON body required."}), 400

    keypoints     = body.get("keypoints")
    correct_label = body.get("correct_label")

    if keypoints is None or correct_label is None:
        return jsonify({"error": "Both 'keypoints' and 'correct_label' are required."}), 400

    if not isinstance(keypoints, list) or len(keypoints) != 15:
        return jsonify({"error": "'keypoints' must be a list of 15 [x,y] pairs."}), 400

    result = add_feedback(keypoints, correct_label)
    return jsonify(result), 200


# ── Retrain ───────────────────────────────────────────────────────────────────
@app.route("/retrain", methods=["POST"])
def retrain_model():
    """
    Retrain the classifier on original data + all saved feedback.
    No body required.
    """
    result = retrain(min_feedback=1)
    return jsonify(result), 200


# ── Stats ─────────────────────────────────────────────────────────────────────
@app.route("/stats", methods=["GET"])
def stats():
    """Return current model stats."""
    return jsonify({
        "feedback_count": get_feedback_count(),
    }), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
