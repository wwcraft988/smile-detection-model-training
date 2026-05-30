"""Flask web server for the smile detection project."""

from flask import Flask, request, jsonify, render_template
from detector import detect_smile, add_feedback, retrain, get_feedback_count

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "photo" not in request.files:
        return jsonify({"error": "No file uploaded. Use field name 'photo'."}), 400
    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not _allowed(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400
    result = detect_smile(file.read())
    return jsonify(result), 200


@app.route("/feedback", methods=["POST"])
def feedback():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "JSON body required."}), 400
    keypoints     = body.get("keypoints")
    correct_label = body.get("correct_label")
    if keypoints is None or correct_label is None:
        return jsonify({"error": "Both 'keypoints' and 'correct_label' are required."}), 400
    if not isinstance(keypoints, list) or len(keypoints) != 15:
        return jsonify({"error": "'keypoints' must be a list of 15 [x,y] pairs."}), 400
    return jsonify(add_feedback(keypoints, correct_label)), 200


@app.route("/retrain", methods=["POST"])
def retrain_model():
    return jsonify(retrain(min_feedback=1)), 200


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({"feedback_count": get_feedback_count()}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
