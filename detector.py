"""
Smile detection using MediaPipe FaceLandmarker + a Logistic Regression
classifier trained on 8 scale-invariant ratio features (mouth width,
openness, eye-to-mouth distance, etc.). Supports multiple faces per image.
"""

import json
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── MediaPipe FaceLandmarker setup ────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")

_base_opts = mp_base.BaseOptions(model_asset_path=_MODEL_PATH)
_face_opts = mp_vision.FaceLandmarkerOptions(
    base_options=_base_opts,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=10,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
_LANDMARKER = mp_vision.FaceLandmarker.create_from_options(_face_opts)

_EYE_SLOTS = [
    33,   # left  eye outer corner
    133,  # left  eye inner corner
    159,  # left  eye top
    145,  # left  eye bottom
    362,  # right eye inner corner
    263,  # right eye outer corner
    386,  # right eye top
    374,  # right eye bottom
]
_MOUTH_SLOTS = [
    61,   # mouth left  corner
    291,  # mouth right corner
    13,   # mouth top   center
    14,   # mouth bottom center
    78,   # mouth top   left
    308,  # mouth top   right
    17,   # lower lip   bottom
]
_ALL_SLOTS = _EYE_SLOTS + _MOUTH_SLOTS

KP_NAMES = [
    "left_eye_outer",  "left_eye_inner",  "left_eye_top",   "left_eye_bottom",
    "right_eye_inner", "right_eye_outer", "right_eye_top",  "right_eye_bottom",
    "mouth_left",      "mouth_right",     "mouth_top_ctr",  "mouth_bot_ctr",
    "mouth_top_left",  "mouth_top_right", "lower_lip",
]

_MIN_INTER_EYE_PX = 30
_MIN_CONFIDENCE = 0.60


def _load_training_data(path: str = "data/facial_keypoints.json") -> list:
    with open(path, "r") as f:
        raw = json.load(f)
    samples = []
    for entry in raw["train"]:
        kps   = [entry[i] for i in range(1, 16)]
        label = entry[16]["smile"]
        samples.append({"keypoints": kps, "smile": label})
    return samples


_TRAINING_DATA = _load_training_data()


def _extract_features_from_training_kps(kps: list) -> np.ndarray:
    """Extract 8 ratio-based features from training-data keypoints (kp0-9 = eyes, kp10-14 = mouth)."""
    pts = np.array(kps, dtype=float)
    eye_pts   = pts[:10]
    mouth_pts = pts[10:]

    eye_span_x  = eye_pts[:, 0].max()  - eye_pts[:, 0].min()
    eye_top_y   = eye_pts[:, 1].min()
    eye_bot_y   = eye_pts[:, 1].max()
    eye_ctr_y   = eye_pts[:, 1].mean()

    mouth_w     = mouth_pts[:, 0].max() - mouth_pts[:, 0].min()
    mouth_h     = mouth_pts[:, 1].max() - mouth_pts[:, 1].min()
    mouth_top_y = mouth_pts[:, 1].min()
    mouth_bot_y = mouth_pts[:, 1].max()
    mouth_ctr_y = mouth_pts[:, 1].mean()

    face_w      = pts[:, 0].max() - pts[:, 0].min()
    face_h      = mouth_bot_y - eye_top_y

    eye_to_mouth = mouth_ctr_y - eye_ctr_y

    return np.array([
        mouth_w      / (eye_span_x    + 1e-6),
        mouth_h      / (mouth_w       + 1e-6),
        mouth_w      / (face_w        + 1e-6),
        eye_to_mouth / (face_h        + 1e-6),
        mouth_top_y  / (face_h        + 1e-6),
        (mouth_bot_y - eye_bot_y) / (face_h + 1e-6),
        mouth_h      / (eye_to_mouth  + 1e-6),
        (mouth_ctr_y - eye_bot_y) / (face_h + 1e-6),
    ])


def _extract_features_from_mediapipe_kps(kps: list) -> np.ndarray:
    """Extract the same 8 ratio-based features from MediaPipe keypoints (kps[0-7] = eyes, kps[8-14] = mouth)."""
    pts = np.array(kps, dtype=float)
    eye_pts   = pts[:8]
    mouth_pts = pts[8:]

    eye_span_x  = abs(pts[0, 0] - pts[5, 0])
    eye_top_y   = min(pts[2, 1], pts[6, 1])
    eye_bot_y   = max(pts[3, 1], pts[7, 1])
    eye_ctr_y   = eye_pts[:, 1].mean()

    mouth_w     = abs(pts[8, 0] - pts[9, 0])
    mouth_top_y = min(pts[10, 1], pts[12, 1], pts[13, 1])
    mouth_bot_y = pts[14, 1]
    mouth_h     = mouth_bot_y - mouth_top_y
    mouth_ctr_y = mouth_pts[:, 1].mean()

    face_w      = eye_span_x
    face_h      = mouth_bot_y - eye_top_y

    eye_to_mouth = mouth_ctr_y - eye_ctr_y

    return np.array([
        mouth_w      / (eye_span_x    + 1e-6),
        mouth_h      / (mouth_w       + 1e-6),
        mouth_w      / (face_w        + 1e-6),
        eye_to_mouth / (face_h        + 1e-6),
        mouth_top_y  / (face_h        + 1e-6),
        (mouth_bot_y - eye_bot_y) / (face_h + 1e-6),
        mouth_h      / (eye_to_mouth  + 1e-6),
        (mouth_ctr_y - eye_bot_y) / (face_h + 1e-6),
    ])


def _build_classifier():
    X = np.array([_extract_features_from_training_kps(s["keypoints"])
                  for s in _TRAINING_DATA])
    y = np.array([int(s["smile"]) for s in _TRAINING_DATA])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_scaled, y)
    return clf, scaler


_CLASSIFIER, _SCALER = _build_classifier()

_FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), "data", "feedback.json")


def _load_feedback() -> list:
    """Load saved feedback entries from disk."""
    if not os.path.exists(_FEEDBACK_PATH):
        return []
    with open(_FEEDBACK_PATH, "r") as f:
        return json.load(f)


def _save_feedback(entries: list) -> None:
    """Persist feedback entries to disk."""
    with open(_FEEDBACK_PATH, "w") as f:
        json.dump(entries, f)


def add_feedback(keypoints: list, correct_label: bool) -> dict:
    """Save a user-corrected label for a set of MediaPipe keypoints."""
    entries = _load_feedback()
    entries.append({
        "keypoints": keypoints,
        "smile": bool(correct_label),
    })
    _save_feedback(entries)
    return {
        "feedback_count": len(entries),
        "message": f"Feedback saved. Total feedback samples: {len(entries)}.",
    }


def retrain(min_feedback: int = 1) -> dict:
    """
    Retrain the classifier on original data + feedback.
    Feedback samples are weighted ×5 to give corrections real impact.
    """
    global _CLASSIFIER, _SCALER

    feedback = _load_feedback()
    if len(feedback) < min_feedback:
        return {
            "retrained": False,
            "message": f"Need at least {min_feedback} feedback sample(s). "
                       f"Currently have {len(feedback)}.",
            "feedback_count": len(feedback),
        }

    X_orig = np.array([_extract_features_from_training_kps(s["keypoints"])
                       for s in _TRAINING_DATA])
    y_orig = np.array([int(s["smile"]) for s in _TRAINING_DATA])

    X_fb = np.array([_extract_features_from_mediapipe_kps(e["keypoints"])
                     for e in feedback])
    y_fb = np.array([int(e["smile"]) for e in feedback])

    feedback_weight = 5
    sample_weight = np.concatenate([
        np.ones(len(X_orig)),
        np.full(len(X_fb), feedback_weight),
    ])

    X_all = np.vstack([X_orig, X_fb])
    y_all = np.concatenate([y_orig, y_fb])

    new_scaler = StandardScaler()
    X_scaled   = new_scaler.fit_transform(X_all)

    new_clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    new_clf.fit(X_scaled, y_all, sample_weight=sample_weight)

    orig_acc = new_clf.score(new_scaler.transform(X_orig), y_orig)
    fb_acc   = new_clf.score(new_scaler.transform(X_fb),   y_fb)

    _CLASSIFIER = new_clf
    _SCALER     = new_scaler

    return {
        "retrained": True,
        "message": "Model retrained successfully.",
        "original_train_samples": len(X_orig),
        "feedback_samples": len(X_fb),
        "feedback_weight": feedback_weight,
        "original_train_accuracy": round(orig_acc, 4),
        "feedback_accuracy": round(fb_acc, 4),
    }


def get_feedback_count() -> int:
    return len(_load_feedback())


def _classify_mediapipe_kps(kps: list) -> tuple:
    """Returns (is_smile: bool, confidence: float)."""
    vec = _extract_features_from_mediapipe_kps(kps).reshape(1, -1)
    vec_scaled = _SCALER.transform(vec)
    pred  = _CLASSIFIER.predict(vec_scaled)[0]
    proba = _CLASSIFIER.predict_proba(vec_scaled)[0]
    return bool(pred), float(proba[pred])


def _extract_keypoints_for_face(lm, w: int, h: int) -> dict:
    """Extract keypoints and inter-eye distance for one face landmark set."""
    keypoints = []
    for idx in _ALL_SLOTS:
        keypoints.append([round(lm[idx].x * w, 2),
                          round(lm[idx].y * h, 2)])

    inter_eye_px = float(np.linalg.norm(
        np.array(keypoints[5]) - np.array(keypoints[0])
    ))

    if inter_eye_px < _MIN_INTER_EYE_PX:
        return {
            "keypoints": None,
            "inter_eye_px": round(inter_eye_px, 1),
            "error": (
                f"Face is too small (inter-eye = {inter_eye_px:.0f} px, "
                f"min = {_MIN_INTER_EYE_PX} px). "
                "Please move closer to the camera."
            ),
        }

    return {"keypoints": keypoints, "inter_eye_px": round(inter_eye_px, 1),
            "error": None}


def _build_face_result(kps: list) -> dict:
    """Classify a single face's keypoints and return a result dict."""
    is_smile, confidence = _classify_mediapipe_kps(kps)

    pts = np.array(kps, dtype=float)
    eye_span = abs(pts[0, 0] - pts[5, 0])
    mouth_w  = abs(pts[8, 0] - pts[9, 0])
    mouth_h  = abs(pts[14, 1] - min(pts[10, 1], pts[12, 1], pts[13, 1]))
    face_h   = pts[14, 1] - min(pts[2, 1], pts[6, 1])
    eye_to_mouth = pts[8:, 1].mean() - pts[:8, 1].mean()

    features = {
        "mouth_w_eye_span_ratio": round(mouth_w / (eye_span + 1e-6), 4),
        "mouth_openness":         round(mouth_h / (mouth_w  + 1e-6), 4),
        "eye_to_mouth_face_ratio":round(eye_to_mouth / (face_h + 1e-6), 4),
    }

    named_kps = {name: kps[i] for i, name in enumerate(KP_NAMES)}

    low_confidence = confidence < _MIN_CONFIDENCE
    warning = (
        f"Low confidence ({confidence:.0%}). "
        "Try facing the camera directly or moving closer."
        if low_confidence else None
    )

    return {
        "smile": bool(is_smile),
        "confidence": round(confidence, 4),
        "low_confidence": low_confidence,
        "keypoints": named_kps,
        "features": features,
        "error": None,
        "warning": warning,
    }


def detect_smile(image_bytes: bytes) -> dict:
    """Run the full pipeline on image bytes. Returns a faces list, one entry per detected face."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"faces": [], "face_count": 0,
                "error": "Could not decode image."}

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = _LANDMARKER.detect(mp_image)

    if not result.face_landmarks:
        return {"faces": [], "face_count": 0,
                "error": "No face detected in the image."}

    faces = []
    for face_idx, lm in enumerate(result.face_landmarks):
        extraction = _extract_keypoints_for_face(lm, w, h)
        if extraction["error"]:
            faces.append({
                "face_index": face_idx,
                "smile": None,
                "confidence": 0.0,
                "low_confidence": False,
                "inter_eye_px": extraction["inter_eye_px"],
                "keypoints": {},
                "features": {},
                "error": extraction["error"],
                "warning": None,
            })
        else:
            face_result = _build_face_result(extraction["keypoints"])
            face_result["face_index"]   = face_idx
            face_result["inter_eye_px"] = extraction["inter_eye_px"]
            faces.append(face_result)

    return {
        "faces": faces,
        "face_count": len(faces),
        "error": None,
    }
