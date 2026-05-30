"""
detector.py
-----------
Smile detection using MediaPipe FaceLandmarker (Tasks API v0.10+).

Approach
--------
The training data (facial_keypoints.json) stores 15 keypoints in a custom
face-relative coordinate space split into two groups:
  kp0-9  : eye region  (y < ~25 in the ~64-px space)
  kp10-14: mouth region (y > ~30)

The classifier is trained on RATIO-BASED features that are invariant to
scale, translation, and coordinate-system orientation:

  Feature 1: mouth_w / eye_span_x        (smile widens the mouth)
  Feature 2: mouth_h / mouth_w           (mouth openness)
  Feature 3: mouth_w / face_w            (mouth width relative to face)
  Feature 4: eye_to_mouth_y / face_h     (smile lifts cheeks, shrinks lower face)
  Feature 5: mouth_top_y / face_h        (mouth position in face)
  Feature 6: lower_face_h / face_h       (lower face proportion)
  Feature 7: mouth_h / eye_to_mouth_y    (mouth height relative to eye-mouth gap)
  Feature 8: mouth_ctr_y_rel_eye / face_h

These same ratios are computed from MediaPipe landmarks using the
equivalent anatomical points:

  eye_span_x  = |lm[33].x  - lm[263].x|   outer eye corners
  eye_top_y   = min(lm[159].y, lm[386].y)  top of eyes
  eye_bot_y   = max(lm[145].y, lm[374].y)  bottom of eyes
  eye_ctr_y   = mean of all 8 eye landmark y values
  mouth_w     = |lm[61].x  - lm[291].x|    mouth corners
  mouth_top_y = min(lm[13].y, lm[78].y, lm[308].y)
  mouth_bot_y = lm[17].y                   lower lip bottom
  mouth_h     = mouth_bot_y - mouth_top_y
  mouth_ctr_y = mean of all 7 mouth landmark y values
  face_h      = mouth_bot_y - eye_top_y
  face_w      = |lm[33].x  - lm[263].x|  (same as eye_span for this dataset)
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
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
_LANDMARKER = mp_vision.FaceLandmarker.create_from_options(_face_opts)

# MediaPipe landmark indices we extract
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

# Minimum inter-eye distance in pixels for reliable landmark detection
_MIN_INTER_EYE_PX = 30
# Minimum classifier confidence to return a definitive answer
_MIN_CONFIDENCE = 0.60


# ── Training data ─────────────────────────────────────────────────────────────
def _load_training_data(path: str = "data/facial_keypoints.json") -> list:
    with open(path, "r") as f:
        raw = json.load(f)
    samples = []
    for entry in raw["train"]:
        # entry[0]      → id string
        # entry[1..15]  → [x, y] keypoints (15 total)
        # entry[16]     → {"smile": bool}
        kps   = [entry[i] for i in range(1, 16)]
        label = entry[16]["smile"]
        samples.append({"keypoints": kps, "smile": label})
    return samples


_TRAINING_DATA = _load_training_data()


# ── Feature extraction ────────────────────────────────────────────────────────
def _extract_features_from_training_kps(kps: list) -> np.ndarray:
    """
    Extract 8 ratio-based features from training-data keypoints.
    kp0-9  = eye region, kp10-14 = mouth region.
    """
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
    face_h      = mouth_bot_y - eye_top_y          # eye-top to mouth-bottom

    eye_to_mouth = mouth_ctr_y - eye_ctr_y

    return np.array([
        mouth_w     / (eye_span_x    + 1e-6),   # f1: mouth width / eye span
        mouth_h     / (mouth_w       + 1e-6),   # f2: mouth openness
        mouth_w     / (face_w        + 1e-6),   # f3: mouth width / face width
        eye_to_mouth / (face_h       + 1e-6),   # f4: eye-to-mouth / face height
        mouth_top_y  / (face_h       + 1e-6),   # f5: mouth top position
        (mouth_bot_y - eye_bot_y) / (face_h + 1e-6),  # f6: lower face proportion
        mouth_h     / (eye_to_mouth  + 1e-6),   # f7: mouth height / eye-to-mouth
        (mouth_ctr_y - eye_bot_y) / (face_h + 1e-6),  # f8: mouth center rel eye bottom
    ])


def _extract_features_from_mediapipe_kps(kps: list) -> np.ndarray:
    """
    Extract the same 8 ratio-based features from MediaPipe keypoints.
    kps[0-7]  = eye landmarks (_EYE_SLOTS order)
    kps[8-14] = mouth landmarks (_MOUTH_SLOTS order)

    Mapping to training-data equivalents:
      eye_span_x  = |kps[0].x - kps[5].x|   (outer eye corners: lm33, lm263)
      eye_top_y   = min(kps[2].y, kps[6].y)  (eye tops: lm159, lm386)
      eye_bot_y   = max(kps[3].y, kps[7].y)  (eye bottoms: lm145, lm374)
      eye_ctr_y   = mean of kps[0-7] y
      mouth_w     = |kps[8].x - kps[9].x|    (mouth corners: lm61, lm291)
      mouth_top_y = min(kps[10].y, kps[12].y, kps[13].y)  (lm13, lm78, lm308)
      mouth_bot_y = kps[14].y                (lower lip: lm17)
      mouth_h     = mouth_bot_y - mouth_top_y
      mouth_ctr_y = mean of kps[8-14] y
      face_w      = eye_span_x  (same as training)
      face_h      = mouth_bot_y - eye_top_y
    """
    pts = np.array(kps, dtype=float)
    eye_pts   = pts[:8]
    mouth_pts = pts[8:]

    eye_span_x  = abs(pts[0, 0] - pts[5, 0])    # lm33 to lm263
    eye_top_y   = min(pts[2, 1], pts[6, 1])      # lm159, lm386
    eye_bot_y   = max(pts[3, 1], pts[7, 1])      # lm145, lm374
    eye_ctr_y   = eye_pts[:, 1].mean()

    mouth_w     = abs(pts[8, 0] - pts[9, 0])     # lm61 to lm291
    mouth_top_y = min(pts[10, 1], pts[12, 1], pts[13, 1])  # lm13, lm78, lm308
    mouth_bot_y = pts[14, 1]                      # lm17
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


# ── Classifier ────────────────────────────────────────────────────────────────
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

# ── Feedback store ────────────────────────────────────────────────────────────
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
    """
    Save a user-corrected label for a set of MediaPipe keypoints.

    Parameters
    ----------
    keypoints     : 15 [x, y] pairs from extract_keypoints_from_image()
    correct_label : True = smile, False = no smile

    Returns
    -------
    dict with feedback_count and message
    """
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
    Retrain the classifier using original training data + all feedback.

    The feedback samples are weighted more heavily (×5) so even a small
    number of corrections can shift the decision boundary.

    Parameters
    ----------
    min_feedback : minimum feedback samples required to retrain

    Returns
    -------
    dict with accuracy stats and sample counts
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

    # ── Build feature matrix from original training data ──────────────────
    X_orig = np.array([_extract_features_from_training_kps(s["keypoints"])
                       for s in _TRAINING_DATA])
    y_orig = np.array([int(s["smile"]) for s in _TRAINING_DATA])

    # ── Build feature matrix from feedback (MediaPipe keypoints) ─────────
    X_fb = np.array([_extract_features_from_mediapipe_kps(e["keypoints"])
                     for e in feedback])
    y_fb = np.array([int(e["smile"]) for e in feedback])

    # Weight feedback samples ×5 so corrections have real impact
    feedback_weight = 5
    sample_weight = np.concatenate([
        np.ones(len(X_orig)),
        np.full(len(X_fb), feedback_weight),
    ])

    X_all = np.vstack([X_orig, X_fb])
    y_all = np.concatenate([y_orig, y_fb])

    # ── Fit new scaler + classifier ───────────────────────────────────────
    new_scaler = StandardScaler()
    X_scaled   = new_scaler.fit_transform(X_all)

    new_clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    new_clf.fit(X_scaled, y_all, sample_weight=sample_weight)

    # ── Evaluate on original training set (unweighted) ───────────────────
    X_orig_scaled = new_scaler.transform(X_orig)
    orig_acc = new_clf.score(X_orig_scaled, y_orig)

    # ── Evaluate on feedback set ──────────────────────────────────────────
    X_fb_scaled = new_scaler.transform(X_fb)
    fb_acc = new_clf.score(X_fb_scaled, y_fb)

    # ── Swap in the new model ─────────────────────────────────────────────
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
    """Return the number of saved feedback samples."""
    return len(_load_feedback())


def _classify_mediapipe_kps(kps: list) -> tuple:
    """
    Classify MediaPipe keypoints using the trained classifier.
    Returns (is_smile: bool, confidence: float 0-1).
    """
    vec = _extract_features_from_mediapipe_kps(kps).reshape(1, -1)
    vec_scaled = _SCALER.transform(vec)
    pred  = _CLASSIFIER.predict(vec_scaled)[0]
    proba = _CLASSIFIER.predict_proba(vec_scaled)[0]
    return bool(pred), float(proba[pred])


# ── Public API ────────────────────────────────────────────────────────────────
def extract_keypoints_from_image(image_bytes: bytes) -> dict:
    """
    Run MediaPipe FaceLandmarker on raw image bytes.
    Returns dict with keypoints list, inter_eye_px, and error.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"keypoints": None, "inter_eye_px": 0.0,
                "error": "Could not decode image."}

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = _LANDMARKER.detect(mp_image)

    if not result.face_landmarks:
        return {"keypoints": None, "inter_eye_px": 0.0,
                "error": "No face detected in the image."}

    lm = result.face_landmarks[0]
    keypoints = []
    for idx in _ALL_SLOTS:
        keypoints.append([round(lm[idx].x * w, 2),
                          round(lm[idx].y * h, 2)])

    # Inter-eye distance: lm[33] (index 0) to lm[263] (index 5)
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


def detect_smile(image_bytes: bytes) -> dict:
    """
    Full pipeline: extract keypoints -> classify -> return result dict.
    """
    extraction = extract_keypoints_from_image(image_bytes)

    if extraction["error"]:
        return {
            "smile": None,
            "confidence": 0.0,
            "low_confidence": False,
            "inter_eye_px": extraction["inter_eye_px"],
            "keypoints": {},
            "features": {},
            "error": extraction["error"],
            "warning": None,
        }

    kps = extraction["keypoints"]
    is_smile, confidence = _classify_mediapipe_kps(kps)

    # Human-readable features for the UI
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
        "inter_eye_px": extraction["inter_eye_px"],
        "keypoints": named_kps,
        "features": features,
        "error": None,
        "warning": warning,
    }
