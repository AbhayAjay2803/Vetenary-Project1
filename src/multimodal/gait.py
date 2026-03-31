"""
Real video processing for gait analysis using MediaPipe pose estimation.
Provides fallback when MediaPipe is not available or misconfigured.
"""

import numpy as np
import cv2
from typing import Tuple

# Try to import MediaPipe
MEDIAPIPE_AVAILABLE = False
pose = None

try:
    import mediapipe as mp
    # Check if solutions attribute exists (old versions may not have it)
    if hasattr(mp, 'solutions'):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        MEDIAPIPE_AVAILABLE = True
        print("✅ MediaPipe loaded successfully.")
    else:
        print("⚠️ MediaPipe installed but 'solutions' module not found. Reinstall with: pip install --upgrade mediapipe")
except ImportError:
    print("⚠️ MediaPipe not installed. Gait analysis will use fallback.")
except Exception as e:
    print(f"⚠️ MediaPipe error: {e}. Gait analysis will use fallback.")

def extract_gait_feature_vector(video_path: str) -> np.ndarray:
    """
    Extract pose landmarks from video (66-dim).
    Returns array of shape (66,) with average x,y for 33 landmarks.
    """
    if not MEDIAPIPE_AVAILABLE or pose is None:
        return np.zeros(66, dtype=np.float32)
    try:
        cap = cv2.VideoCapture(video_path)
        landmarks_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                frame_landmarks = []
                for l in lm:
                    frame_landmarks.append(l.x)
                    frame_landmarks.append(l.y)
                landmarks_list.append(frame_landmarks)
        cap.release()
        if not landmarks_list:
            return np.zeros(66, dtype=np.float32)
        avg_landmarks = np.mean(landmarks_list, axis=0)
        return avg_landmarks.astype(np.float32)
    except Exception as e:
        print(f"Error processing video: {e}")
        return np.zeros(66, dtype=np.float32)

def extract_gait_features(video_path: str) -> Tuple[float, str]:
    """
    Return heuristic lameness probability and label.
    Used as fallback if models are not available.
    """
    landmarks = extract_gait_feature_vector(video_path)
    if len(landmarks) < 56:
        return 0.5, "unknown"
    left_hip_y = landmarks[23*2 + 1]
    right_hip_y = landmarks[24*2 + 1]
    left_knee_y = landmarks[25*2 + 1]
    right_knee_y = landmarks[26*2 + 1]
    asymmetry = abs(left_hip_y - right_hip_y) + abs(left_knee_y - right_knee_y)
    lameness = np.clip(asymmetry * 2, 0, 1)
    label = "lameness" if lameness > 0.5 else "normal"
    return lameness, label