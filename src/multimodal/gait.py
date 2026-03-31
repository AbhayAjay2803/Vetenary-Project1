"""
Real video processing for gait analysis using MediaPipe pose estimation.
"""

import numpy as np
import cv2
from typing import Tuple

# Try to import MediaPipe (optional dependency)
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not installed. Gait analysis will use fallback.")

def extract_gait_features(video_path: str) -> Tuple[float, str]:
    """
    Extract pose landmarks from video and estimate lameness probability.
    Returns (lameness_probability, label) where label is 'lameness' or 'normal'.
    """
    if not MEDIAPIPE_AVAILABLE:
        return 0.5, "unknown"
    
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
                # Extract x,y for all 33 landmarks
                lm = results.pose_landmarks.landmark
                frame_landmarks = []
                for l in lm:
                    frame_landmarks.append(l.x)
                    frame_landmarks.append(l.y)
                landmarks_list.append(frame_landmarks)
        cap.release()
        
        if not landmarks_list:
            return 0.5, "unknown"
        
        # Average landmarks over frames
        avg_landmarks = np.mean(landmarks_list, axis=0)
        
        # Simple lameness detection based on hip/knee asymmetry
        left_hip_y = avg_landmarks[23*2 + 1]
        right_hip_y = avg_landmarks[24*2 + 1]
        left_knee_y = avg_landmarks[25*2 + 1]
        right_knee_y = avg_landmarks[26*2 + 1]
        
        asymmetry = abs(left_hip_y - right_hip_y) + abs(left_knee_y - right_knee_y)
        lameness = np.clip(asymmetry * 2, 0, 1)
        label = "lameness" if lameness > 0.5 else "normal"
        return lameness, label
    except Exception as e:
        print(f"Error processing video: {e}")
        return 0.5, "unknown"