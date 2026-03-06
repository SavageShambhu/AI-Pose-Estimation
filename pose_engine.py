# pose_engine.py — MediaPipe Pose detection engine

import cv2
import mediapipe as mp
from config import MIN_VISIBILITY

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Custom drawing spec for cleaner skeleton
LANDMARK_STYLE = mp_drawing.DrawingSpec(
    color=(0, 255, 180), thickness=3, circle_radius=4
)
CONNECTION_STYLE = mp_drawing.DrawingSpec(
    color=(255, 200, 0), thickness=2
)


def create_pose_detector():
    """
    Initialise MediaPipe Pose.
    model_complexity=1: balanced speed/accuracy for real-time webcam.
    smooth_landmarks=True: reduces jitter between frames.
    """
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )


def process_frame(pose, frame):
    """
    Run MediaPipe pose detection on a BGR frame.
    Returns (results, annotated_frame).
    """
    h, w = frame.shape[:2]

    # BGR → RGB (MediaPipe requires RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)
    rgb.flags.writeable = True

    # Draw skeleton overlay
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=LANDMARK_STYLE,
            connection_drawing_spec=CONNECTION_STYLE
        )

    return results


def get_landmarks(results):
    """Extract landmark list or return None."""
    if results and results.pose_landmarks:
        return results.pose_landmarks.landmark
    return None


def get_world_landmarks(results):
    """Extract world landmark list (metric 3D coords) or return None."""
    if results and results.pose_world_landmarks:
        return results.pose_world_landmarks.landmark
    return None


def open_webcam(cam_index=0, width=1280, height=720):
    """Open webcam with given resolution."""
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera permissions in System Settings → Privacy → Camera.")
    return cap
