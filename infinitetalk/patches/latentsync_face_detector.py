import os

from insightface.app import FaceAnalysis
import numpy as np
import onnxruntime
import torch


INSIGHTFACE_DETECT_SIZE = 512
INSIGHTFACE_ROOT = os.environ.get(
    "LATENTSYNC_INSIGHTFACE_ROOT",
    "/opt/ComfyUI/models/latentsync/auxiliary",
)

onnxruntime.preload_dlls()


class FaceDetector:
    def __init__(self, device="cuda"):
        self.app = FaceAnalysis(
            allowed_modules=[
                "detection",
                "landmark_2d_106",
                "landmark_3d_68",
            ],
            root=INSIGHTFACE_ROOT,
            providers=["CUDAExecutionProvider"],
        )
        self.app.prepare(
            ctx_id=cuda_to_int(device),
            det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE),
        )
        self.pose_history = []

    def __call__(self, frame, threshold=0.5):
        frame_height, frame_width, _channels = frame.shape
        faces = self.app.get(frame)

        selected_face = None
        largest_size = 0
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(np.int_).tolist()
            width, height = x2 - x1, y2 - y1
            if width < 50 or height < 80:
                continue
            if width / height > 1.5 or width / height < 0.2:
                continue
            if face.det_score < threshold:
                continue
            current_size = width * height
            if current_size > largest_size:
                largest_size = current_size
                selected_face = face

        if selected_face is None:
            return None, None

        face = selected_face
        pose = getattr(face, "pose", None)
        yaw = float(pose[1]) if pose is not None and len(pose) >= 2 else 0.0
        self.pose_history.append(yaw)
        landmarks = np.round(face.landmark_2d_106).astype(np.int_)
        half_face_coord = np.mean([landmarks[74], landmarks[73]], axis=0)
        sub_landmarks = landmarks[LMK_ADAPT_ORIGIN_ORDER]
        half_face_distance = np.max(sub_landmarks[:, 1]) - half_face_coord[1]
        upper_bound = half_face_coord[1] - half_face_distance

        x1 = np.min(sub_landmarks[:, 0])
        y1 = int(upper_bound)
        x2 = np.max(sub_landmarks[:, 0])
        y2 = np.max(sub_landmarks[:, 1])
        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            x1, y1, x2, y2 = face.bbox.astype(np.int_).tolist()

        y2 += int((x2 - x1) * 0.1)
        x1 -= int((x2 - x1) * 0.05)
        x2 += int((x2 - x1) * 0.05)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width, x2)
        y2 = min(frame_height, y2)
        return (x1, y1, x2, y2), landmarks


def cuda_to_int(cuda_str: str) -> int:
    if cuda_str == "cuda":
        return 0
    device = torch.device(cuda_str)
    if device.type != "cuda":
        raise ValueError(f"Device type must be 'cuda', got: {device.type}")
    return device.index


LMK_ADAPT_ORIGIN_ORDER = [
    1,
    10,
    12,
    14,
    16,
    3,
    5,
    7,
    0,
    23,
    21,
    19,
    32,
    30,
    28,
    26,
    17,
    43,
    48,
    49,
    51,
    50,
    102,
    103,
    104,
    105,
    101,
    73,
    74,
    86,
]
