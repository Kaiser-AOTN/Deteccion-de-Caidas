import cv2

from collections import deque
from ultralytics import YOLO
import numpy as np

class Fall_Detection:
    def __init__(self, model_path='yolov8n-pose.pt', fall_aspect_ratio=1.2, fall_angle_threshold=45, confirm_frames=10, history_len=15):
        self.model_path = model_path
        self.fall_aspect_ratio = fall_aspect_ratio
        self.fall_angle_threshold = fall_angle_threshold
        self.confirm_frames = confirm_frames
        self.history_len = history_len

        # Cargar el modelo YOLOv8 para detección de poses
        self.model = YOLO(self.model_path)
        self.history = deque(maxlen=self.history_len)
        self.cap = cv2.VideoCapture(0)
        self.track_history = {}

    def Generic_angle(self, shoulder_mid, hip_mid):
        dx, dy = np.abs(hip_mid - shoulder_mid)
        return float(np.degrees(np.arctan2(dx, dy + 1e-6)))

    def run_detection(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model.track(frame, persist=True, verbose=False)
            any_fall_detected = False

            for r in results:
                if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None or r.boxes.id is None:
                    continue

                boxes = r.boxes.xyxy.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                keypoints = r.keypoints.xy.cpu().numpy()
                kp_confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
                track_ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None

                for idx, (box, cls, conf, kpts , track_id) in enumerate(zip(boxes, clss, confs, keypoints, track_ids)):
                    if int(cls) != 0 or conf < 0.50:
                        continue

                    if kp_confs is not None:
                        if (kp_confs[idx][L_SHOULDER] < 0.4 or kp_confs[idx][R_SHOULDER] < 0.4 or
                            kp_confs[idx][L_HIP] < 0.4 or kp_confs[idx][R_HIP] < 0.4):
                            continue

