
from ultralytics import YOLO

class SignalForHelpDetector:
    def __init__(self, model_path="yolov8n-pose.pt"):
        # Usamos el modelo de pose general para evaluar la posición de los brazos
        self.model_path = model_path
        try:
            self.model = YOLO(self.model_path)
            self.enabled = True
        except Exception as e:
            print(f"No se pudo cargar el modelo de gestos/postura: {e}")
            self.enabled = False

    def detect_signal_for_help(self, frame):
        if not self.enabled or frame is None:
            return False

        results = self.model(frame, verbose=False)
        signal_detected = False

        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            keypoints = r.keypoints.xy.cpu().numpy()
            kp_confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

            for idx, (box, cls, kpts) in enumerate(zip(boxes, clss, keypoints)):
                # Filtrar solo la clase persona (0)
                if int(cls) != 0:
                    continue

                # Validar que tengamos las confianzas de hombros (5, 6) y muñecas (9, 10)
                if kp_confs is not None and len(kp_confs[idx]) > 10:
                    l_shoulder_y = kpts[5][1]
                    r_shoulder_y = kpts[6][1]
                    l_wrist_y = kpts[9][1]
                    r_wrist_y = kpts[10][1]
                    
                    l_wrist_conf = kp_confs[idx][9]
                    r_wrist_conf = kp_confs[idx][10]

                    # Heurística de emergencia: Si ambas muñecas están significativamente
                    # por encima de los hombros (manos levantadas pidiendo auxilio)
                    if (l_wrist_conf > 0.4 and l_wrist_y < l_shoulder_y) and \
                        (r_wrist_conf > 0.4 and r_wrist_y < r_shoulder_y):
                        
                        x1, y1, x2, y2 = map(int, box)
                        
                        signal_detected = True

        return signal_detected