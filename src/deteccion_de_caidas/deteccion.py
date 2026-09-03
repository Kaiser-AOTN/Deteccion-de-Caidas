import cv2
import numpy as np
from collections import deque
from .base_detector import BaseDetector

# Hereda de BaseDetector para reutilizar la carga de YOLO
class Fall_Detection(BaseDetector):
    def __init__(self,
                 model_path='yolov8n-pose.pt',
                 fall_aspect_ratio=1.3,         # más estricto que antes
                 fall_angle_threshold=65,
                 fall_velocity_threshold=0.6,   # % de altura perdida en la ventana
                 height_drop_ratio=0.55,        # % de altura perdida vs máx. histórico
                 confirm_frames=10,
                 history_len=15,
                 velocity_window=5,
                 floor_persist_frames=45):  # ~1.5s a 30fps: tiempo quieto en el piso para confirmar

        super().__init__(model_path)

        self.fall_aspect_ratio = fall_aspect_ratio
        self.fall_angle_threshold = fall_angle_threshold
        self.fall_velocity_threshold = fall_velocity_threshold
        self.height_drop_ratio = height_drop_ratio
        self.confirm_frames = confirm_frames
        self.history_len = history_len
        self.velocity_window = velocity_window
        self.floor_persist_frames = floor_persist_frames

        self.L_SHOULDER, self.R_SHOULDER = 5, 6
        self.L_HIP, self.R_HIP = 11, 12
        self.L_KNEE, self.R_KNEE = 13, 14

        # Memoria por persona
        self.track_history = {}     # posturas sospechosas recientes (bool) - detecta el evento de caer
        self.height_history = {}    # alturas de bbox recientes (para velocidad)
        self.max_height = {}        # altura máxima observada de pie (referencia)
        self.floor_streak = {}      # frames CONSECUTIVOS en postura de piso - detecta permanecer caído

    def Generic_angle(self, shoulder_mid, hip_mid):
        """Ángulo de inclinación entre hombros y caderas"""
        dx, dy = np.abs(hip_mid - shoulder_mid)
        return float(np.degrees(np.arctan2(dx, dy + 1e-6)))

    def _fall_velocity(self, person_id, current_h):
        """Qué tan rápido bajó la altura del bbox dentro de la ventana reciente."""
        hist = self.height_history[person_id]
        hist.append(current_h)
        if len(hist) < 2:
            return 0.0
        oldest = hist[0]
        return max((oldest - current_h) / (oldest + 1e-6), 0.0)

    def run_detection(self, frame):
        any_fall_detected = False
        active_ids = []

        for boxes, clss, confs, keypoints, kp_confs, track_ids in self.process_tracking(frame):
            for idx, (box, cls, conf, track_id) in enumerate(zip(boxes, clss, confs, track_ids)):
                if int(cls) != 0 or conf < 0.50:
                    continue
                if keypoints is None or kp_confs is None or len(kp_confs) <= idx:
                    continue

                kpts = keypoints[idx]
                kpcs = kp_confs[idx]

                if (kpcs[self.L_SHOULDER] < 0.4 or kpcs[self.R_SHOULDER] < 0.4 or
                        kpcs[self.L_HIP] < 0.4 or kpcs[self.R_HIP] < 0.4):
                    continue

                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                aspect_ratio = float(w / (h + 1e-6))

                shoulder_mid = (kpts[self.L_SHOULDER] + kpts[self.R_SHOULDER]) / 2.0
                hip_mid = (kpts[self.L_HIP] + kpts[self.R_HIP]) / 2.0
                angle = self.Generic_angle(shoulder_mid, hip_mid)

                # --- Filtro anti-agachado: torso erguido pese a caja baja = agacharse ---
                knees_visible = kpcs[self.L_KNEE] > 0.4 and kpcs[self.R_KNEE] > 0.4
                torso_upright_despite_low_box = knees_visible and angle < (self.fall_angle_threshold * 0.5)

                person_id = int(track_id)
                person_fell = False

                if person_id != -1:
                    active_ids.append(person_id)
                    if person_id not in self.track_history:
                        self.track_history[person_id] = deque(maxlen=self.history_len)
                        self.height_history[person_id] = deque(maxlen=self.velocity_window)
                        self.max_height[person_id] = h
                        self.floor_streak[person_id] = 0

                    # Actualiza altura de referencia solo si está claramente de pie
                    if angle < 30 and h > self.max_height[person_id]:
                        self.max_height[person_id] = h

                    velocity = self._fall_velocity(person_id, h)
                    height_ratio = h / (self.max_height[person_id] + 1e-6)

                    posture_suspect = aspect_ratio > self.fall_aspect_ratio or angle > self.fall_angle_threshold
                    if torso_upright_despite_low_box:
                        posture_suspect = False  # descarta agachado

                    # Debe cumplir postura Y (caída rápida O quedó muy bajo vs su altura normal)
                    dynamic_suspect = (velocity > self.fall_velocity_threshold or
                                        height_ratio < (1 - self.height_drop_ratio))

                    is_suspect = posture_suspect and dynamic_suspect
                    self.track_history[person_id].append(is_suspect)

                    # Camino 1: detecta el EVENTO de caer (postura + movimiento brusco)
                    fast_fall = (len(self.track_history[person_id]) == self.history_len and
                                 sum(self.track_history[person_id]) >= self.confirm_frames)

                    # Camino 2: detecta que la persona SIGUE en el piso, sin depender de velocidad.
                    # Cubre: personas que ya estaban caídas al empezar el tracking, o que dejaron
                    # de moverse tras la caída (velocity cae a 0 y el camino 1 se apaga).
                    if posture_suspect:
                        self.floor_streak[person_id] += 1
                    else:
                        self.floor_streak[person_id] = 0
                    on_floor = self.floor_streak[person_id] >= self.floor_persist_frames

                    if fast_fall or on_floor:
                        person_fell = True
                        any_fall_detected = True
                else:
                    posture_suspect = aspect_ratio > self.fall_aspect_ratio or angle > self.fall_angle_threshold
                    person_fell = posture_suspect and not torso_upright_despite_low_box
                    if person_fell:
                        any_fall_detected = True

                color = (0, 0, 255) if person_fell else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                id_text = str(person_id) if person_id != -1 else "Buscando..."
                cv2.putText(frame, f"ID:{id_text} | ang:{angle:.0f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        stale_ids = [pid for pid in self.track_history.keys() if pid not in active_ids]
        for pid in stale_ids:
            del self.track_history[pid]
            del self.height_history[pid]
            del self.max_height[pid]
            del self.floor_streak[pid]

        return frame, any_fall_detected