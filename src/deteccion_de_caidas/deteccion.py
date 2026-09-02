import cv2
import numpy as np
from collections import deque
from .base_detector import BaseDetector

# Hereda de BaseDetector para reutilizar la carga de YOLO
class Fall_Detection(BaseDetector):
    def __init__(self, 
    model_path='yolov8n-pose.pt',
    fall_aspect_ratio=1.0, # Proporción ancho/alto de la caja
    fall_angle_threshold=50, # Ángulo de inclinación del torso3
    confirm_frames=10, # Cuántos fotogramas seguidos debe durar la postura para confirmar caída
    history_len=15): # Tamaño de la memoria temporal para cada persona


        # Inicializa la clase padre pasando el modelo de cuerpo humano
        super().__init__(model_path)
        
        # Umbrales para decidir cuándo una postura cuenta como caída
        self.fall_aspect_ratio = fall_aspect_ratio # Proporción ancho/alto de la caja
        self.fall_angle_threshold = fall_angle_threshold # Ángulo de inclinación del torso
        self.confirm_frames = confirm_frames # Cuántos fotogramas seguidos debe durar la postura
        self.history_len = history_len # Tamaño de la memoria temporal
        
        # Índices de las articulaciones del cuerpo humano en YOLO Pose
        self.L_SHOULDER, self.R_SHOULDER = 5, 6
        self.L_HIP, self.R_HIP = 11, 12
        
        # Historial de memoria para guardar la postura de cada persona
        self.track_history = {}

    def Generic_angle(self, shoulder_mid, hip_mid):
        """Calcula el ángulo de inclinación entre los hombros y las caderas"""
        dx, dy = np.abs(hip_mid - shoulder_mid)
        return float(np.degrees(np.arctan2(dx, dy + 1e-6)))

    def run_detection(self, frame):
        any_fall_detected = False
        active_ids = []

        # Recibe los datos procesados desde el BaseDetector
        for boxes, clss, confs, keypoints, kp_confs, track_ids in self.process_tracking(frame):
            for idx, (box, cls, conf, track_id) in enumerate(zip(boxes, clss, confs, track_ids)):
                # Filtrar si no es una persona (clase 0) o si la confianza es baja
                if int(cls) != 0 or conf < 0.50:
                    continue
                
                # Verificar que el modelo nos entregó puntos clave del esqueleto
                if keypoints is None or kp_confs is None or len(kp_confs) <= idx:
                    continue
                
                kpts = keypoints[idx]
                kpcs = kp_confs[idx]
                
                # Validar que los hombros y caderas sean visibles
                if (kpcs[self.L_SHOULDER] < 0.4 or kpcs[self.R_SHOULDER] < 0.4 or
                    kpcs[self.L_HIP] < 0.4 or kpcs[self.R_HIP] < 0.4):
                    continue

                # Calcular la relación de aspecto (ancho / alto) de la persona
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                aspect_ratio = float(w / (h + 1e-6))

                # Calcular el centro de los hombros y caderas para medir el ángulo
                shoulder_mid = (kpts[self.L_SHOULDER] + kpts[self.R_SHOULDER]) / 2.0
                hip_mid = (kpts[self.L_HIP] + kpts[self.R_HIP]) / 2.0
                angle = self.Generic_angle(shoulder_mid, hip_mid)

                person_id = int(track_id)
                person_fell = False

                # Si la persona tiene un ID asignado, analizamos su memoria
                if person_id != -1:
                    active_ids.append(person_id)
                    if person_id not in self.track_history:
                        self.track_history[person_id] = deque(maxlen=self.history_len)

                    # Determinar si la postura en este instante parece una caída
                    is_suspect = aspect_ratio > self.fall_aspect_ratio or angle > self.fall_angle_threshold
                    self.track_history[person_id].append(is_suspect)

                    # Si ha mantenido la postura de caída por suficiente tiempo, confirmar alerta
                    if len(self.track_history[person_id]) == self.history_len and sum(self.track_history[person_id]) >= self.confirm_frames:
                        person_fell = True
                        any_fall_detected = True
                else:
                    # Si no hay ID, evaluamos únicamente el estado del fotograma actual
                    person_fell = aspect_ratio > self.fall_aspect_ratio or angle > self.fall_angle_threshold
                    if person_fell: 
                        any_fall_detected = True

                # Dibujar recuadro verde (normal) o rojo (caída) en la imagen
                color = (0, 0, 255) if person_fell else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                id_text = str(person_id) if person_id != -1 else "Buscando..."
                cv2.putText(frame, f"ID:{id_text} | ang:{angle:.0f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Eliminar de la memoria las personas que salieron de la cámara
        stale_ids = [pid for pid in self.track_history.keys() if pid not in active_ids]
        for pid in stale_ids:
            del self.track_history[pid]

        return frame, any_fall_detected