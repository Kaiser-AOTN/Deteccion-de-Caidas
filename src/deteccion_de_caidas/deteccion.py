import cv2
from collections import deque
from ultralytics import YOLO
import numpy as np

class Fall_Detection:
    def __init__(self, model_path='yolov8n-pose.pt', fall_aspect_ratio=1.0, fall_angle_threshold=45, confirm_frames=10, history_len=15):
        # Parámetros de sensibilidad para la detección geométrica de caídas
        self.model_path = model_path
        self.fall_aspect_ratio = fall_aspect_ratio
        self.fall_angle_threshold = fall_angle_threshold
        
        # Parámetros de confirmación temporal para evitar falsos positivos por movimientos rápidos
        self.confirm_frames = confirm_frames
        self.history_len = history_len 
        
        # Índices de los puntos clave (keypoints) del esqueleto según el formato estándar COCO
        self.L_SHOULDER, self.R_SHOULDER = 5, 6
        self.L_HIP, self.R_HIP = 11, 12

        # Inicialización del modelo YOLO y del diccionario de historial por ID de rastreo
        self.model = YOLO(self.model_path)
        self.track_history = {} 

    def Generic_angle(self, shoulder_mid, hip_mid):
        # Calcula el ángulo de inclinación del torso respecto al eje vertical
        dx, dy = np.abs(hip_mid - shoulder_mid)
        return float(np.degrees(np.arctan2(dx, dy + 1e-6)))

    def run_detection(self, frame):
        # Ejecutar el modelo con .track() y persist=True para mantener IDs estables entre frames.
        # Puedes añadir imgsz=320 aquí si deseas forzar mayor velocidad de procesamiento en la gráfica.
        results = self.model.track(frame, persist=True, verbose=False)
        any_fall_detected = False

        for r in results:
            # Descartar frames donde no se detecten cajas delimitadoras o puntos clave
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
                continue

            # Extraer las matrices de datos del tensor al CPU/NumPy
            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            keypoints = r.keypoints.xy.cpu().numpy()
            kp_confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
            
            # Asignar los IDs de seguimiento si el tracker ya los generó; si no, usar un comodín temporal (-1)
            if r.boxes.id is not None:
                track_ids = r.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [-1] * len(boxes)

            # Lista temporal para registrar los IDs que siguen activos en el frame actual
            active_ids = [] 

            for idx, (box, cls, conf, kpts, track_id) in enumerate(zip(boxes, clss, confs, keypoints, track_ids)):
                # Filtrar detecciones con baja confianza o que no correspondan a la clase persona (clase 0)
                if int(cls) != 0 or conf < 0.50:
                    continue

                # Validar que los puntos clave esenciales (hombros y caderas) tengan un nivel de visibilidad aceptable
                if kp_confs is not None and len(kp_confs) > idx:
                    if (kp_confs[idx][self.L_SHOULDER] < 0.4 or kp_confs[idx][self.R_SHOULDER] < 0.4 or
                        kp_confs[idx][self.L_HIP] < 0.4 or kp_confs[idx][self.R_HIP] < 0.4):
                        continue
                
                # Calcular las proporciones geométricas de la caja delimitadora (ancho frente a alto)
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                aspect_ratio = float(w / (h + 1e-6))

                # Calcular los puntos medios corporales y el ángulo de inclinación del torso
                shoulder_mid = (kpts[self.L_SHOULDER] + kpts[self.R_SHOULDER]) / 2.0
                hip_mid = (kpts[self.L_HIP] + kpts[self.R_HIP]) / 2.0
                angle = self.Generic_angle(shoulder_mid, hip_mid)

                person_id = int(track_id)
                person_fell = False

                # Evaluar el historial de la persona únicamente si el tracker ha emitido un ID válido
                if person_id != -1:
                    active_ids.append(person_id)
                    
                    # Inicializar la cola de historial individual si es una nueva persona detectada
                    if person_id not in self.track_history:
                        self.track_history[person_id] = deque(maxlen=self.history_len)

                    # Evaluar si la postura actual encaja con los parámetros de una posible caída
                    is_suspect = aspect_ratio > self.fall_aspect_ratio or angle > self.fall_angle_threshold
                    self.track_history[person_id].append(is_suspect)

                    # Confirmar la caída de forma estable si se cumple el umbral mínimo dentro del historial reciente
                    if len(self.track_history[person_id]) == self.history_len and sum(self.track_history[person_id]) >= self.confirm_frames:
                        person_fell = True
                        any_fall_detected = True
                else:
                    # Evaluación instantánea de respaldo si el tracker aún está asignando el ID
                    person_fell = aspect_ratio > self.fall_aspect_ratio or angle > self.fall_angle_threshold
                    if person_fell: 
                        any_fall_detected = True

                # Definir el color visual del cuadro (rojo para alerta de caída, verde para estado normal)
                color = (0, 0, 255) if person_fell else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Dibujar las etiquetas informativas sobre la persona analizada
                id_text = str(person_id) if person_id != -1 else "Buscando..."
                cv2.putText(frame, f"ID:{id_text} | ang:{angle:.0f} | conf {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Purgar el diccionario eliminando el historial de las personas que han salido del encuadre de la cámara
            stale_ids = [pid for pid in self.track_history.keys() if pid not in active_ids]
            for pid in stale_ids:
                del self.track_history[pid]

        return frame, any_fall_detected