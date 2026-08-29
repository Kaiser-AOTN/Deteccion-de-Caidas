import cv2

from collections import deque
from ultralytics import YOLO
import numpy as np

#Configuración 
MODEL_PATH = 'yolov8n-pose.pt'  
FALL_ASPECT_RATIO = 1.2
FALL_ANGLE_THRESHOLD = 45
CONFIRM_FRAMES = 10
HISTORY_LEN = 15

# Índices COCO keypoints
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

model = YOLO(MODEL_PATH)
history = deque(maxlen=HISTORY_LEN)
cap = cv2.VideoCapture(0)

track_history = {}
# Función para calcular el ángulo del torso
def Generic_angle(shoulder_mid, hip_mid):
    dx, dy = np.abs(hip_mid - shoulder_mid)
    return float(np.degrees(np.arctan2(dx, dy + 1e-6)))


while cap.isOpened():
    
    #Capturar frames
    ret, frame = cap.read()
    #Si no se puede capturar el frame, salir del bucle
    if not ret:
        break

    #Realizar detección de poses
    results = model.track(frame,persist = True ,verbose=False)
    fall_detected = False
    any_fall_detected = False

    for r in results:
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None or r.boxes.id is None:
            continue

        # Obtener cajas, clases y confianzas de detección
        boxes = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        keypoints = r.keypoints.xy.cpu().numpy()
        kp_confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
        track_ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None

        for idx, (box, cls, conf, kpts , track_id) in enumerate(zip(boxes, clss, confs, keypoints, track_ids)):
            
            # --- FILTRO 1: Solo procesar si es una PERSONA (clase 0) con buena confianza (> 50%) ---
            if int(cls) != 0 or conf < 0.50:
                continue

            # --- FILTRO 2: Validar visibilidad de hombros y caderas ---
            if kp_confs is not None:
                # Si los puntos de hombros o caderas tienen muy poca confianza, descartar (es un objeto)
                if (kp_confs[idx][L_SHOULDER] < 0.4 or kp_confs[idx][R_SHOULDER] < 0.4 or
                    kp_confs[idx][L_HIP] < 0.4 or kp_confs[idx][R_HIP] < 0.4):
                    continue

            # Extraer coordenadas de la persona confirmada
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            aspect_ratio = float(w / (h + 1e-6))

            # Puntos medios de torso
            shoulder_mid = (kpts[L_SHOULDER] + kpts[R_SHOULDER]) / 2.0
            hip_mid = (kpts[L_HIP] + kpts[R_HIP]) / 2.0
            angle = Generic_angle(shoulder_mid, hip_mid)

            # Identificador de la persona (track_id) para seguimiento
            person_id = int(track_id)

            if person_id not in track_history:
                track_history[person_id] = deque(maxlen=HISTORY_LEN)

            # Evaluación de sospecha de caída
            is_suspect = aspect_ratio > FALL_ASPECT_RATIO and angle > FALL_ANGLE_THRESHOLD
            track_history[person_id].append(is_suspect)

            # Verificar si la persona ha caído
            person_fell = False
            if len(track_history[person_id]) == HISTORY_LEN and sum(track_history[person_id]) >= CONFIRM_FRAMES:
                person_fell = True
                any_fall_detected = True
                fall_detected = True

            # Dibujar la caja delimitadora
            
            color = (0, 0, 255) if fall_detected else (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Texto unificado: ID, Confianza y Ángulo en el mismo renglón
            texto = f"ID:{person_id} | Conf:{conf:.2f} | Ang:{angle:.0f}"
            cv2.putText(frame, texto, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar alerta de caída global si se detecta
    if any_fall_detected:
        cv2.putText(frame, "CAIDA DETECTADA", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        # TODO: Implementar notificacion o llamada para aviso
        
    # Mostrar el frame con las detecciones
    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()