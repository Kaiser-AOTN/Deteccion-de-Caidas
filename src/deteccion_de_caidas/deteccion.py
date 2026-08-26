import cv2

from collections import deque
from ultralytics import YOLO
import numpy as np

#Configuración 
MODEL_PATH = 'yolov8n-pose.pt'  
FALL_ASPECT_RATIO = 1.5
FALL_ANGLE_THRESHOLD = 45
CONFIRM_FRAMES = 10
HISTORY_LEN = 15

# Índices COCO keypoints
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

model = YOLO(MODEL_PATH)
history = deque(maxlen=HISTORY_LEN)
cap = cv2.VideoCapture(0)

# Función para calcular el ángulo del torso
def torso_angle(shoulder_mid, hip_mid):
    dx, dy = np.abs(hip_mid - shoulder_mid)
    return float(np.degrees(np.arctan2(dx, dy + 1e-6)))


while cap.isOpened():
    
    #Capturar frames
    ret, frame = cap.read()
    #Si no se puede capturar el frame, salir del bucle
    if not ret:
        break

    #Realizar detección de poses
    results = model(frame, verbose=False)
    fall_detected = False

    for r in results:
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            continue

        #Obtener las cajas delimitadoras y los keypoints de las personas detectadas
        boxes = r.boxes.xyxy.cpu().numpy()
        keypoints = r.keypoints.xy.cpu().numpy()

        #Iterar sobre cada persona detectada y sus keypoints
        for box, kpts in zip(boxes, keypoints):

            # Calcular el aspecto y el ángulo del torso
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            aspect_ratio = float(w / (h + 1e-6))
            
            # Calcular el punto medio de los hombros
            shoulder_mid = (kpts[L_SHOULDER] + kpts[R_SHOULDER]) / 2.0
            # Calcular el punto medio de las caderas
            hip_mid = (kpts[L_HIP] + kpts[R_HIP]) / 2.0

            # Calcular el angulo del torso y determinar si es sospechoso de caída
            angle = torso_angle(shoulder_mid, hip_mid)
            is_suspect = aspect_ratio > FALL_ASPECT_RATIO and angle > FALL_ANGLE_THRESHOLD
            history.append(is_suspect)

            # Si se han detectado suficientes frames sospechosos, marcar como caída
            if len(history) == HISTORY_LEN and sum(history) >= CONFIRM_FRAMES:
                fall_detected = True

            # Dibujar la caja delimitadora y el texto en el frame
            color = (0, 0, 255) if fall_detected else (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ang:{angle:.0f} ar:{aspect_ratio:.2f}",
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
    # Mostrar alerta de caída si se detecta
    if fall_detected:
        cv2.putText(frame, "CAIDA DETECTADA", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
        # TODO: Implementar notificacion o llamada para aviso
        
    # Mostrar el frame con las detecciones
    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()