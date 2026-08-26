import cv2
import math
from collections import deque
from ultralytics import YOLO

#Configuración 
MODEL_PATH = 'yolov8n-pose.pt'  
FALL_ASPECT_RATIO = 1.3
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
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


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

        #Iterar sobre cada persona detectada y sus keypoints
        for box, kpts in zip(r.boxes, r.keypoints.xy):

            # Calcular el aspecto y el ángulo del torso
            x1, y1, x2, y2 = box.xyxy[0]
            w, h = x2 - x1, y2 - y1
            aspect_ratio = w / (h + 1e-6)
            
            # Calcular el punto medio de los hombros
            shoulder_mid = ((kpts[L_SHOULDER][0] + kpts[R_SHOULDER][0]) / 2,
                            (kpts[L_SHOULDER][1] + kpts[R_SHOULDER][1]) / 2)
            # Calcular el punto medio de las caderas
            hip_mid = ((kpts[L_HIP][0] + kpts[R_HIP][0]) / 2,
                    (kpts[L_HIP][1] + kpts[R_HIP][1]) / 2)

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