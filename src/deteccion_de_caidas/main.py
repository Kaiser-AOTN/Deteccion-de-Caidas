import sys
from pathlib import Path
import cv2
import time

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from deteccion_de_caidas.deteccion import Fall_Detection
else:
    from .deteccion import Fall_Detection


def main():
    print("Iniciando Sistema de Detección de Caídas...")

    # Inicializar únicamente el detector de caídas con el modelo de pose
    detector_caidas = Fall_Detection(model_path="yolov8n-pose.pt")

    # Abrir la cámara web principal
    cap = cv2.VideoCapture(0)
    
    # Forzar una resolución estándar para asegurar una tasa alta de FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Verifica que esté conectada.")

    # Variables para el cálculo dinámico de los fotogramas por segundo (FPS)
    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- CÁLCULO Y VISUALIZACIÓN DE FPS ---
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (520, 30),  # Coordenadas en la esquina superior derecha
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        # -------------------------------------

        # Ejecutar la detección de caídas sobre el frame actual
        frame, alerta_caida = detector_caidas.run_detection(frame)
        
        # Mostrar alerta visual en pantalla si se detecta una caída
        if alerta_caida:
            cv2.putText(
                frame,
                "¡CAIDA DETECTADA!",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

        # Mostrar el resultado final en la ventana de OpenCV
        cv2.imshow("Sistema de Deteccion de Caidas", frame)

        # Salir del bucle al presionar la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Liberar los recursos de la cámara y cerrar ventanas abiertas
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()