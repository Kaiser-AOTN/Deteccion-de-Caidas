import sys
from pathlib import Path
import cv2
import time

# Configurar rutas para importar los archivos sin problemas de carpetas
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    try:
        from deteccion_de_caidas.deteccion import Fall_Detection
        from deteccion_de_caidas.signal_for_help import SignalForHelpDetector
    except ModuleNotFoundError:
        from .deteccion import Fall_Detection
        from .signal_for_help import SignalForHelpDetector
else:
    from .deteccion import Fall_Detection
    from .signal_for_help import SignalForHelpDetector


def main():
    print("Iniciando Sistema Integrado de Vigilancia AI...")

    # Cargar los modelos entrenados
    detector_caidas = Fall_Detection()
    detector_gestos = SignalForHelpDetector()

    # Iniciar la cámara web principal
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        raise RuntimeError("No se pudo conectar con la cámara web.")

    prev_time = 0

    # Bucle principal de lectura de la cámara
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo el fotograma de la cámara.")
            break

        # Calcular los fotogramas por segundo (FPS)
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # 1. Procesar el detector de señales de auxilio
        alerta_auxilio = detector_gestos.detect_signal_for_help(frame)
        # 2. Procesar el detector de caídas de personas
        frame, alerta_caida = detector_caidas.run_detection(frame)

        # Dibujar contador de FPS en la parte superior
        cv2.putText(frame, f"FPS: {int(fps)}", (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Mostrar banners de alerta globales si se detecta algún peligro
        if alerta_caida:
            cv2.putText(frame, "¡ALERTA: CAIDA DETECTADA!", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        if alerta_auxilio:
            cv2.putText(frame, "¡ALERTA: SENAL DE AYUDA!", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)

        # Mostrar la ventana con el video resultante
        cv2.imshow("Sistema Integrado - AI Monitor", frame)

        # Salir del programa presionando la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Liberar recursos y cerrar la ventana al finalizar
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()