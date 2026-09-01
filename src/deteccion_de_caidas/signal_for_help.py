
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class SignalForHelpDetector:
    def __init__(self,
                model_path="hand_landmarker.task",
                max_num_hands=8,
                min_detection_confidence=0.6,
                min_presence_confidence=0.6,
                min_tracking_confidence=0.6):

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

        # --- Índices de landmarks relevantes (esquema estándar de 21 puntos) ---
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_MCP = 5
        self.INDEX_TIP = 8
        self.PINKY_MCP = 17

        # --- Estado de la máquina de gestos por mano ("Left" / "Right") ---
        self.hand_states = {}

        self.timeout = 5.0
        self.SMOOTH_ALPHA = 0.6

        self.RATIO_ABIERTO = 1.15
        self.RATIO_CERRADO = 1.05
        self.UMBRAL_PULGAR = 0.8

        self.REPETICIONES_REQUERIDAS = 3

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #

    def _distancia(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _suavizar(self, estado, clave, nuevo_valor):
        anterior = estado.get(clave)
        if anterior is None:
            estado[clave] = nuevo_valor
        else:
            a = self.SMOOTH_ALPHA
            estado[clave] = a * nuevo_valor + (1 - a) * anterior
        return estado[clave]

    def _dibujar_esqueleto(self, frame, landmarks_px):
        for idx, (x, y) in enumerate(landmarks_px):
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(idx), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # ------------------------------------------------------------------ #
    # Detección principal
    # ------------------------------------------------------------------ #
    
    def detect_signal_for_help(self, frame):
        """Procesa un frame BGR de OpenCV, dibuja el estado sobre él y
        devuelve True si se confirmó la señal de auxilio en este frame."""
        
        alto, ancho, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.time() - self._start_time) * 1000)
        resultado = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        señal_detectada = False
        current_time = time.time()
        manos_activas = set()

        for hand_landmarks, handedness in zip(resultado.hand_landmarks, resultado.handedness):
            etiqueta = handedness[0].category_name  # "Left" o "Right"
            confianza = handedness[0].score
            manos_activas.add(etiqueta)

            puntos = [(lm.x * ancho, lm.y * alto) for lm in hand_landmarks]
            self._dibujar_esqueleto(frame, puntos)

            if etiqueta not in self.hand_states:
                self.hand_states[etiqueta] = {'phase': 0, 'timestamp': current_time, 'count': 0}
            estado = self.hand_states[etiqueta]

            # --- Cálculos geométricos ---
            escala_cruda = self._distancia(puntos[self.INDEX_MCP], puntos[self.WRIST])
            escala_palma = self._suavizar(estado, 'palm_scale', escala_cruda)
            if escala_palma < 1.0:
                continue

            dist_indice_cruda = self._distancia(puntos[self.INDEX_TIP], puntos[self.WRIST])
            dist_indice = self._suavizar(estado, 'index_wrist', dist_indice_cruda)

            dedos_abiertos = dist_indice > (escala_palma * self.RATIO_ABIERTO)
            dedos_cerrados = dist_indice < (escala_palma * self.RATIO_CERRADO)

            dist_pulgar_cruda = self._distancia(puntos[self.THUMB_TIP], puntos[self.PINKY_MCP])
            dist_pulgar = self._suavizar(estado, 'thumb_pinky', dist_pulgar_cruda)
            proporcion_pulgar = dist_pulgar / escala_palma
            pulgar_escondido = proporcion_pulgar < self.UMBRAL_PULGAR

            # --- Máquina de estados del gesto ---
            if current_time - estado['timestamp'] > self.timeout:
                estado['phase'] = 0
                estado['count'] = 0

            if estado['phase'] in (0, 3) and dedos_abiertos and not pulgar_escondido:
                estado['phase'] = 1
                estado['timestamp'] = current_time
            elif estado['phase'] == 1 and dedos_abiertos and pulgar_escondido:
                estado['phase'] = 2
                estado['timestamp'] = current_time
            elif estado['phase'] == 2 and dedos_cerrados:
                estado['phase'] = 3
                estado['count'] += 1
                estado['timestamp'] = current_time

            # --- Render de estado sobre el frame ---
            xs = [p[0] for p in puntos]
            ys = [p[1] for p in puntos]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

            mensaje = ""
            color = (0, 255, 0)
            grosor = 2
            fase = estado['phase']
            conteo = estado['count']

            if fase == 1:
                mensaje = f"FASE 1: MANO ABIERTA (x{conteo})"
                color = (255, 255, 0)
            elif fase == 2:
                mensaje = f"FASE 2: PULGAR ESCONDIDO (x{conteo})"
                color = (0, 255, 255)
            elif fase == 3:
                if conteo >= self.REPETICIONES_REQUERIDAS:
                    mensaje = "ALERTA: SEÑAL DE AUXILIO DETECTADA"
                    color = (0, 0, 255)
                    grosor = 3
                    señal_detectada = True
                else:
                    mensaje = f"FASE 3: GESTO {conteo}/{self.REPETICIONES_REQUERIDAS}"
                    color = (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor)
            if mensaje:
                cv2.putText(frame, f"[ {confianza:.2f}] {mensaje}", 
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            debug_y = y2 + 20
            cv2.putText(frame, f"Abierto: {dedos_abiertos}", (x1, debug_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(frame, f"PulgarEsc: {pulgar_escondido} ({proporcion_pulgar:.2f})",
                        (x1, debug_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(frame, f"Cerrado: {dedos_cerrados}", (x1, debug_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        for etiqueta in list(self.hand_states.keys()):
            if etiqueta not in manos_activas:
                del self.hand_states[etiqueta]

        return señal_detectada



