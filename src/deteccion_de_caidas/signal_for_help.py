import cv2
import numpy as np
import time
from .base_detector import BaseDetector

class SignalForHelpDetector(BaseDetector):
    def __init__(self, model_path="exp.pt"):
        super().__init__(model_path)
        self.enabled = True
        
        # Puntos clave de la mano 
        self.WRIST = 0       # Muneca
        self.INDEX_TIP = 8   # Punta del indice
        self.INDEX_MCP = 5   # Base del indice (Nudillo)
        self.THUMB_TIP = 4   # Punta del pulgar
        self.PINKY_MCP = 17  # Base del menique

        # Calculo del punto maximo requerido para evitar errores de limite de arreglo
        self.max_needed_kpt = max(self.WRIST, self.INDEX_TIP, self.THUMB_TIP, self.PINKY_MCP)
        
        # Memoria interna para recordar el progreso de cada persona
        self.hand_states = {}
        
        # Parametros configurables generales
        self.timeout = 5.0                # Segundos permitidos para completar el gesto
        self.MIN_KP_CONF = 0.15           # Confianza minima para usar una coordenada de la mano
        self.MIN_PALM_SCALE = 1.2         # Tamano minimo de la mano en pixeles
        self.SMOOTH_ALPHA = 0.60          # Velocidad de reaccion del suavizado de movimiento
        
        # Umbrales para decidir los estados de la mano
        self.RATIO_ABIERTO = 1.15         # Distancia minima para considerar dedos estirados
        self.RATIO_CERRADO = 1.05         # Distancia maxima para considerar puno cerrado
        
        # AJUSTA ESTE VALOR SEGUN EL NUMERO QUE VEAS EN LA PANTALLA DE DEBUG
        self.UMBRAL_PULGAR = 0.8          

    def calculate_distance(self, p1, p2):
        # Distancia euclidiana usando NumPy para mayor velocidad
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def draw_hand_skeleton(self, frame, kpts, kpcs):
        # Dibuja puntos y numeros de indice sobre las articulaciones detectadas
        for idx, (kx, ky) in enumerate(kpts):
            if idx < len(kpcs) and kpcs[idx] > 0.20 and (kx > 0 or ky > 0):
                x, y = int(kx), int(ky)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(idx), (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    def _smooth(self, state, key, new_value):
        # Aplica un filtro de suavizado para evitar que las medidas tiemblen de frame a frame
        prev = state.get(key)
        if prev is None:
            state[key] = new_value
        else:
            a = self.SMOOTH_ALPHA
            state[key] = a * new_value + (1 - a) * prev
        return state[key]

    def detect_signal_for_help(self, frame):
        # Verifica si el detector esta activado y el frame es valido
        if not self.enabled or frame is None:
            return False

        signal_detected = False
        active_ids = set() 
        current_time = time.time()

        for boxes, clss, confs, keypoints, kp_confs, track_ids in self.process_tracking(frame):
            for idx, (box, cls, conf, track_id) in enumerate(zip(boxes, clss, confs, track_ids)):
                
                person_id = int(track_id)
                
                # Ignorar identificadores no validos o de baja confianza
                if person_id == -1 or conf < 0.40: 
                    continue

                active_ids.add(person_id) 

                # Validar la existencia de detecciones clave para esta persona
                if keypoints is None or kp_confs is None or len(keypoints) <= idx:
                    continue
                
                kpts = keypoints[idx]
                kpcs = kp_confs[idx]

                self.draw_hand_skeleton(frame, kpts, kpcs)

                # Validar la cantidad de puntos clave generados por la red
                if len(kpcs) <= self.max_needed_kpt:
                    continue

                # Validar que los puntos criticos que necesitamos superen la confianza minima
                needed_points = [self.WRIST, self.INDEX_TIP, self.INDEX_MCP, self.THUMB_TIP, self.PINKY_MCP]
                if any(kpcs[p] < self.MIN_KP_CONF for p in needed_points):
                    continue

                # Registrar un nuevo usuario si acaba de aparecer en camara
                if person_id not in self.hand_states:
                    self.hand_states[person_id] = {'phase': 0, 'timestamp': current_time, 'count': 0}

                state = self.hand_states[person_id]

                # CALCULOS DE DISTANCIA FISICA Y PROPORCIONES
                
                # Medir el tamano base de la palma y suavizar su valor
                raw_palm_scale = self.calculate_distance(kpts[self.INDEX_MCP], kpts[self.WRIST])
                if raw_palm_scale < self.MIN_PALM_SCALE: 
                    continue
                palm_scale = self._smooth(state, 'palm_scale', raw_palm_scale)

                # Verificar el estado del dedo indice para detectar mano abierta o puno
                raw_dist_index_tip_wrist = self.calculate_distance(kpts[self.INDEX_TIP], kpts[self.WRIST])
                dist_index_tip_wrist = self._smooth(state, 'index_tip_wrist', raw_dist_index_tip_wrist)
                
                fingers_open = dist_index_tip_wrist > (palm_scale * self.RATIO_ABIERTO)
                fingers_closed = dist_index_tip_wrist < (palm_scale * self.RATIO_CERRADO)

                # Verificar posicion del pulgar respecto a la base del menique
                raw_dist_thumb_pinky = self.calculate_distance(kpts[self.THUMB_TIP], kpts[self.PINKY_MCP])
                dist_thumb_pinky = self._smooth(state, 'thumb_pinky', raw_dist_thumb_pinky)
                
                proporcion_pulgar = dist_thumb_pinky / palm_scale
                thumb_folded = proporcion_pulgar < self.UMBRAL_PULGAR

                # CONTROLADOR DE LA MAQUINA DE ESTADOS
                
                # Reiniciar si la persona tardo demasiado tiempo a mitad del gesto
                if current_time - state['timestamp'] > self.timeout:
                    state['phase'] = 0
                    state['count'] = 0

                # PASO 1: Inicia con la mano totalmente abierta
                if state['phase'] in [0, 3] and fingers_open and not thumb_folded:
                    state['phase'] = 1
                    state['timestamp'] = current_time

                # PASO 2: Mantiene los dedos estirados pero esconde el pulgar
                elif state['phase'] == 1 and fingers_open and thumb_folded:
                    state['phase'] = 2
                    state['timestamp'] = current_time

                # PASO 3: Cierra todos los dedos formando un puno alrededor del pulgar
                elif state['phase'] == 2 and fingers_closed:
                    state['phase'] = 3
                    state['count'] += 1
                    state['timestamp'] = current_time

                # RENDERIZADO VISUAL EN PANTALLA
                
                x1, y1, x2, y2 = map(int, box)
                message = ""
                color = (0, 255, 0)
                grosor = 2
                
                fase_actual = state['phase']
                conteo_actual = state['count']

                # Asignacion de textos segun el progreso
                if fase_actual == 1:
                    message = f"FASE 1: MANO ABIERTA (x{conteo_actual})"
                    color = (255, 255, 0) 
                
                elif fase_actual == 2:
                    message = f"FASE 2: PULGAR CERRADO (x{conteo_actual})"
                    color = (0, 255, 255) 
                
                elif fase_actual == 3:
                    if conteo_actual >= 3:
                        message = "ALERTA FINAL: AYUDA REQUERIDA"
                        color = (0, 0, 255) 
                        grosor = 3
                        signal_detected = True
                    else:
                        message = f"FASE 3: GESTO {conteo_actual}/3 COMPLETADO"
                        color = (0, 165, 255) 

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor)
                
                if message:
                    cv2.putText(frame, f"ID:{person_id} - {message}", (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # TEXTOS DE DIAGNOSTICO EN TIEMPO REAL
                
                # Mira la fraccion en PulgarDob para calibrar tu variable self.UMBRAL_PULGAR
                debug_y = y2 + 20
                cv2.putText(frame, f"Abierto: {fingers_open}", (x1, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.putText(frame, f"PulgarDob: {thumb_folded} ({proporcion_pulgar:.2f})", (x1, debug_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.putText(frame, f"Cerrado: {fingers_closed}", (x1, debug_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # GESTION Y LIMPIEZA DE MEMORIA
        
        # Eliminar a los sujetos que abandonaron el area de la camara
        stale_ids = set(self.hand_states.keys()) - active_ids 
        for pid in stale_ids:
            del self.hand_states[pid]

        return signal_detected