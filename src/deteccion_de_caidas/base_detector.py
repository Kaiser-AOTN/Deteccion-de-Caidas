from ultralytics import YOLO

class BaseDetector:
    def __init__(self, model_path):
        # Carga el modelo YOLO de la ruta que le enviemos (ej: yolov8n-pose.pt o best.pt)
        self.model = YOLO(model_path)

    def process_tracking(self, frame):
        """
        Método común para procesar el video.
        Se encarga de pasarle el fotograma a YOLO y extraer los datos limpios.
        """
        # Ejecuta el seguimiento (tracking) sobre el fotograma actual
        results = self.model.track(frame, persist=True, verbose=False, device="cpu")
        
        for r in results:
            # Si no hay detecciones en la pantalla, saltamos al siguiente fotograma
            if r.boxes is None or len(r.boxes) == 0:
                continue

            # Convertimos las coordenadas de las cajas, clases y confianzas a arreglos de NumPy
            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            
            # Extraemos los puntos clave (Keypoints) si el modelo los soporta (si es de Pose)
            keypoints = r.keypoints.xy.cpu().numpy() if hasattr(r, 'keypoints') and r.keypoints is not None else None
            kp_confs = r.keypoints.conf.cpu().numpy() if hasattr(r, 'keypoints') and r.keypoints is not None and r.keypoints.conf is not None else None
            
            # Obtenemos el ID único de seguimiento de cada persona o mano
            track_ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(boxes)

            # Enviamos los datos procesados a la clase que solicitó la detección
            yield boxes, clss, confs, keypoints, kp_confs, track_ids