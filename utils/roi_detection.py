import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Union, Tuple, Optional

class AnemoScanROIDetector:
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model not found at {model_path}')
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names  # {0: 'conjunctiva', 1: 'nail'}

    def detect(self, image_input: Union[str, np.ndarray]) -> List[Dict]:
        
        results = self.model(image_input, conf = self.conf_threshold)
        detections = []
        result = results[0]  # single image
        
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.class_names[cls_id]
                
                # Load original image if path was given, otherwise use already loaded array
                if isinstance(image_input, str):
                    img = cv2.imread(image_input)
                    if img is None:
                        raise ValueError(f'Cannot read image {image_input}')
                else:
                    img = image_input
                
                # Crop ROI
                cropped = img[y1 : y2, x1 : x2].copy()
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'cropped_image': cropped
                })
        return detections

    def extract_and_save_rois(self, image_input: Union[str, np.ndarray], output_folder: str, prefix: str = 'roi') -> List[str]:
        
        os.makedirs(output_folder, exist_ok = True)
        detections = self.detect(image_input)
        saved_paths = []
        for i, det in enumerate(detections):
            class_name = det['class_name']
            cropped = det['cropped_image']
            filename = f'{prefix}{class_name}{i + 1}.jpg'
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, cropped)
            saved_paths.append(save_path)
        return saved_paths

# For convenience, a function to get a singleton detector (load model once)
_detector_instance = None

def get_detector(model_path: Optional[str] = None, conf_threshold: float = 0.5):
    
    global _detector_instance
    if _detector_instance is None:
        if model_path is None:
            model_path = os.path.join('models', 'best.pt')
        _detector_instance = AnemoScanROIDetector(model_path, conf_threshold)
    return _detector_instance