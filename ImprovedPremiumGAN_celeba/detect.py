
from ultralytics import YOLO
import cv2
import numpy as np
import os


class MaskDetector:
    """
    Detects face region and creates mask coordinates matching the training data.
    Uses OpenCV DNN face detector for accurate face detection.
    """
    def __init__(self, model_path=None):
        # Use OpenCV DNN face detector
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "models")
        
        prototxt_path = os.path.join(model_dir, "deploy.prototxt")
        caffemodel_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        
        # Check if models exist
        if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            self.use_dnn = True
        else:
            # Fallback to basic detection
            print("Warning: Face detector model not found. Using fallback detection.")
            self.use_dnn = False

    def detect_mask(self, image_path):
        """
        Detects face region and returns mask coordinates.
        
        Returns: 
            image (numpy BGR), box (x, y, w, h) of the mask region
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not open or find the image: {image_path}")

        h, w = img.shape[:2]
        best_box = None
        
        if self.use_dnn:
            # Use OpenCV DNN for face detection
            blob = cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)), 
                1.0, 
                (300, 300), 
                (104.0, 177.0, 123.0)
            )
            
            self.net.setInput(blob)
            detections = self.net.forward()
            
            best_confidence = 0
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        
                        face_h = y2 - y1
                        face_w = x2 - x1
                        
                        # Mask region: lower 45% of face (from 55% to 100%)
                        # This matches prepare_masked_dataset.py settings
                        mask_x = x1
                        mask_y = y1 + int(face_h * 0.55)
                        mask_w = face_w
                        mask_h = int(face_h * 0.45)
                        
                        best_box = (mask_x, mask_y, mask_w, mask_h)
        
        # Fallback: CelebA images are already face-centered
        if best_box is None:
            mask_y_start = int(h * 0.55)
            mask_h = int(h * 0.45)
            mask_x_start = 0
            mask_w = w
            best_box = (mask_x_start, mask_y_start, mask_w, mask_h)

        return img, best_box

    def apply_blackout(self, img, box):
        """
        Black out the region defined by box.
        box: (x, y, w, h)
        """
        if box is None:
            return img 
            
        x, y, w, h = box
        img_h, img_w = img.shape[:2]
        
        # Clip to image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        # Black out the region
        img[y:y+h, x:x+w] = 0
        return img
