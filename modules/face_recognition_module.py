import cv2
import numpy as np
import os
import pandas as pd

class FaceRecognitionModule:
    def __init__(self, face_dir="database/faces", threshold=300):
        self.face_dir = face_dir
        self.threshold = threshold 
        
        # Advanced Contrast Enhancement (CLAHE)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # LBPH Configuration: 
        # radius=1, neighbors=8 (default)
        # grid_x=10, grid_y=10 (Enhanced from 8x8 for better accuracy on Pi 1)
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=10, grid_y=10)
        except AttributeError:
            # Fallback if opencv-contrib is not installed
            print("LBPH not found, falling back to basic detector only.")
            self.recognizer = None
            
        # Robust Haar Cascade path detection for Pi 1
        cascade_path = None
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        
        fallback_paths = [
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml'
        ]
        
        if cascade_path is None or not os.path.exists(cascade_path):
            for path in fallback_paths:
                if os.path.exists(path):
                    cascade_path = path
                    break
                    
        if cascade_path is None:
            print("Warning: Haar cascade file not found. System may fail.")
            self.haar_detector = None
        else:
            # minNeighbors=8 is more "strict" to prevent detecting objects as faces
            self.haar_detector = cv2.CascadeClassifier(cascade_path)
            self.detect_params = {"scaleFactor": 1.1, "minNeighbors": 8, "minSize": (50, 50)}
        
        self.label_map = {} # ID -> Name
        self.load_known_faces()

    def load_known_faces(self):
        """Trains the fast LBPH model from existing photos on the fly."""
        if self.recognizer is None: return
        
        faces = []
        labels = []
        label_id = 0
        
        print("Feeding AI with known employees...")
        if not os.path.exists(self.face_dir):
            os.makedirs(self.face_dir)
            
        # Sort filenames to group multiple images of the same person
        # Pattern: "Name.jpg", "Name_2.jpg", "Name_3.jpg" etc.
        for filename in sorted(os.listdir(self.face_dir)):
            if filename.lower().endswith(('.jpg', '.png')):
                # Extract clean name (remove _1, _2 suffixes)
                raw_name = os.path.splitext(filename)[0]
                name = raw_name.split('_')[0].strip()
                
                path = os.path.join(self.face_dir, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                # Use Advanced CLAHE instead of simple equalization
                img = self.clahe.apply(img)
                img = cv2.resize(img, (100, 100))
                
                faces.append(img)
                labels.append(label_id)
                self.label_map[label_id] = name
                label_id += 1
                
        if len(faces) > 0:
            self.recognizer.train(faces, np.array(labels))
            print(f"Total Database Size: {len(faces)} employees")
        else:
            print("Total Database Size: 0 employees")

    def register_new_face(self, name, image_path):
        """Saves a grayscale face photo and re-trains the model instantly."""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.equalizeHist(img) # Normalize lighting
            face = cv2.resize(img, (100, 100))
            
            save_path = os.path.join(self.face_dir, f"{name}.jpg")
            cv2.imwrite(save_path, face)
            
            # Re-load everything so the AI knows the new face instantly
            self.load_known_faces()
            return True, "Registered!"
        except Exception as e:
            return False, str(e)

    def detect_and_recognize(self, frame_rgb):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        # Higher minNeighbors (8) prevents false detections
        faces = self.haar_detector.detectMultiScale(gray, **self.detect_params)
        
        recognized_names = []
        for (x, y, w, h) in faces:
            if self.recognizer is not None and len(self.label_map) > 0:
                # Add 10% padding to include ears/forehead (improves LBPH match)
                pad_w = int(w * 0.1)
                pad_h = int(h * 0.1)
                y1 = max(0, y - pad_h)
                y2 = min(gray.shape[0], y + h + pad_h)
                x1 = max(0, x - pad_w)
                x2 = min(gray.shape[1], x + w + pad_w)
                
                roi_gray = gray[y1:y2, x1:x2]
                
                # Advanced CLAHE Filtering
                roi_gray = self.clahe.apply(roi_gray)
                roi_gray = cv2.resize(roi_gray, (100, 100))
                
                label_id, confidence = self.recognizer.predict(roi_gray)
                print(f" [AI] Decoded Face: {self.label_map.get(label_id, '???')} (Score: {int(confidence)} / Limit: {self.threshold})")
                
                # Confidence in LBPH is distance (Lower is better)
                if confidence < self.threshold:
                    recognized_names.append(self.label_map[label_id])
                else:
                    recognized_names.append("Unknown")
            else:
                recognized_names.append("Unknown")
        return recognized_names
