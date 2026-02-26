import cv2
import numpy as np
import os
import pandas as pd

class FaceRecognitionModule:
    def __init__(self, face_dir="database/faces", threshold=80):
        self.face_dir = face_dir
        self.threshold = threshold # Lower is more accurate for LBPH
        
        # Use LBPH - The "Classic" and FASTEST method for Pi 1 hardware
        # This module doesn't need any 100MB files to load!
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            # Fallback if opencv-contrib is not installed
            print("LBPH not found, falling back to basic detector only.")
            self.recognizer = None
            
        # Fast Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            cascade_path = 'haarcascade_frontalface_default.xml'
        self.haar_detector = cv2.CascadeClassifier(cascade_path)
        
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
            
        for filename in os.listdir(self.face_dir):
            if filename.lower().endswith(('.jpg', '.png')):
                name = os.path.splitext(filename)[0]
                path = os.path.join(self.face_dir, filename)
                
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                # Ensure photo is a standard size for training
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
        faces = self.haar_detector.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        
        recognized_names = []
        for (x, y, w, h) in faces:
            if self.recognizer is not None and len(self.label_map) > 0:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (100, 100))
                
                label_id, confidence = self.recognizer.predict(roi_gray)
                
                # Confidence in LBPH is distance (Lower is better)
                if confidence < self.threshold:
                    recognized_names.append(self.label_map[label_id])
                else:
                    recognized_names.append("Unknown")
            else:
                recognized_names.append("Unknown")
        return recognized_names
