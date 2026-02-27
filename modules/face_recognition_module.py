import cv2
import numpy as np
import os
import pandas as pd
import time

class FaceRecognitionModule:
    def __init__(self, face_dir="database/faces", tolerance=0.5):
        self.face_dir = face_dir
        self.tolerance = tolerance # 0.5 is strict, 0.6 is default
        self.known_encodings = []
        self.known_names = []
        
        # Fast Haar Cascade for Detection
        cascade_path = None
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        
        fallback_paths = [
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml'
        ]
        
        for path in fallback_paths:
            if cascade_path is None or not os.path.exists(cascade_path):
                if os.path.exists(path):
                    cascade_path = path
                    break
                    
        self.haar_detector = cv2.CascadeClassifier(cascade_path)
        self.detect_params = {"scaleFactor": 1.1, "minNeighbors": 10, "minSize": (60, 60)}
        
        # Load known faces from disk
        self.load_known_faces()

    def load_known_faces(self):
        """Loads .npy encoding files from disk (created during registration)."""
        print(" [AI] Loading Professional Encodings...")
        self.known_encodings = []
        self.known_names = []
        
        if not os.path.exists(self.face_dir):
            os.makedirs(self.face_dir)
            
        for filename in os.listdir(self.face_dir):
            if filename.endswith('.npy'):
                try:
                    encoding = np.load(os.path.join(self.face_dir, filename))
                    # Group names (Handle Multi-Sync profiles)
                    name = filename.split('.')[0].split('_')[1] # Expecting "enc_Name_1.npy"
                    self.known_encodings.append(encoding)
                    self.known_names.append(name)
                except:
                    continue
        print(f" [AI] Database Ready: {len(set(self.known_names))} Unique Employees Loaded.")

    def register_new_face(self, name, image_path):
        """Generates a 128D Deep Learning encoding and saves it as .npy for speed."""
        import face_recognition # Lazy import to save RAM
        try:
            img = face_recognition.load_image_file(image_path)
            # Detect face specifically for encoding
            locations = face_recognition.face_locations(img, model="hog")
            if not locations:
                return False, "No face found in photo"
                
            # Create encoding (This is the slow part, about 20s on Pi 1)
            encodings = face_recognition.face_encodings(img, known_face_locations=locations, model="small")
            
            if encodings:
                # Save as .npy so the Pi NEVER has to re-calculate this again
                save_path = os.path.join(self.face_dir, f"enc_{name}.npy")
                np.save(save_path, encodings[0])
                self.load_known_faces()
                return True, "Registered!"
            return False, "Encoding failed"
        except Exception as e:
            return False, str(e)

    def detect_and_recognize(self, frame_rgb):
        """Hybrid approach: Fast Haar Detection -> Slow Dlib Matching (Every few seconds)"""
        import face_recognition # Lazy import
        
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.haar_detector.detectMultiScale(gray, **self.detect_params)
        
        recognized_names = []
        for (x, y, w, h) in faces:
            # Add padding for better dlib alignment
            pad = 20
            y1, y2 = max(0, y-pad), min(frame_rgb.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(frame_rgb.shape[1], x+w+pad)
            face_crop = frame_rgb[y1:y2, x1:x2]
            
            # This is the 'Proper' 128D Math (takes ~15 seconds on Pi 1)
            # We only do it once a face is detected by Haar
            # Passing current crop locations to skip Dlib's internal HOG detection
            top, right, bottom, left = pad, face_crop.shape[1]-pad, face_crop.shape[0]-pad, pad
            encodings = face_recognition.face_encodings(face_crop, known_face_locations=[(top, right, bottom, left)], model="small")
            
            if encodings and self.known_encodings:
                matches = face_recognition.compare_faces(self.known_encodings, encodings[0], tolerance=self.tolerance)
                if True in matches:
                    # Voting logic: find best match
                    face_distances = face_recognition.face_distance(self.known_encodings, encodings[0])
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_names[best_match_index]
                        score = int((1 - face_distances[best_match_index]) * 100)
                        print(f" [DEEP AI] Match: {name} (Confidence: {score}%)")
                        recognized_names.append(name)
                        continue
            
            recognized_names.append("Unknown")
            
        return recognized_names
