import cv2
import dlib
import numpy as np
import face_recognition
import os
import pandas as pd
from scipy.spatial.distance import cosine
from collections import defaultdict

class FaceRecognitionModule:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat", 
                 face_dir="database/faces", threshold=0.6):
        self.predictor_path = predictor_path
        self.face_dir = face_dir
        self.enc_dir = os.path.join(os.path.dirname(face_dir), "encodings")
        self.threshold = threshold
        
        if not os.path.exists(self.enc_dir):
            os.makedirs(self.enc_dir, exist_ok=True)

        
        # Initialize dlib components
        self.face_detector = dlib.get_frontal_face_detector()
        try:
            self.shape_predictor = dlib.shape_predictor(self.predictor_path)
        except Exception as e:
            print(f"Warning: Could not load shape predictor from {predictor_path}: {e}")
            self.shape_predictor = None
            
        self.known_faces = defaultdict(list)
        self.known_names = []
        self.load_known_faces()

    def align_face(self, image, face):
        if self.shape_predictor is None:
            return None
        try:
            shape = self.shape_predictor(image, face)
            face_chip = dlib.get_face_chip(image, shape)
            return face_chip
        except Exception as e:
            print(f"Face alignment failed: {e}")
            return None

    def get_face_encodings(self, image_np, num_samples=5):
        try:
            face_locations = face_recognition.face_locations(image_np, model="hog")
            if not face_locations:
                return []
            
            encodings = []
            for i in range(num_samples):
                encoding = face_recognition.face_encodings(
                    image_np, 
                    face_locations,
                    num_jitters=2,
                    model="large"
                )
                if encoding:
                    encodings.append(encoding[0])
            return encodings
        except Exception as e:
            print(f"Error getting face encodings: {e}")
            return []

    def register_new_face(self, name, image_path):
        """Registers a new face: aligns, extracts multiple encodings, and saves both image and encoding."""
        try:
            image = face_recognition.load_image_file(image_path)
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                return False, "No face detected in the image."
            
            # Use the first face found
            top, right, bottom, left = face_locations[0]
            # Convert to dlib rectangle
            face_rect = dlib.rectangle(left, top, right, bottom)
            
            # Align face
            aligned_face = self.align_face(image, face_rect)
            if aligned_face is None:
                # Fallback to simple crop
                aligned_face = image[top:bottom, left:right]

            # Better quality: convert aligned_face (which is RGB from face_recognition/dlib) to BGR for saving with cv2
            aligned_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(self.face_dir, f"{name}.jpg")
            cv2.imwrite(save_path, aligned_bgr)

            # Extract robust encoding (multiple jitters)
            encodings = face_recognition.face_encodings(aligned_face, num_jitters=10, model="large")
            if not encodings:
                return False, "Failed to extract features from face."

            # Save encoding to file for fast loading later
            enc_path = os.path.join(self.enc_dir, f"{name}.npy")
            np.save(enc_path, encodings[0])
            
            # Update local list
            self.known_faces[name] = [encodings[0]]
            if name not in self.known_names:
                self.known_names.append(name)
            
            return True, "Face registered successfully."
        except Exception as e:
            return False, f"Registration error: {str(e)}"

    def load_known_faces(self):
        print("Loading known faces and encodings...")
        self.known_faces = defaultdict(list)
        self.known_names = []
        
        if not os.path.exists(self.enc_dir):
            os.makedirs(self.enc_dir, exist_ok=True)

        # Load pre-saved .npy encodings (faster)
        for filename in os.listdir(self.enc_dir):
            if filename.lower().endswith('.npy'):
                name = os.path.splitext(filename)[0]
                try:
                    encoding = np.load(os.path.join(self.enc_dir, filename))
                    self.known_faces[name] = [encoding]
                    self.known_names.append(name)
                except Exception as e:
                    print(f"Error loading encoding {filename}: {e}")

        # Sync with images (if image exists but no encoding, generate it)
        for filename in os.listdir(self.face_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                if name not in self.known_names:
                    print(f"Generating missing encoding for: {name}")
                    path = os.path.join(self.face_dir, filename)
                    success, _ = self.register_new_face(name, path)
                    if success:
                        print(f"Recovered encoding for {name}")

        print(f"Total Database Size: {len(self.known_names)} employees")


    def verify_face(self, face_encoding):
        best_match = None
        min_dist = 1.0
        
        for name, saved_encodings in self.known_faces.items():
            distances = [cosine(face_encoding, saved_encoding) for saved_encoding in saved_encodings]
            if not distances:
                continue
            
            # Average of top matches
            distances.sort()
            avg_distance = sum(distances[:3]) / min(3, len(distances))
            
            if avg_distance < self.threshold and avg_distance < min_dist:
                min_dist = avg_distance
                best_match = name
                
        return best_match if min_dist < self.threshold else None

    def detect_and_recognize(self, frame_rgb):
        # Convert to gray for dlib if needed, but dlib likes RGB too
        dlib_faces = self.face_detector(frame_rgb, 1)
        
        recognized_names = []
        
        for face in dlib_faces:
            aligned = self.align_face(frame_rgb, face)
            if aligned is None:
                # Fallback to simple crop if alignment fails
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                aligned = frame_rgb[max(0, y):y+h, max(0, x):x+w]
            
            try:
                encodings = face_recognition.face_encodings(aligned)
                if encodings:
                    name = self.verify_face(encodings[0])
                    recognized_names.append(name if name else "Unknown")
            except Exception:
                continue
                
        return recognized_names
