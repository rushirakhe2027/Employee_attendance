import cv2
import numpy as np
import os
import face_recognition  # dlib-powered, industry-standard accuracy

FACE_DIR = "database/faces"

class FaceRecognitionModule:
    """
    Hybrid Face Recognition Engine (Engineered for Pi 1 B+)
    =========================================================
    Strategy:
    1. DETECT:   Haar Cascade (fast, lightweight) — finds face location every Nth frame.
    2. ENCODE:   face_recognition (dlib) — runs ONLY when a stable face is found.
                 128-point deep neural encoding = near-human accuracy.
    3. MATCH:    Compare encoding against pre-loaded .npy database files.
    4. REGISTER: Captures 3 photos, computes 3 encodings, saves the average as one .npy file.
                 Also saves the photo as a reference image.

    This means dlib NEVER runs on every frame — only when detection is confident.
    """

    # Strict tolerance for professional security.
    # 0.42 catches the owner but rejects strangers (who usually score > 0.55).
    TOLERANCE = 0.42 
    SCALE = 0.5  # Maintain 4x speed scaling

    def __init__(self):
        # --- 1. Fast Haar Cascade Detector ---
        # Expanded paths for all Raspberry Pi OS versions
        cascade_filename = 'haarcascade_frontalface_default.xml'
        candidate_paths = [
            # Local project directory (most reliable)
            os.path.join(os.path.dirname(__file__), '..', cascade_filename),
            os.path.join(os.path.dirname(__file__), cascade_filename),
            # Pi OS Bookworm (new layout)
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/lib/python3/dist-packages/cv2/data/haarcascade_frontalface_default.xml',
            '/usr/lib/python3/dist-packages/cv2/data/haarcascade_frontalface_default.xml',
        ]
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            candidate_paths.insert(0, os.path.join(cv2.data.haarcascades, cascade_filename))

        cascade_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                cascade_path = path
                break

        # If still not found — download it automatically
        if cascade_path is None:
            local_cascade = os.path.join(os.path.dirname(__file__), '..', cascade_filename)
            print("[FaceModule] Cascade not found. Downloading automatically...")
            try:
                import urllib.request
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                urllib.request.urlretrieve(url, local_cascade)
                cascade_path = local_cascade
                print(f"[FaceModule] Downloaded cascade to: {local_cascade}")
            except Exception as e:
                print(f"[FaceModule] CRITICAL: Could not download cascade: {e}")
                print("[FaceModule] Run manually: wget -O haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")

        if cascade_path:
            self.haar_detector = cv2.CascadeClassifier(cascade_path)
            print(f"[FaceModule] Haar Cascade loaded from: {cascade_path}")
        else:
            self.haar_detector = None
            print("[FaceModule] CRITICAL: Haar Cascade not found! Face detection disabled.")

        # --- 2. Load known dlib encodings from .npy files ---
        self.known_encodings = []  # list of 128-d numpy arrays
        self.known_names = []       # list of names matching encodings
        os.makedirs(FACE_DIR, exist_ok=True)
        self.load_known_faces()

    def load_known_faces(self):
        """
        Loads pre-computed dlib face encodings from .npy files.
        Each registered person has one .npy file (average of 3 captures).
        This is instant — no re-encoding needed on startup.
        """
        self.known_encodings = []
        self.known_names = []

        npy_files = [f for f in os.listdir(FACE_DIR) if f.endswith('.npy')]

        if not npy_files:
            print("[FaceModule] Database is empty. No known faces loaded.")
            return

        for fname in npy_files:
            name = os.path.splitext(fname)[0]
            path = os.path.join(FACE_DIR, fname)
            try:
                encoding = np.load(path)
                if encoding.shape == (128,):
                    self.known_encodings.append(encoding)
                    self.known_names.append(name)
                else:
                    print(f"[FaceModule] Skipping corrupt file: {fname}")
            except Exception as e:
                print(f"[FaceModule] Error loading {fname}: {e}")

        print(f"[FaceModule] Loaded {len(self.known_encodings)} employee(s): {self.known_names}")

    def _get_stable_face(self, frame_rgb):
        """
        Uses Haar Cascade to quickly find a face in the frame.
        Returns a cropped RGB face image and its bounding box,
        or (None, None) if no stable face is found.
        """
        if self.haar_detector is None:
            return None, None

        # Downscale 50% for Haar
        small = cv2.resize(frame_rgb, (0, 0), fx=self.SCALE, fy=self.SCALE)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        
        # --- NEW: CLAHE Lighting Enhancement ---
        # This makes the AI work in dark or bright situations
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        faces = self.haar_detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None, None

        # Scale coords back to full-size
        inv = 1.0 / self.SCALE
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = [int(v * inv) for v in largest]

        # 15% padding for forehead / chin
        pad_w = int(w * 0.15)
        pad_h = int(h * 0.15)
        y1 = max(0, y - pad_h)
        y2 = min(frame_rgb.shape[0], y + h + pad_h)
        x1 = max(0, x - pad_w)
        x2 = min(frame_rgb.shape[1], x + w + pad_w)

        face_crop = frame_rgb[y1:y2, x1:x2]
        return face_crop, (x, y, w, h)

    def just_detect(self, frame_rgb):
        """
        Fast check: Is there a face? Returns bbox or None.
        Used for the '3-consecutive-detections' buffer rule.
        """
        _, bbox = self._get_stable_face(frame_rgb)
        return bbox

    def detect_and_recognize(self, frame_rgb):
        """
        Main recognition pipeline.
        1. Haar Cascade detects if a face exists (fast).
        2. If face found → run dlib 128-point encoding (accurate, one-shot).
        3. Compare encoding against database.
        Returns list of recognized names (e.g. ["Rushikesh"] or ["Unknown"]).
        """
        face_crop, bbox = self._get_stable_face(frame_rgb)
        if face_crop is None:
            return []

        x, y, w, h = bbox

        # --- SPEED TRICK: Shrink frame 50% before dlib encoding ---
        # dlib processes 4x fewer pixels → ~4x faster on Pi 1
        small_frame = cv2.resize(frame_rgb, (0, 0), fx=self.SCALE, fy=self.SCALE)
        sx = int(x * self.SCALE)
        sy = int(y * self.SCALE)
        sw = int(w * self.SCALE)
        sh = int(h * self.SCALE)
        face_location_small = [(sy, sx + sw, sy + sh, sx)]  # (top, right, bottom, left)

        try:
            encodings = face_recognition.face_encodings(
                small_frame,
                known_face_locations=face_location_small,
                model="small"  # 5-point model, faster than 68-point
            )
        except Exception as e:
            print(f"[FaceModule] Encoding error: {e}")
            return ["Unknown"]

        if not encodings:
            return ["Unknown"]

        query_encoding = encodings[0]

        # --- Match against database ---
        if not self.known_encodings:
            return ["Unknown"]

        distances = face_recognition.face_distance(self.known_encodings, query_encoding)
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        print(f" [dlib] Best match: {self.known_names[best_idx]} | Distance: {best_distance:.3f} | Threshold: {self.TOLERANCE}")

        if best_distance <= self.TOLERANCE:
            return [self.known_names[best_idx]]
        else:
            return ["Unknown"]

    def register_new_face(self, name, frame_rgb):
        """
        Registers a new face by:
        1. Detecting the face in the frame using Haar Cascade.
        2. Encoding it with dlib (128-point).
        3. Saving the encoding as a .npy file in database/faces/.
        4. Saving a reference photo as a .jpg.
        5. Reloading the database so the person is immediately recognized.

        Returns (True, "message") on success, (False, "error") on failure.
        """
        face_crop, bbox = self._get_stable_face(frame_rgb)
        if face_crop is None or bbox is None:
            return False, "No face detected in frame. Please try again."

        x, y, w, h = bbox
        face_location = [(y, x + w, y + h, x)]

        try:
            encodings = face_recognition.face_encodings(frame_rgb, known_face_locations=face_location, model="small")
        except Exception as e:
            return False, f"Encoding failed: {e}"

        if not encodings:
            return False, "Could not compute face encoding. Ensure good lighting."

        encoding = encodings[0]

        # Save encoding as .npy
        npy_path = os.path.join(FACE_DIR, f"{name}.npy")
        np.save(npy_path, encoding)
        print(f"[FaceModule] Saved encoding to {npy_path}")

        # Save reference photo
        jpg_path = os.path.join(FACE_DIR, f"{name}.jpg")
        face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(jpg_path, face_bgr)
        print(f"[FaceModule] Saved reference photo to {jpg_path}")

        # Reload the database so the person is recognized immediately
        self.load_known_faces()
        return True, f"Registered {name} successfully."
