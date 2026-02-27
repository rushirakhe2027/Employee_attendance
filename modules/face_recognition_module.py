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

    # Tightened tolerance to 0.38 to prevent 'wrong predictions'
    # Smaller = Stricter. 
    TOLERANCE = 0.38 
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
        Loads face encodings from .npy files.
        Supports both:
          - (128,)   shape → single encoding (old format)
          - (N, 128) shape → multi-sample encoding (new format, more accurate)
        For multi-sample files, ALL N encodings are stored separately so
        matching uses the BEST (minimum) distance across all samples.
        """
        self.known_encodings = []  # flat list of 128-d vectors
        self.known_names = []      # parallel list of names

        npy_files = [f for f in os.listdir(FACE_DIR) if f.endswith('.npy')]

        if not npy_files:
            print("[FaceModule] Database is empty. No known faces loaded.")
            return

        for fname in npy_files:
            name = os.path.splitext(fname)[0]
            # Strip sample suffix like 'rushikesh_0', 'rushikesh_1' → 'rushikesh'
            base_name = name.rsplit('_', 1)[0] if name[-1].isdigit() else name
            path = os.path.join(FACE_DIR, fname)
            try:
                data = np.load(path)
                if data.ndim == 1 and data.shape[0] == 128:
                    # Old single-encoding format
                    self.known_encodings.append(data)
                    self.known_names.append(base_name)
                elif data.ndim == 2 and data.shape[1] == 128:
                    # New multi-sample format (N, 128)
                    for enc in data:
                        self.known_encodings.append(enc)
                        self.known_names.append(base_name)
                    print(f"[FaceModule] Loaded {data.shape[0]} samples for '{base_name}'")
                else:
                    print(f"[FaceModule] Skipping corrupt file: {fname}")
            except Exception as e:
                print(f"[FaceModule] Error loading {fname}: {e}")

        # Deduplicate names for display
        unique = list(dict.fromkeys(self.known_names))
        print(f"[FaceModule] Loaded {len(self.known_encodings)} encoding(s) for {len(unique)} employee(s): {unique}")

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
            scaleFactor=1.1,   # Faster detection
            minNeighbors=5,    # More sensitive
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

        # --- SPEED: Shrink 50% before dlib ---
        small_frame = cv2.resize(frame_rgb, (0, 0), fx=self.SCALE, fy=self.SCALE)

        # --- ACCURACY: CLAHE on the dlib input frame too ---
        # Convert to LAB, apply CLAHE only to L channel, convert back
        lab = cv2.cvtColor(small_frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        small_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        sx = int(x * self.SCALE)
        sy = int(y * self.SCALE)
        sw = int(w * self.SCALE)
        sh = int(h * self.SCALE)
        face_location_small = [(sy, sx + sw, sy + sh, sx)]

        try:
            encodings = face_recognition.face_encodings(
                small_frame,
                known_face_locations=face_location_small,
                num_jitters=1, # Add 1-shot jitter for stability
                model="large"  # High precision
            )
        except Exception as e:
            print(f"[FaceModule] Encoding error: {e}")
            return ["Unknown"]

        if not encodings:
            return ["Unknown"]

        face_encoding = encodings[0]

        if not self.known_encodings:
            return ["Unknown"]

        # 1. Calculate distances to all stored sample encodings
        distances = face_recognition.face_distance(self.known_encodings, face_encoding)
        best_match_index = np.argmin(distances)
        min_distance = distances[best_match_index]

        # 2. Threshold Check
        if min_distance <= self.TOLERANCE:
            name = self.known_names[best_match_index]
            confidence = round((1 - min_distance) * 100, 1)
            print(f"[AI] Confirmed Match: {name} ({confidence}% confidence)")
            return [name]
        else:
            # Optional: Log close but failed matches for tuning
            if min_distance < 0.5:
                print(f"[AI] Rejected Candidate: {self.known_names[best_match_index]} (too far: {round(min_distance, 3)})")
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
        """
        Multi-Sample Registration — the most impactful accuracy upgrade.
        Instead of 1 photo, captures 5 frames spread over ~3 seconds.
        All 5 encodings are stored as a (5, 128) array in one .npy file.

        During recognition, ALL 5 are compared and the BEST (minimum)
        distance is used — so the system recognises you even from a
        slightly different angle or in different lighting.

        Args:
            name      : employee name string
            frames_rgb: list of 5 RGB frames captured by face_app
        """
        if not isinstance(frames_rgb, list):
            frames_rgb = [frames_rgb]  # backwards-compat with single frame

        collected_encodings = []
        reference_face_crop = None

        for idx, frame_rgb in enumerate(frames_rgb):
            face_crop, bbox = self._get_stable_face(frame_rgb)
            if face_crop is None or bbox is None:
                print(f"[FaceModule] Sample {idx+1}: No face detected, skipping.")
                continue

            x, y, w, h = bbox
            face_location = [(y, x + w, y + h, x)]

            try:
                encs = face_recognition.face_encodings(
                    frame_rgb, known_face_locations=face_location, model="small"
                )
                if encs:
                    collected_encodings.append(encs[0])
                    print(f"[FaceModule] Sample {idx+1}/{len(frames_rgb)} captured. ✓")
                    if reference_face_crop is None:
                        reference_face_crop = face_crop
            except Exception as e:
                print(f"[FaceModule] Sample {idx+1} encoding error: {e}")

        if len(collected_encodings) < 2:
            return False, f"Only got {len(collected_encodings)} good sample(s). Need at least 2. Try better lighting."

        # Stack into (N, 128) array and save as single .npy
        multi_encoding = np.stack(collected_encodings)  # shape: (N, 128)
        npy_path = os.path.join(FACE_DIR, f"{name}.npy")
        np.save(npy_path, multi_encoding)
        print(f"[FaceModule] Saved {len(collected_encodings)}-sample encoding to {npy_path}")

        # Save reference photo
        if reference_face_crop is not None:
            jpg_path = os.path.join(FACE_DIR, f"{name}.jpg")
            face_bgr = cv2.cvtColor(reference_face_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(jpg_path, face_bgr)
            print(f"[FaceModule] Saved reference photo to {jpg_path}")

        self.load_known_faces()
        return True, f"Registered {name} with {len(collected_encodings)} face samples."
