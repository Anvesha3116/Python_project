import os
import sys
import json
import string
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("model.h5 not found at:", MODEL_PATH)
    sys.exit(1)

# Download MediaPipe's hand model once (needed by the Tasks API)
if not os.path.exists(HAND_MODEL_PATH):
    print("Downloading hand_landmarker.task ...")
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    except Exception as e:
        print("Download failed:", e)
        print("Download it manually from:\n", HAND_MODEL_URL)
        print("and put it in:", BASE_DIR)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Classifier model: Keras if available, else NumPy fallback (Dense layers only)
# ---------------------------------------------------------------------------
class NumpyDenseModel:
    def __init__(self, path):
        import h5py
        self.layers = []
        with h5py.File(path, "r") as f:
            cfg = f.attrs["model_config"]
            if isinstance(cfg, bytes):
                cfg = cfg.decode("utf-8")
            cfg = json.loads(cfg)
            layer_cfgs = cfg["config"]["layers"] if isinstance(cfg["config"], dict) else cfg["config"]
            wg = f["model_weights"]
            for lc in layer_cfgs:
                cls, name = lc["class_name"], lc["config"]["name"]
                if cls in ("InputLayer", "Dropout", "Flatten"):
                    continue
                if cls != "Dense":
                    raise RuntimeError(f"Unsupported layer for fallback: {cls}")
                found = {}

                def visit(n, obj):
                    if hasattr(obj, "shape"):
                        found[n.split("/")[-1]] = np.array(obj)

                wg[name].visititems(visit)
                kernel = next(v for k, v in found.items() if k.startswith("kernel"))
                bias = next(v for k, v in found.items() if k.startswith("bias"))
                self.layers.append((kernel, bias, lc["config"].get("activation", "linear")))

    @staticmethod
    def _act(x, a):
        if a == "relu":
            return np.maximum(x, 0)
        if a == "softmax":
            e = np.exp(x - x.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        if a == "sigmoid":
            return 1 / (1 + np.exp(-x))
        if a == "tanh":
            return np.tanh(x)
        return x

    def predict(self, x, verbose=0):
        x = np.asarray(x, dtype=np.float32)
        for k, b, a in self.layers:
            x = self._act(x @ k + b, a)
        return x


def load_model(path):
    try:
        from tensorflow import keras
        return keras.models.load_model(path)
    except Exception as e:
        print("TensorFlow/Keras unavailable:", repr(e)[:150])
        print("Using NumPy fallback (needs: pip install h5py)")
        return NumpyDenseModel(path)


model = load_model(MODEL_PATH)
alphabet = [str(i) for i in range(1, 10)] + list(string.ascii_uppercase)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def preprocess(points):
    """points: list of (x, y) pixel coordinates -> normalized feature vector."""
    pts = [[x - points[0][0], y - points[0][1]] for x, y in points]
    flat = [v for p in pts for v in p]
    m = max(abs(v) for v in flat)
    if m == 0:
        return [0.0] * len(flat)
    return [v / m for v in flat]


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.HandLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            ok, image = cap.read()
            if not ok:
                print("Ignoring empty camera frame.")
                continue
            image = cv2.flip(image, 1)
            h, w = image.shape[:2]

            rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            for i, hand in enumerate(result.hand_landmarks):
                points = [(min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)) for lm in hand]

                for a, b in HAND_CONNECTIONS:
                    cv2.line(image, points[a], points[b], (0, 255, 0), 2)
                for p in points:
                    cv2.circle(image, p, 4, (0, 0, 255), -1)

                x = np.array(preprocess(points), dtype=np.float32).reshape(1, -1)
                pred = model.predict(x, verbose=0)
                idx = int(np.argmax(pred, axis=1)[0])
                label = alphabet[idx] if idx < len(alphabet) else "?"
                cv2.putText(image, label, (50, 50 + i * 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

            cv2.imshow("Indian sign language detector", image)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()