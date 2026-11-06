import os
import sys
import csv
import string
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "images", "data")
CSV_PATH = os.path.join(BASE_DIR, "keypoint.csv")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

LABELS = list(string.ascii_uppercase) + [str(i) for i in range(1, 10)]
EXTS = (".jpg", ".jpeg", ".png")


def preprocess(points):
    # IDENTICAL to isl_detection_v2.py: relative to wrist, flatten, divide by max
    pts = [[x - points[0][0], y - points[0][1]] for x, y in points]
    flat = [v for p in pts for v in p]
    m = max(abs(v) for v in flat)
    if m == 0:
        return None
    return [v / m for v in flat]


def main():
    if not os.path.isdir(DATA_DIR):
        print("Dataset folder not found:", DATA_DIR)
        sys.exit(1)
    if not os.path.exists(HAND_MODEL_PATH):
        print("Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )

    counts = {}
    with vision.HandLandmarker.create_from_options(options) as detector, \
            open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        for label in LABELS:
            folder = os.path.join(DATA_DIR, label)
            if not os.path.isdir(folder):
                print("WARNING: missing folder for", label)
                continue
            files = sorted(x for x in os.listdir(folder) if x.lower().endswith(EXTS))
            n = 0
            for name in files:
                image = cv2.imread(os.path.join(folder, name))
                if image is None:
                    continue
                image = cv2.flip(image, 1)  # same mirroring as live detection
                h, w = image.shape[:2]
                rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                result = detector.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                for hand in result.hand_landmarks:
                    points = [(min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1))
                              for lm in hand]
                    row = preprocess(points)
                    if row is not None:
                        writer.writerow([label, *row])
                        n += 1
            counts[label] = n
            print(f"{label}: {n} rows from {len(files)} images")

    print("\nDone. Total rows:", sum(counts.values()), "-> ", CSV_PATH)
    print("Labels found:", len(counts), "(should be 35)")


if __name__ == "__main__":
    main()
