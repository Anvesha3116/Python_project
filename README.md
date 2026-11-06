# SignBridge: Indian Sign Language Recognition

SignBridge recognises Indian Sign Language (ISL) hand signs from a webcam in real time and turns them into text and speech. It detects hand landmarks with MediaPipe and classifies them with a small neural network. Each sign is one of the letters **A-Z** or digits **1-9** (35 classes).

## Features

- Live webcam recognition of up to 2 hands, each classified separately
- Shows the detected letter with a confidence bar
- Hold a sign for a moment to add it to the text, which avoids accidental letters
- Text builder with Space, Backspace and Clear buttons
- **Speak** button reads the text aloud (Windows only)
- Adjustable hold time and minimum confidence, and an option to hide the hand skeleton
- Resizable window with maximize and minimize, plus `F11` full screen
- Cream, pink and gold interface theme
- Works without TensorFlow: if it isn't installed, the model is run with NumPy instead

## Project structure

| File | What it does |
|---|---|
| `isl_detectionll.py` | **Main app.** The SignBridge window (Tkinter) with the camera, letter display, text box and settings |
| `isl_detection.py` | Simple version that shows the camera and predicted letter in an OpenCV window |
| `dataset_keypoint_generation_v2.py` | Builds `keypoint.csv` from the image dataset using the current MediaPipe Tasks API |
| `dataset_keypoint_generation.py` | Older version of the dataset script (legacy MediaPipe API) |
| `ISL_classifier.ipynb` | Notebook that trains the classifier and produces `model.h5` |
| `keypoint.csv` | Hand landmark data (42 values per row) extracted from the dataset |
| `model.h5` | Trained classifier used by the app |
| `model_old.h5` | Earlier version of the classifier, kept as a backup |
| `hand_landmarker.task` | MediaPipe hand detection model (downloaded automatically if missing) |
| `requirements.txt` | Python packages needed to run the project |

## Getting started

### Requirements

- Python 3.9 or newer (developed on Python 3.12)
- A webcam
- Windows is recommended, because the Speak button uses Windows speech

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/Anvesha3116/signbridge-isl-recognition.git
   cd signbridge-isl-recognition
   ```

2. Install the packages:

   ```
   pip install -r requirements.txt
   ```

### Run the app

```
python isl_detectionll.py
```

1. Wait for the window to say **Ready**, then click **Start camera**.
2. Show a sign to the camera. The detected letter and its confidence appear on the right.
3. Hold the sign steady until the hold bar fills up and the letter is added to the text.
4. Click **Speak** to hear the text.

**Keyboard shortcuts**

| Key | Action |
|---|---|
| `Space` | Add a space |
| `Backspace` | Delete the last character |
| `Enter` | Speak the text |
| `F11` | Toggle full screen |
| `Esc` | Quit |

To use the simple OpenCV version instead, run `python isl_detection.py` and press `Esc` to exit.

## How it works

1. **Hand detection:** MediaPipe finds the 21 landmark points of each hand in every camera frame.
2. **Preprocessing:** the points are made relative to the wrist and scaled to the range -1 to 1, giving 42 numbers per hand. This makes the result independent of hand position and size in the frame.
3. **Classification:** a feedforward neural network predicts which of the 35 signs the 42 numbers represent.
4. **Text building:** a letter is only added once it has been held above the minimum confidence for the chosen hold time.

## Training your own model

1. Download the [ISL image dataset from Kaggle](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl) and arrange it as `images/data/<label>/`, with one folder per sign: `A` to `Z` and `1` to `9`.
2. Generate the landmark file (this overwrites `keypoint.csv`):

   ```
   python dataset_keypoint_generation_v2.py
   ```

3. Open `ISL_classifier.ipynb`, run all cells to train the classifier, and save it as `model.h5`.

## Future improvements

- Recognise words and full ISL gestures, not just letters and digits
- Try a more advanced model to improve accuracy
- Train on a larger and more varied dataset
- Add speech output on macOS and Linux

## Acknowledgements

- Dataset: [Indian Sign Language (ISL) on Kaggle](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
- Hand tracking: [MediaPipe](https://developers.google.com/mediapipe)