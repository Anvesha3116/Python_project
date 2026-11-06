import json
import os
import string
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.request
from collections import namedtuple
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")

NUM_HANDS = 2                      # each visible hand is classified on its own
VIDEO_W, VIDEO_H = 560, 420       # size of the video area in the window

# The model outputs a class number; this list turns it back into a character.
# The order must match the order used when the model was trained.
alphabet = [str(i) for i in range(1, 10)] + list(string.ascii_uppercase)

# Which of the 21 hand points are joined by a line when drawing the skeleton.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

BG = "#f6e9d2"       
PANEL = "#fdf6e7"   
INSET = "#f1e2c8"    
PLUM = "#f9d3da"     
FG = "#5a2a3a"       
PINK = "#ec5b78"      
MUTED = "#a0687a"     
ACCENT = "#d9a22b"    
GOOD = "#6fa86f"
BAD = "#c9184a"

Hand = namedtuple("Hand", "points label conf")


# ---------------------------------------------------------------------------
# 2. Setup helpers
# ---------------------------------------------------------------------------
def ensure_files():
    """Return None if everything needed is present, else an error message."""
    if not os.path.exists(MODEL_PATH):
        return f"model.h5 was not found.\n\nPut it in this folder:\n{BASE_DIR}"

    # MediaPipe's Tasks API needs its own hand model; download it once.
    if not os.path.exists(HAND_MODEL_PATH):
        print("Downloading hand_landmarker.task ...")
        tmp = HAND_MODEL_PATH + ".part"
        try:
            with urllib.request.urlopen(HAND_MODEL_URL, timeout=30) as resp, \
                    open(tmp, "wb") as out:
                out.write(resp.read())
            os.replace(tmp, HAND_MODEL_PATH)
        except Exception as e:
            return ("Could not download hand_landmarker.task:\n"
                    f"{e}\n\nDownload it manually from:\n{HAND_MODEL_URL}\n"
                    f"and put it in:\n{BASE_DIR}")
    return None


# ---------------------------------------------------------------------------
# 3. Classifier: Keras if available, else a NumPy fallback (Dense layers only)
# ---------------------------------------------------------------------------
class NumpyDenseModel:
    """Reads model.h5 directly and runs the Dense layers with plain NumPy."""

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


def preprocess(points):
    """(x, y) pixel points -> 42 numbers: relative to the wrist, scaled to -1..1."""
    pts = [[x - points[0][0], y - points[0][1]] for x, y in points]
    flat = [v for p in pts for v in p]
    m = max(abs(v) for v in flat)
    if m == 0:
        return [0.0] * len(flat)
    return [v / m for v in flat]


# ---------------------------------------------------------------------------
# 4. SignEngine: one mirrored camera frame in, a list of Hand results out
# ---------------------------------------------------------------------------
class SignEngine:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        hands = []
        for lms in result.hand_landmarks:
            points = [(min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)) for lm in lms]
            x = np.array(preprocess(points), dtype=np.float32).reshape(1, -1)
            probs = np.asarray(self.model.predict(x, verbose=0))[0]
            idx = int(np.argmax(probs))
            label = alphabet[idx] if idx < len(alphabet) else "?"
            hands.append(Hand(points, label, float(probs[idx])))
        return hands

    def close(self):
        self.detector.close()


def draw_hand(frame, hand, label_colour=(45, 176, 251)):
    """Draw the skeleton and the predicted letter on the frame (BGR colours)."""
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, hand.points[a], hand.points[b], (187, 156, 244), 2)   
    for p in hand.points:
        cv2.circle(frame, p, 4, (74, 24, 201), -1)                           
    x = min(p[0] for p in hand.points)
    y = max(min(p[1] for p in hand.points) - 12, 24)
    cv2.putText(frame, f"{hand.label}  {hand.conf * 100:.0f}%", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_colour, 2)             

# ---------------------------------------------------------------------------
# 5. App: the Tkinter window
# ---------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.engine = None
        self.cap = None
        self.running = False
        self.photo = None            # keeps a reference so Tk doesn't discard the image

        self.sentence = ""
        self.candidate = None        # letter currently being held
        self.cand_since = 0.0
        self.locked = False          # True once the held letter has been added
        self.fps = 0.0
        self.last_t = time.time()

        self.show_skeleton = tk.BooleanVar(value=True)
        self.hold_var = tk.DoubleVar(value=1.2)    # seconds to hold a sign
        self.conf_var = tk.DoubleVar(value=0.60)   # minimum confidence to accept

        self._build_ui()
        self._draw_idle("Loading model ...")
        self._init_job = self.root.after(100, self._init_engine)

    # ---- window ----------------------------------------------------------
    def _build_ui(self):
        r = self.root
        r.title("SignBridge - Indian Sign Language Recognition")
        r.configure(bg=BG, highlightthickness=4,
            highlightbackground=ACCENT, highlightcolor=ACCENT)
        r.resizable(True, True)
        r.minsize(900, 620)
        r.state("zoomed")   # start maximized (Windows)
        r.bind("<F11>", lambda e: r.attributes("-fullscreen",
                                               not r.attributes("-fullscreen")))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Gold.Horizontal.TProgressbar", troughcolor=INSET,
                        background=ACCENT, bordercolor=ACCENT,
                        lightcolor=ACCENT, darkcolor=ACCENT)

        # header
        header = tk.Frame(r, bg=BG)
        header.pack(fill="x", padx=14, pady=(10, 6))
        tk.Label(header, text="SignBridge", bg=BG, fg=PINK,
         font=("Georgia", 20, "italic", "bold")).pack(side="left")

        # controls live at the top so they are always visible
        self._button(header, "Quit", self.on_close).pack(side="right")
        self.start_btn = self._button(header, "Start camera", self.toggle_camera,
                                      accent=True, wide=True)
        self.start_btn.pack(side="right", padx=(0, 6))
        self.start_btn.config(state="disabled")
        self.status_lbl = tk.Label(header, text="", bg=BG, fg=MUTED, font=("Segoe UI", 9))
        self.status_lbl.pack(side="right", padx=10)

        body = tk.Frame(r, bg=BG)
        body.pack(padx=14, pady=(0, 10), fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        # left: video
        self.canvas = tk.Canvas(body, width=VIDEO_W, height=VIDEO_H, bg="black",
                                highlightthickness=3, highlightbackground=ACCENT)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.img_id = self.canvas.create_image(VIDEO_W // 2, VIDEO_H // 2)
        self.idle_id = self.canvas.create_text(VIDEO_W // 2, VIDEO_H // 2, fill=MUTED,
                                               font=("Segoe UI", 14), text="")
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # right: panels
        side = tk.Frame(body, bg=BG)
        side.grid(row=0, column=1, sticky="n", padx=(12, 0))

        # card: detected letter
        card = self._card(side, "DETECTED LETTER")
        self.letter_lbl = tk.Label(card, text="-", bg=PANEL, fg=ACCENT,
                                   font=("Segoe UI", 44, "bold"), width=3)
        self.letter_lbl.pack()
        self.conf_lbl = tk.Label(card, text="Confidence  0%", bg=PANEL, fg=MUTED,
                                 font=("Segoe UI", 9))
        self.conf_lbl.pack(anchor="w")
        self.conf_bar = ttk.Progressbar(card, style="Gold.Horizontal.TProgressbar",
                                        maximum=100, length=270)
        self.conf_bar.pack(pady=(2, 6))
        tk.Label(card, text="Hold the sign to add it", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")
        self.hold_bar = ttk.Progressbar(card, style="Gold.Horizontal.TProgressbar",
                                        maximum=100, length=270)
        self.hold_bar.pack(pady=(2, 2))

        # card: text
        card = self._card(side, "TEXT")
        self.text_lbl = tk.Label(card, text="", bg=INSET, fg=FG, anchor="nw",
                                 justify="left", wraplength=260, width=30, height=2,
                                 font=("Consolas", 13), padx=8, pady=4)
        self.text_lbl.pack(fill="x")
        row = tk.Frame(card, bg=PANEL)
        row.pack(fill="x", pady=(8, 0))
        self._button(row, "Space", self.add_space).pack(side="left", padx=(0, 6))
        self._button(row, "Backspace", self.backspace).pack(side="left", padx=(0, 6))
        self._button(row, "Clear", self.clear_text).pack(side="left", padx=(0, 6))
        self._button(row, "Speak", self.speak, accent=True).pack(side="right")

        # card: settings
        card = self._card(side, "SETTINGS")
        tk.Checkbutton(card, text="Show hand skeleton", variable=self.show_skeleton,
                       bg=PANEL, fg=FG, selectcolor=INSET, activebackground=PANEL,
                       activeforeground=FG, takefocus=0, font=("Segoe UI", 10)).pack(anchor="w")
        self._slider(card, "Hold time (seconds)", self.hold_var, 0.4, 3.0, 0.1)
        self._slider(card, "Minimum confidence", self.conf_var, 0.30, 0.95, 0.05)

        # keyboard shortcuts
        r.bind("<Escape>", lambda e: self.on_close())
        r.bind("<space>", lambda e: self.add_space())
        r.bind("<BackSpace>", lambda e: self.backspace())
        r.bind("<Return>", lambda e: self.speak())
        r.protocol("WM_DELETE_WINDOW", self.on_close)

    def _card(self, parent, title):
        frame = tk.Frame(parent, bg=PANEL, padx=12, pady=6,
                         highlightthickness=2, highlightbackground=ACCENT,
                         highlightcolor=ACCENT)
        frame.pack(fill="x", pady=(0, 6))
        tk.Label(frame, text=title, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        return frame

    def _button(self, parent, text, command, accent=False, wide=False):
        bg, fg = (ACCENT, FG) if accent else (PLUM, FG)
        btn = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                        activebackground=bg, activeforeground=fg, relief="flat",
                        bd=0, takefocus=0, cursor="hand2",
                        font=("Segoe UI", 10, "bold"),
                        padx=16 if wide else 10, pady=6 if wide else 4)
        return btn

    def _slider(self, parent, label, var, lo, hi, step):
        tk.Label(parent, text=label, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 0))
        tk.Scale(parent, from_=lo, to=hi, resolution=step, orient="horizontal",
                 variable=var, bg=PANEL, fg=PINK, troughcolor=INSET,
                 highlightthickness=0, bd=0, length=270, takefocus=0,
                 activebackground=PINK, sliderrelief="flat",
                 sliderlength=24, width=14).pack(anchor="w")

    # ---- start-up --------------------------------------------------------
    def _init_engine(self):
        err = ensure_files()
        if err:
            messagebox.showerror("SignBridge", err)
            self.root.destroy()
            return
        try:
            self.engine = SignEngine()
        except Exception as e:
            messagebox.showerror("SignBridge", f"Could not start the detector:\n{e}")
            self.root.destroy()
            return
        self._draw_idle("Press  Start camera  to begin")
        self.start_btn.config(state="normal")
        self._set_status("Ready")

    # ---- camera loop -----------------------------------------------------
    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        self._set_status("Starting camera ...")
        self.root.update_idletasks()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            self._set_status("Camera not found")
            messagebox.showerror("SignBridge", "Cannot open the webcam.\n"
                                 "Close other apps that may be using it and try again.")
            return
        self.cap = cap
        self.running = True
        self.start_btn.config(text="Stop camera")
        self.last_t = time.time()
        self._loop()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.config(text="Start camera")
        self._reset_readout()
        self._draw_idle("Camera stopped")
        self._set_status("Ready")

    def _loop(self):
        if not self.running:
            return
        try:
            ok, frame = self.cap.read()
            if ok:
                frame = cv2.flip(frame, 1)          # mirror view, same as training
                hands = self.engine.process(frame)
                if self.show_skeleton.get():
                    for hand in hands:
                        draw_hand(frame, hand)
                self._update_readout(hands)
                self._show_frame(frame)

                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self.last_t, 1e-3))
                self.last_t = now
                self._set_status(f"{len(hands)} hand(s)   |   {self.fps:.0f} FPS")
        except Exception as e:
            self.stop_camera()
            messagebox.showerror("SignBridge", f"Something went wrong:\n{e}")
            return
        self.root.after(10, self._loop)

    def _on_canvas_resize(self, event):
        self.canvas.coords(self.img_id, event.width // 2, event.height // 2)
        self.canvas.coords(self.idle_id, event.width // 2, event.height // 2)

    # ---- drawing on the canvas ------------------------------------------
    def _show_frame(self, frame):
        h, w = frame.shape[:2]
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        rgb = cv2.cvtColor(cv2.resize(frame, (nw, nh)), cv2.COLOR_BGR2RGB)
        # Tk can read a raw PPM image directly, so no extra imaging library is needed.
        data = b"P6 %d %d 255\n" % (nw, nh) + rgb.tobytes()
        self.photo = tk.PhotoImage(data=data, format="PPM")
        self.canvas.itemconfig(self.img_id, image=self.photo)
        self.canvas.itemconfig(self.idle_id, text="")

    def _draw_idle(self, message):
        self.canvas.itemconfig(self.img_id, image="")
        self.canvas.itemconfig(self.idle_id, text=message)

    def _set_status(self, text):
        self.status_lbl.config(text=text)

    # ---- readout and word builder ---------------------------------------
    def _reset_readout(self):
        self.candidate = None
        self.locked = False
        self.letter_lbl.config(text="-", fg=ACCENT)
        self.conf_lbl.config(text="Confidence  0%")
        self.conf_bar["value"] = 0
        self.hold_bar["value"] = 0

    def _update_readout(self, hands):
        if not hands:
            self._reset_readout()
            return

        best = max(hands, key=lambda h: h.conf)
        self.conf_lbl.config(text=f"Confidence  {best.conf * 100:.0f}%")
        self.conf_bar["value"] = best.conf * 100

        if best.conf < self.conf_var.get():
            # not sure enough: show it greyed out and do not count it
            self.letter_lbl.config(text=best.label, fg=MUTED)
            self.candidate = None
            self.locked = False
            self.hold_bar["value"] = 0
            return

        self.letter_lbl.config(text=best.label, fg=ACCENT)
        now = time.time()
        if best.label != self.candidate:
            self.candidate = best.label
            self.cand_since = now
            self.locked = False

        hold = self.hold_var.get()
        elapsed = now - self.cand_since
        if not self.locked and elapsed >= hold:
            self._append(best.label)
            self.locked = True      # needs a new sign (or hand removed) before adding again
        self.hold_bar["value"] = 100 if self.locked else min(100, elapsed / hold * 100)

    def _append(self, ch):
        self.sentence += ch
        self._refresh_text()

    def add_space(self):
        self.sentence += " "
        self._refresh_text()

    def backspace(self):
        self.sentence = self.sentence[:-1]
        self._refresh_text()

    def clear_text(self):
        self.sentence = ""
        self._refresh_text()

    def _refresh_text(self):
        self.text_lbl.config(text=self.sentence)

    def speak(self):
        text = self.sentence.strip()
        if not text:
            return
        if os.name != "nt":
            messagebox.showinfo("SignBridge", "Speak uses Windows speech and is only "
                                "available on Windows.")
            return

        def run():
            safe = text.replace("'", "''")
            cmd = ["powershell", "-NoProfile", "-Command",
                   "Add-Type -AssemblyName System.Speech; "
                   f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"]
            try:
                subprocess.run(cmd, creationflags=subprocess.CREATE_NO_WINDOW)
            except Exception:
                pass

        threading.Thread(target=run, daemon=True).start()

    # ---- shutdown --------------------------------------------------------
    def on_close(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        if self.engine is not None:
            self.engine.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
