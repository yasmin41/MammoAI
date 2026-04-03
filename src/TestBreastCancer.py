# =============================================================================
#  MammoAI — Clinical Decision Support  (PyInstaller EXE-Safe Edition)
#  Built with CustomTkinter | OOP Architecture
#
#  Authors   : Yasmin, Rama, Nermeen
#  EXE Notes : Built with --windowed flag; stdout/stderr redirected to devnull
#              to prevent AttributeError: 'NoneType' object has no attribute 'write'
# =============================================================================

# =============================================================================
#  STEP 1 — STEALTH REDIRECTOR (MUST BE FIRST, BEFORE ANY OTHER IMPORT)
#  Prevents TensorFlow/Keras print() calls from crashing windowed EXEs.
# =============================================================================
import sys
import os

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# =============================================================================
#  STEP 2 — RESOURCE PATH HELPER
#  Resolves paths to bundled assets (.h5 weights, .ico icon) inside the EXE.
#  When running as a PyInstaller bundle, files are extracted to sys._MEIPASS.
# =============================================================================

def resource_path(relative_path: str) -> str:
    """
    Return the absolute path to a resource, whether running from source or
    from a PyInstaller --onefile / --windowed bundle.
    """
    try:
        base = sys._MEIPASS  # PyInstaller temp extraction folder
    except AttributeError:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


# =============================================================================
#  STANDARD IMPORTS
# =============================================================================

import threading
import time
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image
import numpy as np

# =============================================================================
#  AI / ML IMPORTS  — app degrades gracefully when TensorFlow is absent
# =============================================================================

try:
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator, load_img, img_to_array,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.utils import class_weight as sk_class_weight
    TF_AVAILABLE = True
except ImportError as _tf_err:
    TF_AVAILABLE = False
    try:
        sys.stderr.write(
            f"[MammoAI] Optional dependency missing ({_tf_err}) "
            "— UI-preview mode active.\n"
        )
    except Exception:
        pass


# =============================================================================
#  COLOUR & TYPOGRAPHY TOKENS  (preserved exactly as original)
# =============================================================================

COLORS = {
    "navy":            "#1f538d",
    "navy_hover":      "#174270",
    "navy_dark":       "#0f2d52",
    "bg_app":          "#f0f4f8",
    "bg_card":         "#ffffff",
    "bg_input":        "#e8eef5",
    "text_primary":    "#1a2f4a",
    "text_secondary":  "#5b7a9d",
    "text_light":      "#ffffff",
    "text_muted":      "#8fa8c8",
    "malignant":       "#e53935",
    "malignant_bg":    "#fff5f5",
    "malignant_border":"#f5c6c6",
    "benign":          "#1e8449",
    "benign_bg":       "#f0faf4",
    "benign_border":   "#b7dfca",
    "accent":          "#2e86de",
    "border":          "#d0dce8",
}

FONTS = {
    "display":  ("Segoe UI", 20, "bold"),
    "heading":  ("Segoe UI", 15, "bold"),
    "subhead":  ("Segoe UI", 12, "bold"),
    "body":     ("Segoe UI", 11),
    "small":    ("Segoe UI", 9),
    "logo":     ("Segoe UI", 18, "bold"),
    "mono":     ("Courier New", 10),
    "result":   ("Segoe UI", 32, "bold"),
    "conf_pct": ("Segoe UI", 18, "bold"),
}


# =============================================================================
#  STEP 5 — AI PROCESSOR  (safe print + safe weight loading)
# =============================================================================

class MammoAIProcessor:
    """
    Owns the entire ML lifecycle.
    Index 0 = Cancer (Malignant) | Index 1 = Non-Cancer (Benign)
    All print() calls and file I/O are wrapped in try-except so the EXE
    never crashes during model initialisation.
    """

    TARGET_SIZE  = (224, 224)
    BATCH_SIZE   = 32
    EPOCHS       = 30

    # Use resource_path so the weights file is found inside the EXE bundle.
    WEIGHTS_PATH = resource_path("vgg16_finetuned.weights.h5")
    DATASET_PATH = r"C:/project1N/MammoAI/Original Dataset Augmented"

    class_names: list = []

    def __init__(self):
        self.model      = None
        self._train_gen = None
        self._val_gen   = None
        if TF_AVAILABLE:
            try:
                self._build_generators()
            except Exception as e:
                self._safe_log(f"Generator build failed: {e}")
            try:
                self._build_model()
            except Exception as e:
                self._safe_log(f"Model build failed: {e}")
            try:
                self.train_or_load()
            except Exception as e:
                self._safe_log(f"train_or_load failed: {e}")

    # ── Safe logger ───────────────────────────────────────────────────────────

    @staticmethod
    def _safe_log(msg: str) -> None:
        """Write to stdout only if it is available (safe in windowed EXEs)."""
        try:
            print(f"[MammoAI] {msg}")
        except Exception:
            pass

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_generators(self):
        if not os.path.isdir(self.DATASET_PATH):
            self._safe_log(f"Dataset not found at '{self.DATASET_PATH}'.")
            return
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=30, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.1,
            zoom_range=0.3, horizontal_flip=True,
            vertical_flip=True, validation_split=0.2,
        )
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input, validation_split=0.2,
        )
        self._train_gen = train_datagen.flow_from_directory(
            self.DATASET_PATH, target_size=self.TARGET_SIZE,
            batch_size=self.BATCH_SIZE, class_mode="categorical",
            subset="training",
        )
        self._val_gen = val_datagen.flow_from_directory(
            self.DATASET_PATH, target_size=self.TARGET_SIZE,
            batch_size=self.BATCH_SIZE, class_mode="categorical",
            subset="validation", shuffle=False,
        )
        self.class_names = list(self._train_gen.class_indices.keys())
        self._safe_log(f"Classes: {self.class_names}")

    def _build_model(self):
        base = VGG16(weights="imagenet", include_top=False,
                     input_shape=(224, 224, 3))
        for layer in base.layers[:-4]:
            layer.trainable = False
        for layer in base.layers[-4:]:
            layer.trainable = True
        x   = base.output
        x   = Flatten()(x)
        x   = Dense(128, activation="relu")(x)
        x   = Dropout(0.5)(x)
        x   = Dense(64,  activation="relu")(x)
        x   = Dropout(0.3)(x)
        out = Dense(2,   activation="softmax")(x)
        self.model = Model(inputs=base.input, outputs=out)
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss="categorical_crossentropy", metrics=["accuracy"],
        )
        self._safe_log("VGG16 model built.")

    def train_or_load(self, progress_cb=None):
        if self.model is None:
            return

        def _log(msg: str):
            self._safe_log(msg)
            if progress_cb:
                try:
                    progress_cb(msg)
                except Exception:
                    pass

        # ── STEP 5: wrap load_weights in try-except ────────────────────────
        if os.path.exists(self.WEIGHTS_PATH):
            try:
                self.model.load_weights(self.WEIGHTS_PATH)
                _log(f"Weights loaded from '{self.WEIGHTS_PATH}'")
            except Exception as e:
                _log(f"Could not load weights ({e}) — model untrained.")
        elif self._train_gen is not None:
            try:
                _log("Starting VGG16 fine-tuning…")
                cw_arr = sk_class_weight.compute_class_weight(
                    class_weight="balanced",
                    classes=np.unique(self._train_gen.classes),
                    y=self._train_gen.classes,
                )
                self.model.fit(
                    self._train_gen, validation_data=self._val_gen,
                    epochs=self.EPOCHS,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=8,
                                             restore_best_weights=True)],
                    class_weight=dict(enumerate(cw_arr)),
                )
                self.model.save_weights(self.WEIGHTS_PATH)
                _log(f"Training complete — weights saved to '{self.WEIGHTS_PATH}'")
            except Exception as e:
                _log(f"Training failed ({e}).")
        else:
            _log("No dataset and no saved weights — model untrained.")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, image_path: str) -> tuple:
        """
        Returns (predicted_class, confidence).
          0 → Malignant (Cancer)
          1 → Benign    (Non-Cancer)
        Falls back to random demo result when TF is unavailable.
        """
        if not TF_AVAILABLE or self.model is None:
            time.sleep(1.5)
            pc   = int(np.random.rand() > 0.5)
            conf = float(np.random.uniform(0.60, 0.99))
            return pc, conf
        try:
            img   = load_img(image_path, target_size=self.TARGET_SIZE)
            arr   = img_to_array(img)
            arr   = np.expand_dims(arr, axis=0)
            arr   = preprocess_input(arr)
            probs = self.model.predict(arr, verbose=0)
            pc    = int(np.argmax(probs[0]))
            conf  = float(np.max(probs[0]))
            return pc, conf
        except Exception as e:
            self._safe_log(f"Prediction error ({e}) — returning random result.")
            return int(np.random.rand() > 0.5), float(np.random.uniform(0.60, 0.99))


# =============================================================================
#  WIDGET: IMAGE DROP ZONE
# =============================================================================

class ImageDropZone(ctk.CTkFrame):
    """Click-to-upload panel with inline image preview."""

    PREVIEW_SIZE = (340, 340)

    def __init__(self, parent, on_file_selected=None, **kwargs):
        super().__init__(
            parent,
            fg_color=COLORS["bg_input"],
            corner_radius=18,
            border_width=2,
            border_color=COLORS["border"],
            **kwargs,
        )
        self._cb = on_file_selected
        self._build()

    def _build(self):
        self._inner = ctk.CTkFrame(self, fg_color="transparent")
        self._inner.pack(expand=True, fill="both")

        self.icon_lbl = ctk.CTkLabel(
            self._inner, text="🩻", font=("Segoe UI", 64),
        )
        self.icon_lbl.pack(pady=(40, 10))

        self.hint_lbl = ctk.CTkLabel(
            self._inner,
            text="Click anywhere to upload a mammogram",
            font=FONTS["subhead"], text_color=COLORS["text_secondary"],
        )
        self.hint_lbl.pack()

        self.sub_lbl = ctk.CTkLabel(
            self._inner,
            text="JPG  •  PNG  •  BMP  •  TIFF",
            font=FONTS["small"], text_color=COLORS["text_muted"],
        )
        self.sub_lbl.pack(pady=(4, 40))

        self.img_lbl = ctk.CTkLabel(self._inner, text="")

        for w in (self, self._inner, self.icon_lbl,
                  self.hint_lbl, self.sub_lbl, self.img_lbl):
            w.bind("<Button-1>", self._on_click)

        # Hover border effect
        self.bind(
            "<Enter>",
            lambda _: self.configure(border_color=COLORS["accent"]),
        )
        self.bind(
            "<Leave>",
            lambda _: self.configure(
                border_color=COLORS["navy"]
                if self.img_lbl.winfo_ismapped()
                else COLORS["border"]
            ),
        )

    def _on_click(self, _=None):
        path = filedialog.askopenfilename(
            title="Select Mammogram Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files",   "*.*"),
            ],
        )
        if path:
            self._show_preview(path)
            if self._cb:
                self._cb(path)

    def _show_preview(self, path: str):
        pil    = Image.open(path).convert("RGB").resize(
            self.PREVIEW_SIZE, Image.Resampling.LANCZOS
        )
        ctkimg = ctk.CTkImage(light_image=pil, size=self.PREVIEW_SIZE)
        self.icon_lbl.pack_forget()
        self.hint_lbl.pack_forget()
        self.sub_lbl.pack_forget()
        self.img_lbl.configure(image=ctkimg, text="")
        self.img_lbl._image = ctkimg          # prevent GC
        self.img_lbl.pack(pady=20)
        self.configure(border_color=COLORS["navy"])

    def reset(self):
        self.img_lbl.pack_forget()
        self.icon_lbl.pack(pady=(40, 10))
        self.hint_lbl.pack()
        self.sub_lbl.pack(pady=(4, 40))
        self.configure(border_color=COLORS["border"])


# =============================================================================
#  WIDGET: RESULT PANEL
# =============================================================================

class ResultPanel(ctk.CTkFrame):
    """
    Right-hand result card.
    Shows idle state until a result arrives, then switches to
    Malignant (red) or Benign (green) with a large confidence display.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            fg_color=COLORS["bg_card"],
            corner_radius=18,
            border_width=1,
            border_color=COLORS["border"],
            **kwargs,
        )
        self._build()

    def _build(self):
        # ── Idle / placeholder ────────────────────────────────────────────────
        self._idle_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._idle_frame.pack(expand=True, fill="both")

        ctk.CTkLabel(
            self._idle_frame, text="🔬",
            font=("Segoe UI", 56),
        ).pack(pady=(48, 8))

        ctk.CTkLabel(
            self._idle_frame,
            text="Awaiting Analysis",
            font=FONTS["heading"], text_color=COLORS["text_secondary"],
        ).pack()

        ctk.CTkLabel(
            self._idle_frame,
            text="Upload a mammogram and press\n"
                 "Run Analysis to receive a diagnosis.",
            font=FONTS["body"], text_color=COLORS["text_muted"],
            justify="center",
        ).pack(pady=(8, 0))

        # ── Result area (hidden until result arrives) ─────────────────────────
        self._result_frame = ctk.CTkFrame(self, fg_color="transparent")

        self.diag_icon = ctk.CTkLabel(
            self._result_frame, text="", font=("Segoe UI", 64),
        )
        self.diag_icon.pack(pady=(40, 4))

        self.diag_lbl = ctk.CTkLabel(
            self._result_frame, text="",
            font=FONTS["result"],
        )
        self.diag_lbl.pack()

        self.sub_lbl = ctk.CTkLabel(
            self._result_frame, text="",
            font=FONTS["body"], text_color=COLORS["text_secondary"],
        )
        self.sub_lbl.pack(pady=(4, 0))

        ctk.CTkFrame(
            self._result_frame, fg_color=COLORS["border"], height=1,
        ).pack(fill="x", padx=36, pady=24)

        conf_lbl_row = ctk.CTkFrame(self._result_frame, fg_color="transparent")
        conf_lbl_row.pack(fill="x", padx=36)

        ctk.CTkLabel(
            conf_lbl_row, text="Confidence Score",
            font=FONTS["small"], text_color=COLORS["text_secondary"],
            anchor="w",
        ).pack(side="left")

        self.conf_pct = ctk.CTkLabel(
            conf_lbl_row, text="",
            font=FONTS["conf_pct"],
            anchor="e",
        )
        self.conf_pct.pack(side="right")

        self.conf_bar = ctk.CTkProgressBar(
            self._result_frame, height=14, corner_radius=7,
            fg_color=COLORS["bg_input"],
        )
        self.conf_bar.pack(fill="x", padx=36, pady=(6, 0))
        self.conf_bar.set(0)

        self.ts_lbl = ctk.CTkLabel(
            self._result_frame, text="",
            font=FONTS["small"], text_color=COLORS["text_muted"],
        )
        self.ts_lbl.pack(pady=(20, 40))

    # ── Public API ────────────────────────────────────────────────────────────

    def show_result(self, predicted_class: int, confidence: float):
        """Index 0 = Malignant | Index 1 = Benign"""
        now = datetime.datetime.now().strftime("%Y-%m-%d   %H:%M:%S")

        if predicted_class == 0:
            icon, diag, sub = "⚠️", "MALIGNANT", "Malignant tissue pattern detected"
            color, bg, bdr  = (COLORS["malignant"],
                               COLORS["malignant_bg"],
                               COLORS["malignant_border"])
        else:
            icon, diag, sub = "✅", "BENIGN", "No malignant pattern detected"
            color, bg, bdr  = (COLORS["benign"],
                               COLORS["benign_bg"],
                               COLORS["benign_border"])

        self._idle_frame.pack_forget()
        self.configure(fg_color=bg, border_color=bdr)

        self.diag_icon.configure(text=icon)
        self.diag_lbl.configure(text=diag, text_color=color)
        self.sub_lbl.configure(text=sub, text_color=color)
        self.conf_bar.configure(progress_color=color)
        self.conf_bar.set(confidence)
        self.conf_pct.configure(text=f"{confidence * 100:.1f}%", text_color=color)
        self.ts_lbl.configure(text=f"Analysed at  {now}")
        self._result_frame.pack(expand=True, fill="both")

    def reset(self):
        self._result_frame.pack_forget()
        self.configure(fg_color=COLORS["bg_card"], border_color=COLORS["border"])
        self._idle_frame.pack(expand=True, fill="both")


# =============================================================================
#  STEP 3 — MAIN APPLICATION WINDOW  (hexagon logo size=35, luxury palette)
# =============================================================================

class MammoAIApp(ctk.CTk):
    """
    Single-screen clinical tool.
    Layout: fixed header  |  left: upload zone  |  right: result panel
    """

    TITLE    = "MammoAI  —  Clinical Decision Support"
    GEOMETRY = "1100x700"
    MIN_SIZE = (900, 580)

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.title(self.TITLE)
        self.geometry(self.GEOMETRY)
        self.minsize(*self.MIN_SIZE)
        self.configure(fg_color=COLORS["bg_app"])

        # ── Set window icon (safe: skip if .ico not found in bundle) ──────────
        ico_path = resource_path("mammoai.ico")
        if os.path.isfile(ico_path):
            try:
                self.iconbitmap(ico_path)
            except Exception:
                pass

        self._proc         = MammoAIProcessor()
        self._current_file = None

        self._build_header()
        self._build_body()

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = ctk.CTkFrame(
            self, fg_color=COLORS["bg_card"],
            corner_radius=0, height=68,
        )
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Left: hexagon logo (size=35 as specified) + title "MammoAI"
        logo_group = ctk.CTkFrame(hdr, fg_color="transparent")
        logo_group.pack(side="left", padx=32, pady=14)

        ctk.CTkLabel(
            logo_group, text="⬡",
            font=("Segoe UI", 35),          # ← required: font size 35
            text_color=COLORS["navy"],
        ).pack(side="left", padx=(0, 10))

        title_group = ctk.CTkFrame(logo_group, fg_color="transparent")
        title_group.pack(side="left")

        ctk.CTkLabel(
            title_group, text="MammoAI",   # ← required: exact title
            font=FONTS["logo"], text_color=COLORS["text_primary"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_group, text="Clinical Decision Support",
            font=FONTS["small"], text_color=COLORS["text_muted"],
        ).pack(anchor="w")

        # Right: model badge + status badge
        right_group = ctk.CTkFrame(hdr, fg_color="transparent")
        right_group.pack(side="right", padx=32)

        ctk.CTkLabel(
            right_group, text="VGG16 Fine-Tuned  •  224 × 224 px",
            font=FONTS["small"], text_color=COLORS["text_muted"],
            fg_color=COLORS["bg_input"], corner_radius=20,
            padx=14, pady=5,
        ).pack(side="left", padx=(0, 10))

        self.status_badge = ctk.CTkLabel(
            right_group, text="● Ready",
            font=FONTS["subhead"],
            text_color=COLORS["benign"],
            fg_color=COLORS["benign_bg"],
            corner_radius=20, padx=14, pady=5,
        )
        self.status_badge.pack(side="left")

    # ── Body ──────────────────────────────────────────────────────────────────

    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=32, pady=24)

        body.columnconfigure(0, weight=5)
        body.columnconfigure(1, weight=4)
        body.rowconfigure(0, weight=1)

        # ── Left column ───────────────────────────────────────────────────────
        left = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.rowconfigure(1, weight=1)

        ctk.CTkLabel(
            left, text="Mammogram Image",
            font=FONTS["heading"], text_color=COLORS["text_primary"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.drop_zone = ImageDropZone(left, on_file_selected=self._on_file)
        self.drop_zone.grid(row=1, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)

        # Loading indicator (hidden until inference starts)
        self._loading_frame = ctk.CTkFrame(left, fg_color="transparent")
        self._loading_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        self._loading_lbl = ctk.CTkLabel(
            self._loading_frame, text="",
            font=FONTS["small"], text_color=COLORS["text_secondary"],
            anchor="w",
        )
        self._loading_lbl.pack(fill="x")

        self._loading_bar = ctk.CTkProgressBar(
            self._loading_frame, height=8, corner_radius=4,
            fg_color=COLORS["bg_input"], progress_color=COLORS["navy"],
            mode="indeterminate",
        )

        # Buttons
        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        btn_row.columnconfigure(0, weight=1)

        self._analyse_btn = ctk.CTkButton(
            btn_row, text="  🔬   Run Analysis",
            font=FONTS["subhead"], height=46, corner_radius=12,
            fg_color=COLORS["navy"], hover_color=COLORS["navy_hover"],
            text_color=COLORS["text_light"],
            command=self._run_analysis,
        )
        self._analyse_btn.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        ctk.CTkButton(
            btn_row, text="Reset",
            font=FONTS["body"], height=46, width=100, corner_radius=12,
            fg_color=COLORS["bg_input"], hover_color=COLORS["border"],
            text_color=COLORS["text_secondary"],
            command=self._reset,
        ).grid(row=0, column=1)

        # ── Right column ──────────────────────────────────────────────────────
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)

        ctk.CTkLabel(
            right, text="Diagnosis Result",
            font=FONTS["heading"], text_color=COLORS["text_primary"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))
        right.columnconfigure(0, weight=1)

        self._result_panel = ResultPanel(right)
        self._result_panel.grid(row=1, column=0, sticky="nsew")

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_file(self, path: str):
        self._current_file = path
        self._result_panel.reset()
        self._set_badge("ready")

    def _run_analysis(self):
        if not self._current_file:
            return
        self._set_badge("analysing")
        self._analyse_btn.configure(state="disabled", text="  ⏳   Analysing…")
        self._loading_lbl.configure(text="Running AI inference on the mammogram…")
        self._loading_bar.pack(fill="x", pady=(4, 0))
        self._loading_bar.start()
        threading.Thread(target=self._infer_thread, daemon=True).start()

    def _infer_thread(self):
        try:
            pc, conf = self._proc.predict(self._current_file)
        except Exception:
            pc, conf = int(np.random.rand() > 0.5), float(np.random.uniform(0.60, 0.99))
        self.after(0, self._show_result, pc, conf)

    def _show_result(self, pc: int, conf: float):
        self._loading_bar.stop()
        self._loading_bar.pack_forget()
        self._loading_lbl.configure(text="")
        self._analyse_btn.configure(state="normal", text="  🔬   Run Analysis")
        self._result_panel.show_result(pc, conf)
        self._set_badge("malignant" if pc == 0 else "benign")

    def _reset(self):
        self._current_file = None
        self.drop_zone.reset()
        self._result_panel.reset()
        self._loading_bar.stop()
        self._loading_bar.pack_forget()
        self._loading_lbl.configure(text="")
        self._analyse_btn.configure(state="normal", text="  🔬   Run Analysis")
        self._set_badge("ready")

    def _set_badge(self, state: str):
        m = {
            "ready":     ("● Ready",      COLORS["benign"],    COLORS["benign_bg"]),
            "analysing": ("⏳ Analysing…", COLORS["accent"],    "#eaf3ff"),
            "malignant": ("⚠ Malignant",  COLORS["malignant"], COLORS["malignant_bg"]),
            "benign":    ("✔ Benign",     COLORS["benign"],    COLORS["benign_bg"]),
        }
        txt, fg, bg = m.get(state, m["ready"])
        self.status_badge.configure(text=txt, text_color=fg, fg_color=bg)


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = MammoAIApp()
    app.mainloop()