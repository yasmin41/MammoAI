import sys
import os

# =============================================================================
# PYINSTALLER FIXES & HELPERS
# =============================================================================
if getattr(sys, 'frozen', False):
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def resource_path(relative_path):
    """Get absolute path to a resource, works for development and PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
from sklearn.utils import class_weight
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import ImageTk, Image
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score
    )
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator,
        load_img,
        img_to_array
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError as e:
    error_message = (
        "An essential library failed to import. Please ensure all libraries are installed correctly.\n\n"
        f"Error: {e}\n\n"
        "Inside your activated virtual environment (tf_env), please run:\n"
        "pip install numpy<2 scikit-learn pillow seaborn matplotlib tensorflow==2.10 tensorflow-io==0.27.0 h5py==3.11.0"
    )
    print(error_message)
    exit()

# =============================================================================
# DATASET PATH
# =============================================================================
dataset_path = r"C:/project1N/MammoAI/Original Dataset Augmented"

if not os.path.exists(dataset_path):
    print(f"Error: The dataset path does not exist: {dataset_path}")
    exit()


# =============================================================================
# MODEL BUILDING, TRAINING, AND LOADING
# =============================================================================
# Data generators with augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# Generators
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

validation_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# Handle class imbalance
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

# Base model (VGG16)
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze all but last 4 layers
for layer in base_model.layers[:-4]:
    layer.trainable = False
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Custom classification head
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(train_generator.num_classes, activation="softmax")(x)

# Build model
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

# Train or load weights
weights_path = "vgg16_finetuned.weights.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
else:
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=30,
        callbacks=[early_stop],
        class_weight=class_weights_dict
    )
    model.save_weights(weights_path)

# Class labels
class_labels = list(train_generator.class_indices.keys())


# =============================================================================
# MAIN APPLICATION & GUI FUNCTIONS
# =============================================================================
def preprocess_and_predict(image_path):
    """Loads, preprocesses, and predicts the class of a single image."""
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class_index]
    predicted_class_name = class_labels[predicted_class_index]

    return predicted_class_name, confidence


def upload_image():
    """Handles the image upload, prediction, and result display."""
    file_path = filedialog.askopenfilename(parent=window)
    if not file_path:
        return

    # Show uploaded image
    img = Image.open(file_path).resize((250, 250), Image.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    panel.configure(image=photo)
    panel.image = photo

    # Predict
    predicted_class_name, confidence = preprocess_and_predict(file_path)

    if predicted_class_name.lower() == "cancer":
        result_text = (
            f"🔬 RESULT: CANCER DETECTED 🔬\n\nConfidence: {confidence * 100:.2f}%"
        )
        result_label.config(text=result_text, fg="red")
    else:
        result_text = (
            f"🔬 RESULT: NON-CANCEROUS 🔬\n\nConfidence: {confidence * 100:.2f}%"
        )
        result_label.config(text=result_text, fg="green")


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Create main window
    window = tk.Tk()
    window.title("MammoAI - Breast Cancer Detection")
    window.iconbitmap(resource_path("icon.ico"))
    window.geometry("500x500")

    # --- Widgets ---
    label = Label(
        window,
        text="Upload a Mammogram Image",
        font=("Arial", 16, "bold")
    )
    label.pack(pady=(10, 5))

    upload_button = Button(
        window,
        text="Upload Image",
        command=upload_image,
        font=("Arial", 12),
        bg="#4CAF50",
        fg="white",
        padx=20,
        pady=5
    )
    upload_button.pack(pady=10)

    panel = Label(window)
    panel.pack(pady=10)

    result_label = Label(
        window,
        text="",
        font=("Arial", 14, "bold"),
        justify="center"
    )
    result_label.pack(pady=(10, 20))

    # Run app
    window.mainloop()
