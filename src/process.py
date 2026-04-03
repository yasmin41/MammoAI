import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -----------------------------------------------------------
# Function to compute dataset mean and std with caching
# -----------------------------------------------------------

def compute_dataset_mean_std(dataset_path, target_size=(224, 224), cache_file='mean_std_cache.npz'):
    # Check if cached mean/std exist
    if os.path.exists(cache_file):
        print(f"✔ Loaded cached mean and std from '{cache_file}'")
        data = np.load(cache_file)
        print("Mean:", data['mean'])
        print("Standard Deviation:", data['std'])
        return data['mean'], data['std']

    print("🔄 Calculating mean and standard deviation from dataset images...")

    pixel_values = []

    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            try:
                # Load image in RGB mode
                img = load_img(img_path, target_size=target_size, color_mode='rgb')
                img_array = img_to_array(img)

                # Ensure the image has 3 channels (RGB)
                if img_array.shape[2] != 3:
                    print(f"⚠️ Skipping {img_path}: Not an RGB image.")
                    continue

                pixel_values.append(img_array)

            except Exception as e:
                print(f"⚠️ Error loading {img_path}: {e}")

    if not pixel_values:
        raise ValueError("❌ No valid RGB images found in the dataset.")

    pixel_values = np.array(pixel_values, dtype=np.float32)
    mean = np.mean(pixel_values, axis=(0, 1, 2))
    std = np.std(pixel_values, axis=(0, 1, 2))

    # Save the values to cache
    np.savez(cache_file, mean=mean, std=std)
    print("✔ Mean and std saved to cache.")

    return mean, std
