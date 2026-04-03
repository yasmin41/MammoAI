import os
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa

# --------------------------
# 1. Augmentation Setup
# --------------------------
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Rotate((-20, 20)),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
    iaa.Dropout(p=(0, 0.1)),
    iaa.Resize({"height": (0.8, 1.2), "width": (0.8, 1.2)}),
    iaa.Crop(percent=(0, 0.1)),
    iaa.ElasticTransformation(alpha=(0, 5.0), sigma=1.0),
    iaa.PiecewiseAffine(scale=(0.02, 0.05)),
    iaa.PerspectiveTransform(scale=(0.03, 0.08)),
    iaa.LinearContrast((0.8, 1.5)),
    iaa.Multiply((0.8, 1.2), per_channel=0.5),
], random_order=True)

# --------------------------
# 2. Number of Augmented Images
# --------------------------
AUGMENT_PER_IMAGE = 4

# --------------------------
# 3. Paths
# --------------------------
original_dataset_dir = r"C:\project1N\MammoAI\Original Dataset"
augmented_dataset_dir = r"C:\project1N\MammoAI\Original Dataset Augmented"
os.makedirs(augmented_dataset_dir, exist_ok=True)

# --------------------------
# 4. Categories to process
# --------------------------
categories = ['Cancer']

for category in categories:
    original_category_path = os.path.join(original_dataset_dir, category)
    augmented_category_path = os.path.join(augmented_dataset_dir, category)

    if not os.path.exists(original_category_path):
        print(f"⚠️ Warning: Original category folder '{category}' does not exist. Skipping this category.")
        continue

    os.makedirs(augmented_category_path, exist_ok=True)

    for image_name in os.listdir(original_category_path):
        image_path = os.path.join(original_category_path, image_name)

        try:
         
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)

            base_name = os.path.splitext(image_name)[0]

       
            orig_save_path = os.path.join(augmented_category_path, f"{base_name}_orig.jpg")
            image.save(orig_save_path)

           
            for i in range(AUGMENT_PER_IMAGE):
                augmented_img = seq(image=image_np)
                aug_filename = f"{base_name}_aug{i}.jpg"
                aug_path = os.path.join(augmented_category_path, aug_filename)
                Image.fromarray(augmented_img).save(aug_path)

            print(f"✅ Done: {image_name}")

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")

print("🎉 Augmentation completed!")