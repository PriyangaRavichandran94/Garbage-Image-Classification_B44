import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ===============================
# Dataset Path
# ===============================
DATA_DIR = "data/raw"

# Get class names
classes = sorted([d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d))])

# ===========================================
# 1. Number of Images Per Class
# ===========================================

image_counts = []

for cls in classes:
    class_path = os.path.join(DATA_DIR, cls)

    count = len([
        img for img in os.listdir(class_path)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    image_counts.append(count)

print("\nNumber of Images Per Class\n")

for cls, count in zip(classes, image_counts):
    print(f"{cls}: {count}")

plt.figure(figsize=(8,5))
plt.bar(classes, image_counts)
plt.title("Number of Images per Class")
plt.xlabel("Garbage Category")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===========================================
# 2. Display Sample Images
# ===========================================
# ===========================================
# 2. Display Sample Images
# ===========================================

num_classes = len(classes)

cols = 5                      # 5 images in each row
rows = math.ceil(num_classes / cols)

plt.figure(figsize=(20, 8))

for i, cls in enumerate(classes):

    class_path = os.path.join(DATA_DIR, cls)

    image_files = [
        img for img in os.listdir(class_path)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        continue

    image_path = os.path.join(class_path, image_files[0])

    img = Image.open(image_path)

    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.suptitle("Sample Images from Each Garbage Category", fontsize=18)

plt.tight_layout()

plt.show()


print("\nEDA Completed Successfully!")