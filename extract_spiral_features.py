import cv2
import os
import numpy as np
import pandas as pd

# Paths to the folders
base_path = r"C:\Users\shank\Downloads\PD (1)\PD\VOICE&TYPE&HANDWRITING\HANDWRITING\Improved Spiral Test Using Digitized Graphics Tablet for Monitoring Parkinson's Disease\drawings"
folders = ["Dynamic Spiral Test", "Dynamic Spiral Test"]

# 📌 Feature extraction function
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"❌ Error loading {image_path}")
        return None

    # Apply thresholding to get binary image
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"❌ No contour found in {image_path}")
        return None

    # Take the largest contour (spiral)
    contour = max(contours, key=cv2.contourArea)

    # 1️**Contour Area**
    area = cv2.contourArea(contour)

    # 2️**Perimeter (Arc Length)**
    perimeter = cv2.arcLength(contour, closed=True)

    # 3️**Bounding Box (Width & Height)**
    x, y, w, h = cv2.boundingRect(contour)

    # 4️**Circularity (Closer to 1 = more circular)**
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # 5️**Hu Moments (Shape Descriptors)**
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Convert Hu Moments to log scale for better range
    hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    # Return extracted features
    return [image_path, area, perimeter, w, h, circularity] + hu_moments_log.tolist()

# 📌 Process images & store results
data = []
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        continue

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            features = extract_features(image_path)
            if features:
                data.append([folder] + features)

# Save results to CSV
columns = ["Type", "Image_Path", "Area", "Perimeter", "Width", "Height", "Circularity"] + \
          [f"Hu_Moment_{i}" for i in range(7)]

df = pd.DataFrame(data, columns=columns)
output_csv = os.path.join(base_path, "spiral_features.csv")
df.to_csv(output_csv, index=False)

print(f"✅ Feature extraction complete. Results saved to {output_csv}")
