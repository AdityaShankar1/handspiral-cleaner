import cv2
import os
import numpy as np

# Windows file paths (use raw string format r"...")
input_folder = r"C:\Users\shank\Downloads\PD (1)\PD\VOICE&TYPE&HANDWRITING\HANDWRITING\Improved Spiral Test Using Digitized Graphics Tablet for Monitoring Parkinson's Disease\drawings\Dynamic Spiral Test"
output_folder = r"C:\Users\shank\Downloads\PD (1)\PD\VOICE&TYPE&HANDWRITING\HANDWRITING\processed_spiral_test"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Pre-processing function
def preprocess_image(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if image is None:
        print(f"Skipping {image_path}, unable to read.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding for binarization
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize image to fixed size (128x128)
    resized = cv2.resize(binary, (128, 128))

    # Save the processed image
    cv2.imwrite(output_path, resized)

# Process all images in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        preprocess_image(input_path, output_path)
        print(f"Processed: {filename}")

print("✅ Pre-processing complete. Check the 'processed_spiral_test' folder.")
