import os
import numpy as np
import pandas as pd
from skimage import io, color, img_as_ubyte
import math

# Define angles and distance here for easy modification
ANGLES = [0, 45, 90, 135]  # in degrees
DISTANCE = 1  # pixel distance

def calculate_glcm(image, mask, angle, distance):
    height, width = image.shape
    glcm = np.zeros((257, 257), dtype=float)
    pixel_counter = 0

    angle_rad = np.radians(angle)
    dx = int(round(distance * np.cos(angle_rad)))
    dy = int(round(distance * np.sin(angle_rad)))

    for y in range(height):
        for x in range(width):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if mask[y, x] and mask[ny, nx]:
                    i, j = image[y, x], image[ny, nx]
                    glcm[i, j] += 1
                    glcm[j, i] += 1
                    pixel_counter += 2

    if pixel_counter > 0:
        glcm /= pixel_counter
    return glcm

def calculate_glcm_features(glcm):
    asm = np.sum(glcm**2)
    
    contrast = 0
    idm = 0
    entropy = 0
    for i in range(257):
        for j in range(257):
            contrast += (i-j)**2 * glcm[i,j]
            idm += glcm[i,j] / (1 + (i-j)**2)
            if glcm[i,j] > 0:
                entropy -= glcm[i,j] * math.log(glcm[i,j])

    # Correlation calculation
    px = py = 0
    for i in range(257):
        for j in range(257):
            px += i * glcm[i][j]
            py += j * glcm[i][j]

    stdevx = stdevy = 0
    for i in range(257):
        for j in range(257):
            stdevx += (i - px) * (i - px) * glcm[i][j]
            stdevy += (j - py) * (j - py) * glcm[i][j]
    
    stdevx = np.sqrt(stdevx)
    stdevy = np.sqrt(stdevy)

    correlation = 0
    for i in range(257):
        for j in range(257):
            correlation += ((i - px) * (j - py) * glcm[i][j] / (stdevx * stdevy))

    return {
        'Angular Second Moment': asm,
        'Contrast': contrast,
        'Correlation': correlation,
        'Inverse Difference Moment': idm,
        'Entropy': entropy
    }

def process_image(image_path, mask_path):
    # Load the image and mask
    image = io.imread(image_path)
    mask = io.imread(mask_path)

    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    # Ensure the image is 8-bit
    image = img_as_ubyte(image)

    # Calculate GLCM for all specified angles and average the features
    features_sum = {
        'Angular Second Moment': 0,
        'Contrast': 0,
        'Correlation': 0,
        'Inverse Difference Moment': 0,
        'Entropy': 0
    }

    for angle in ANGLES:
        glcm = calculate_glcm(image, mask, angle, DISTANCE)
        features = calculate_glcm_features(glcm)
        for key in features_sum:
            features_sum[key] += features[key]

    # Average the features
    for key in features_sum:
        features_sum[key] /= len(ANGLES)

    return features_sum

def process_folder(image_folder, mask_folder, output_file):
    results = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.tif')):  # Add or remove file extensions as needed
            image_path = os.path.join(image_folder, filename)
            mask_path = os.path.join(mask_folder, filename)
            
            if os.path.exists(mask_path):
                print(f"Processing {filename}")
                result = process_image(image_path, mask_path)
                result['Filename'] = filename
                results.append(result)
            else:
                print(f"Mask not found for {filename}")

    # Create a DataFrame and save to Excel
    df = pd.DataFrame(results)
    df = df[['Filename', 'Angular Second Moment', 'Contrast', 'Correlation', 'Inverse Difference Moment', 'Entropy']]
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

# Usage
image_folder = "your_image_folder_path"
mask_folder = "your_mask_folder_path"
output_file = ".results.xlsx"



process_folder(image_folder, mask_folder, output_file)
