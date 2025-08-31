import os
import numpy as np
import pandas as pd
from skimage import io, color, img_as_ubyte

def calculate_lbp(image, mask):
    """
    Calculate LBP following Ojala's original method:
    1. 3x3 window
    2. Compare center with 8 neighbors
    3. Generate binary pattern
    4. Weight by powers of 2
    5. Sum to get final pattern value
    """
    rows, cols = image.shape
    lbp_image = np.zeros_like(image)
    
    # Define the 8 neighbors in a circular pattern (clockwise from top-left)
    # Following standard 3x3 grid
    neighbors = [
        (-1,-1), (-1,0), (-1,1),    # Top row
        (0,1),           (1,1),      # Right side
        (1,0), (1,-1),   (0,-1)      # Bottom and left
    ]
    
    # Powers of 2 for each position (corresponding to neighbors)
    weights = [1, 2, 4, 8, 16, 32, 64, 128]
    
    for y in range(1, rows-1):
        for x in range(1, cols-1):
            if not mask[y, x]:
                continue
            
            center_value = image[y, x]
            pattern = 0
            
            # Compare with each neighbor
            for i, (dy, dx) in enumerate(neighbors):
                if mask[y+dy, x+dx]:  # Only consider pixels within mask
                    # If neighbor >= center, set bit to 1
                    if image[y+dy, x+dx] >= center_value:
                        pattern += weights[i]
            
            lbp_image[y, x] = pattern
            
    return lbp_image

def calculate_texture_features(lbp_image, mask):
    """
    Calculate energy and entropy from LBP image as mentioned in paper.
    Only considers pixels within the ROI (mask).
    """
    # Extract LBP values only within the mask
    valid_lbps = lbp_image[mask > 0]
    
    if len(valid_lbps) == 0:
        return {
            'LBP_Energy': 0,
            'LBP_Entropy': 0,
            'ROI_Size': 0
        }
    
    # Calculate histogram (256 possible patterns)
    hist, _ = np.histogram(valid_lbps, bins=np.arange(257), density=True)
    
    # Calculate energy (sum of squared probabilities)
    energy = np.sum(hist * hist)
    
    # Calculate entropy (-sum(p * log2(p)))
    # Avoid log(0) by only considering non-zero probabilities
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    
    return {
        'LBP_Energy': energy,
        'LBP_Entropy': entropy,
        'ROI_Size': len(valid_lbps)
    }

def process_image(image_path, mask_path):
    """Process a single image and its mask."""
    try:
        # Load image and mask
        image = io.imread(image_path)
        mask = io.imread(mask_path)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = color.rgb2gray(image)
        
        # Convert to 8-bit
        image = img_as_ubyte(image)
        
        # Ensure mask is binary
        mask = mask > 0
        
        # Calculate LBP
        lbp_image = calculate_lbp(image, mask)
        
        # Calculate texture features
        features = calculate_texture_features(lbp_image, mask)
        
        return features
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_folder(image_folder, mask_folder, output_file):
    """Process all images in the specified folders."""
    results = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.tif', '.tiff')):
            image_path = os.path.join(image_folder, filename)
            mask_path = os.path.join(mask_folder, filename)
            
            if os.path.exists(mask_path):
                print(f"Processing {filename}")
                result = process_image(image_path, mask_path)
                if result is not None:
                    result['Filename'] = filename
                    results.append(result)
            else:
                print(f"Mask not found for {filename}")
    
    if results:
        # Create DataFrame and save results
        df = pd.DataFrame(results)
        cols = ['Filename'] + [col for col in df.columns if col != 'Filename']
        df = df[cols]
        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No results to save")

# Paths for processing
image_folder = "your_image_folder_path"
mask_folder = "your_mask_folder_path"
output_file = ".results.xlsx"

process_folder(image_folder, mask_folder, output_file)