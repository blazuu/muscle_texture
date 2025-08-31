import os
import numpy as np
import pandas as pd
from skimage import io, color, img_as_ubyte
from skimage.measure import regionprops

def crop_to_roi(image, mask):
    """Crop image and mask to the bounding box of the ROI."""
    # Get properties of the mask
    props = regionprops(mask.astype(int))
    if not props:
        return None, None
    
    # Get bounding box coordinates
    minr, minc, maxr, maxc = props[0].bbox
    
    # Crop both image and mask
    cropped_image = image[minr:maxr, minc:maxc]
    cropped_mask = mask[minr:maxr, minc:maxc]
    
    return cropped_image, cropped_mask

def calculate_rlm(image, mask, angle):
    """Calculate run-length matrix only for pixels within the ROI."""
    rows, cols = image.shape
    max_length = max(rows, cols)
    rlm = np.zeros((256, max_length), dtype=int)
    
    if angle == 0:  # Horizontal
        for i in range(rows):
            run_length = 1
            run_value = None
            for j in range(cols):
                if not mask[i,j]:
                    if run_value is not None:
                        rlm[run_value, run_length-1] += 1
                    run_value = None
                    continue
                    
                current_value = image[i,j]
                if run_value is None:
                    run_value = current_value
                    run_length = 1
                elif current_value == run_value:
                    run_length += 1
                else:
                    rlm[run_value, run_length-1] += 1
                    run_value = current_value
                    run_length = 1
            
            if run_value is not None:
                rlm[run_value, run_length-1] += 1
    
    elif angle == 90:  # Vertical
        for j in range(cols):
            run_length = 1
            run_value = None
            for i in range(rows):
                if not mask[i,j]:
                    if run_value is not None:
                        rlm[run_value, run_length-1] += 1
                    run_value = None
                    continue
                    
                current_value = image[i,j]
                if run_value is None:
                    run_value = current_value
                    run_length = 1
                elif current_value == run_value:
                    run_length += 1
                else:
                    rlm[run_value, run_length-1] += 1
                    run_value = current_value
                    run_length = 1
            
            if run_value is not None:
                rlm[run_value, run_length-1] += 1
    
    # Similar modifications for 45 and 135 degrees...
    elif angle == 45:  # 45 degrees
        for k in range(rows + cols - 1):
            run_length = 1
            run_value = None
            for i in range(max(0, k-cols+1), min(k+1, rows)):
                j = k - i
                if j >= cols:
                    continue
                    
                if not mask[i,j]:
                    if run_value is not None:
                        rlm[run_value, run_length-1] += 1
                    run_value = None
                    continue
                    
                current_value = image[i,j]
                if run_value is None:
                    run_value = current_value
                    run_length = 1
                elif current_value == run_value:
                    run_length += 1
                else:
                    rlm[run_value, run_length-1] += 1
                    run_value = current_value
                    run_length = 1
            
            if run_value is not None:
                rlm[run_value, run_length-1] += 1
    
    elif angle == 135:  # 135 degrees
        for k in range(1-cols, rows):
            run_length = 1
            run_value = None
            for i in range(max(0, k), min(rows, k+cols)):
                j = i - k
                if j < 0 or j >= cols:
                    continue
                    
                if not mask[i,j]:
                    if run_value is not None:
                        rlm[run_value, run_length-1] += 1
                    run_value = None
                    continue
                    
                current_value = image[i,j]
                if run_value is None:
                    run_value = current_value
                    run_length = 1
                elif current_value == run_value:
                    run_length += 1
                else:
                    rlm[run_value, run_length-1] += 1
                    run_value = current_value
                    run_length = 1
            
            if run_value is not None:
                rlm[run_value, run_length-1] += 1
    
    return rlm

def calculate_rlm_features(rlm):
    """Calculate texture features from the run-length matrix."""
    # Convert to float64 at the very beginning
    total_runs = float(np.sum(rlm))  # explicitly convert to float
    total_runs_squared = float(total_runs * total_runs)  # calculate square as float
    
    if total_runs == 0:
        return {
            'Short Run Emphasis': 0,
            'Long Run Emphasis': 0,
            'Gray Level Non-uniformity': 0,
            'Run Length Non-uniformity': 0,
            'Run Percentage': 0
        }
        
    gray_levels, run_lengths = rlm.shape
    rlm = rlm.astype(np.float64)  # convert matrix to float64
    i_indices, j_indices = np.meshgrid(range(gray_levels), range(run_lengths), indexing='ij')
    
    
    gray_sums = np.sum(rlm, axis=1, dtype=np.float64)
    run_sums = np.sum(rlm, axis=0, dtype=np.float64)
    
    sre = np.sum(rlm / (j_indices + 1)**2) / total_runs
    lre = np.sum(rlm * (j_indices + 1)**2) / total_runs
    gln = np.sum(gray_sums**2) / total_runs_squared
    rln = np.sum(run_sums**2) / total_runs_squared
    rp = total_runs / (gray_levels * run_lengths)
    
    return {
        'Short Run Emphasis': sre,
        'Long Run Emphasis': lre,
        'Gray Level Non-uniformity': gln,
        'Run Length Non-uniformity': rln,
        'Run Percentage': rp
    }

def process_image(image_path, mask_path):
    """Process a single image and its mask to calculate texture features."""
    try:
        # Load the image and mask
        image = io.imread(image_path)
        mask = io.imread(mask_path)
        
        # Convert the image to grayscale if it's not already
        if len(image.shape) == 3:
            image = color.rgb2gray(image)
        
        # Convert the image to 8-bit unsigned integer
        image = img_as_ubyte(image)
        
        # Make sure mask is binary
        mask = mask > 0
        
        # Crop to ROI
        cropped_image, cropped_mask = crop_to_roi(image, mask)
        if cropped_image is None:
            print(f"No ROI found in mask for {image_path}")
            return None
        
        angles = [0, 45, 90, 135]
        features = {}
        averaged_features = {}
        
        # Calculate RLM for different angles
        for angle in angles:
            rlm = calculate_rlm(cropped_image, cropped_mask, angle)
            angle_features = calculate_rlm_features(rlm)
            for key, value in angle_features.items():
                features[f'{key}_{angle}deg'] = value
                if key not in averaged_features:
                    averaged_features[key] = []
                averaged_features[key].append(value)
        
        # Calculate averages
        for key, values in averaged_features.items():
            features[f'{key}_mean'] = np.mean(values)
        
        # Add ROI size information
        features['ROI_area'] = np.sum(cropped_mask)
        
        return features
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_folder(image_folder, mask_folder, output_file):
    """Process all images in a folder and save results to Excel."""
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
    
    if not results:
        print("No valid results to save")
        return
    
    # Create a DataFrame and save to Excel
    df = pd.DataFrame(results)
    
    # Separate angle-specific and averaged features
    angle_features = [col for col in df.columns if 'deg' in col]
    mean_features = [col for col in df.columns if 'mean' in col or col in ['Filename', 'ROI_area']]
    
    # Save detailed results
    df.to_excel(output_file, index=False)
    print(f"Detailed results saved to {output_file}")
    
    # Save averaged results
    df_mean = df[mean_features]
    mean_output_file = output_file.replace('.xlsx', '_averaged.xlsx')
    df_mean.to_excel(mean_output_file, index=False)
    print(f"Averaged results saved to {mean_output_file}")

# Example usage
image_folder = "your_image_folder_path"
mask_folder = "your_mask_folder_path"
output_file = ".results.xlsx"

process_folder(image_folder, mask_folder, output_file)