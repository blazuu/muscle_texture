import os
import numpy as np
import cv2
from pathlib import Path
import pandas as pd

def analyze_brightness(image_folder, mask_folder, output_file='brightness_results.xlsx'):
    """
    Analyze brightness of segmented regions in ultrasound images using corresponding masks.
    Masks should have white (255) regions as the area of interest and black (0) as background.
    Results are saved to an Excel file with formatted sheets.
    
    Parameters:
    image_folder (str): Path to folder containing original ultrasound images
    mask_folder (str): Path to folder containing binary masks (white on black)
    output_file (str): Name of the Excel file to save results
    """
    
    # Create results dictionary
    results = {
        'Image Name': [],
        'Mean Brightness': [],
        'Median Brightness': [],
        'Std Deviation': [],
        'Min Brightness': [],
        'Max Brightness': [],
        'ROI Area (pixels)': []
    }
    
    # Ensure folders exist
    image_path = Path(image_folder)
    mask_path = Path(mask_folder)
    
    if not image_path.exists() or not mask_path.exists():
        raise ValueError("Image or mask folder does not exist!")
    
    # Get list of image files
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for image_file in image_files:
        try:
            # Load image and corresponding mask
            image = cv2.imread(str(image_path / image_file), cv2.IMREAD_GRAYSCALE)
            mask_file = image_file  # Assuming mask has same filename as image
            mask = cv2.imread(str(mask_path / mask_file), cv2.IMREAD_GRAYSCALE)
            
            # Verify image and mask dimensions match
            if image.shape != mask.shape:
                print(f"Warning: Dimensions mismatch for {image_file}. Skipping...")
                continue
            
            # Create binary mask (255 -> 1, 0 -> 0)
            binary_mask = (mask > 127).astype(np.uint8)
            
            # Get pixels only in the white mask region
            valid_pixels = image[mask > 127]
            
            if len(valid_pixels) == 0:
                print(f"Warning: No valid pixels in mask for {image_file}. Skipping...")
                continue
            
            # Calculate statistics
            results['Image Name'].append(image_file)
            results['Mean Brightness'].append(round(np.mean(valid_pixels), 2))
            results['Median Brightness'].append(round(np.median(valid_pixels), 2))
            results['Std Deviation'].append(round(np.std(valid_pixels), 2))
            results['Min Brightness'].append(int(np.min(valid_pixels)))
            results['Max Brightness'].append(int(np.max(valid_pixels)))
            results['ROI Area (pixels)'].append(len(valid_pixels))
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Create Excel writer object
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Write main results to first sheet
        df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Detailed Results']
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D9E1F2',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'border': 1
        })
        
        number_format = workbook.add_format({
            'border': 1,
            'num_format': '0.00'
        })
        
        # Apply formats
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            # Set column width based on maximum length of data in each column
            max_length = max(
                df[value].astype(str).apply(len).max(),
                len(value)
            )
            worksheet.set_column(col_num, col_num, max_length + 2)
        
        # Apply cell formats to data
        for row_num in range(len(df)):
            for col_num in range(len(df.columns)):
                if col_num == 0:  # Image Name column
                    worksheet.write(row_num + 1, col_num, df.iloc[row_num, col_num], cell_format)
                else:  # Numeric columns
                    worksheet.write(row_num + 1, col_num, df.iloc[row_num, col_num], number_format)
        
        # Add summary statistics sheet
        summary_stats = df.describe()
        summary_stats.to_excel(writer, sheet_name='Summary Statistics')
        
        # Format summary sheet
        summary_sheet = writer.sheets['Summary Statistics']
        for col_num, value in enumerate(summary_stats.columns.values):
            summary_sheet.write(0, col_num + 1, value, header_format)
            summary_sheet.set_column(col_num + 1, col_num + 1, max(len(value) + 2, 12))
        
        # Add summary formulas sheet
        summary_data = {
            'Metric': ['Total Images Processed', 'Average Mean Brightness', 'Overall Brightness Range',
                      'Total ROI Area (pixels)', 'Average ROI Area (pixels)'],
            'Value': [
                len(df),
                f"=AVERAGE('Detailed Results'!B:B)",
                f"=MAX('Detailed Results'!E:E) - MIN('Detailed Results'!E:E)",
                f"=SUM('Detailed Results'!G:G)",
                f"=AVERAGE('Detailed Results'!G:G)"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary Metrics', index=False)
        
        # Format summary metrics sheet
        summary_metrics_sheet = writer.sheets['Summary Metrics']
        for col_num, value in enumerate(summary_df.columns.values):
            summary_metrics_sheet.write(0, col_num, value, header_format)
            summary_metrics_sheet.set_column(col_num, col_num, max(len(value) + 2, 20))
    
    print(f"Results saved to {output_file}")
    return df

# Example usage
if __name__ == "__main__":
    # Update these paths to match your folder structure
    IMAGE_FOLDER = "your_image_folder_path"
    MASK_FOLDER = "your_mask_folder_path"
    
    # Run analysis
    results_df = analyze_brightness(IMAGE_FOLDER, MASK_FOLDER, ".results.xlsx")