import os
import numpy as np
from PIL import Image

def tune_color(img, color_h):
    # Convert the image to HSV (Hue, Saturation, Value) color space
    img_hsv = img.convert('HSV')
    img_hsv_array = np.array(img_hsv)
    img_hsv_array[..., 0] = color_h
    # Convert back to RGB
    img_rgb = Image.fromarray(img_hsv_array, 'HSV').convert('RGB')
    return img_rgb

def crop_and_resize(input_folder, output_folder, M, N):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get a list of all PNG files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Open the image
            with Image.open(input_path) as img:
                # Get the image size (width, height)
                width, height = img.size
                
                # Calculate the coordinates of the center MxM block
                left = (width - M) // 2
                top = (height - M) // 2
                right = left + M
                bottom = top + M
                
                # Crop the center MxM block
                cropped_img = img.crop((left, top, right, bottom))
                
                # Resize the cropped image to NxN
                resized_img = cropped_img.resize((N, N))

                if '620' in filename:
                    resized_img = tune_color(resized_img, 0)
                elif '510' in filename:
                    resized_img = tune_color(resized_img, 85)
                elif '460' in filename:
                    resized_img = tune_color(resized_img, 170)
                else:
                    continue
                
                # Save the resized image
                resized_img.save(output_path)
                print(f"Saved {filename} to {output_folder}")

# Example usage
input_folder = "/Users/yuanxy/Drives/aDriveSync/Aurora_Figures/Paper figures/images/fig2/metalens/PSF/PSF_simulation"
# output_folder = "/Users/yuanxy/Drives/aDriveSync/Aurora_Figures/Paper figures/images/fig2/metalens/PSF/PSF_simulation_resize"
output_folder = "/Users/yuanxy/Drives/aDriveSync/Aurora_Figures/Paper figures/images/fig2/metalens/PSF/PSF_simulation_resize_recolor"
M = 107  # Size of the cropped block (M x M)
N = 300  # Size of the resized block (N x N)

crop_and_resize(input_folder, output_folder, M, N)
