import os
import numpy as np
from PIL import Image

def crop_mean_pixel_block(image_path, output_path, crop_size, scale = 1):
    with Image.open(image_path) as img:
        # Convert image to grayscale and then to numpy array
        gray_img = img.convert("L")
        img_array = np.array(gray_img).astype(np.float32)
        img_array[img_array < 10] = 0
        
        # Calculate the total intensity
        total_intensity = np.sum(img_array)
        
        # Calculate the weighted mean position
        y_indices, x_indices = np.indices(img_array.shape)
        mean_x = np.sum(x_indices * img_array) / total_intensity
        mean_y = np.sum(y_indices * img_array) / total_intensity
        
        # Determine crop boundaries
        half_crop_size = crop_size // 2
        left = max(0, int(mean_x) - half_crop_size)
        upper = max(0, int(mean_y) - half_crop_size)
        right = min(img.width, left + crop_size)
        lower = min(img.height, upper + crop_size)
        
        # Adjust the boundaries if the crop area exceeds the image borders
        if right - left < crop_size:
            if left == 0:
                right = min(img.width, crop_size)
            else:
                left = max(0, right - crop_size)
        if lower - upper < crop_size:
            if upper == 0:
                lower = min(img.height, crop_size)
            else:
                upper = max(0, lower - crop_size)
        
        # Crop the image
        cropped_img = img.crop((left, upper, right, lower))
        cropped_img = cropped_img.resize([crop_size * scale, crop_size * scale], Image.NEAREST)
        
        # Save the cropped image
        cropped_img.save(output_path)
        print(f"Cropped image saved to {output_path}")

def crop_max_pixel_block(image_path, output_path, crop_size):
    with Image.open(image_path) as img:
        # Convert image to numpy array
        img_array = np.array(img)
        
        # Find the position of the maximum pixel
        max_pos = np.unravel_index(np.argmax(img_array, axis=None), img_array.shape)
        max_y, max_x = max_pos[:2]  # Get the y, x coordinates (ignore color channels if present)
        
        # Determine crop boundaries
        half_crop_size = crop_size // 2
        left = max(0, max_x - half_crop_size)
        upper = max(0, max_y - half_crop_size)
        right = min(img.width, max_x + half_crop_size + 1)
        lower = min(img.height, max_y + half_crop_size + 1)
        
        # Adjust the boundaries if the crop area exceeds the image borders
        if right - left < crop_size:
            if left == 0:
                right = crop_size
            else:
                left = right - crop_size
        if lower - upper < crop_size:
            if upper == 0:
                lower = crop_size
            else:
                upper = lower - crop_size
        
        # Crop the image
        cropped_img = img.crop((left, upper, right, lower))
        
        # Save the cropped image
        cropped_img.save(output_path)
        print(f"Cropped image saved to {output_path}")

def batch_crop_max_pixel_block(input_path, crop_size, scale):
    for filename in os.listdir(input_path):
        if filename.endswith(".png") and (filename.find('block') == -1):
            image_path = os.path.join(input_path, filename)
            filename_out = filename.replace(".png", "_block.png")
            image_out_path = os.path.join(input_path, filename_out)
            # crop_max_pixel_block(image_path, image_out_path, crop_size, scale)
            crop_mean_pixel_block(image_path, image_out_path, crop_size, scale)

# Example usage
input_path = "/Users/yuanxy/Drives/aDriveSync/Aurora_Figures/OV6946/psf/0626"
crop_size = 71
scale = 10
batch_crop_max_pixel_block(input_path, crop_size, scale)
