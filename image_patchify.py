import numpy as np
import cv2
import matplotlib.pyplot as plt

def pad_image(image, padding_size, borderType=cv2.BORDER_REFLECT_101):
    """Symmetrically pad image according to specified padding size."""
    padded_image = cv2.copyMakeBorder(
        image,
        top=padding_size,
        bottom=padding_size,
        left=padding_size,
        right=padding_size,
        borderType=borderType
    )
    return padded_image

def create_blending_mask(patch_size, blending_pad, padding_size):
    """Create a blending mask for smooth transitions near patch borders."""
    h, w = patch_size
    mask = np.ones((h, w), dtype=np.float32)

    if blending_pad > 0:
        ramp = np.linspace(0, 1, blending_pad)
        # Top
        mask[:blending_pad, :] *= ramp[:, None]
        # Bottom
        mask[-blending_pad:, :] *= ramp[::-1, None]
        # Left
        mask[:, :blending_pad] *= ramp[None, :]
        # Right
        mask[:, -blending_pad:] *= ramp[None, ::-1]

    mask = pad_image(mask, padding_size, cv2.BORDER_CONSTANT)

    return mask

def split_image_into_patches(image, patch_size=(256, 256), stride=128, padding_size=32, blending_pad=32):
    """
    Split image into overlapping patches with blending masks and adjustable padding.
    """
    padded_image = pad_image(image, padding_size)

    img_h, img_w = padded_image.shape[:2]
    patch_h, patch_w = patch_size
    patch_h = patch_h + padding_size * 2
    patch_w = patch_w + padding_size * 2
    patches = []
    masks = []
    positions = []

    blending_mask = create_blending_mask(patch_size, blending_pad, padding_size)

    for y in range(0, img_h - patch_h + 1 + 2 * padding_size, stride):
        for x in range(0, img_w - patch_w + 1 + 2 * padding_size, stride):
            patch = padded_image[y:y + patch_h, x:x + patch_w]
            patches.append(patch)
            masks.append(blending_mask)
            # Position relative to the original (non-padded) image
            orig_x = x - padding_size + patch_w // 2
            orig_y = y - padding_size + patch_h // 2
            positions.append((orig_x, orig_y))

    return patches, masks, positions

def visualize_patches(image, patches, masks, positions, patch_size, stride, padding_size):
    """Visualize original image, patch positions, and ALL patches."""
    import math
    # ==========================
    # Now plot ALL patches nicely
    # ==========================
    num_patches = len(patches)
    cols = 6  # Number of columns you want (adjustable)
    rows = math.ceil(num_patches / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)  # make sure axes is 2D

    for idx, patch in enumerate(patches):
        r = idx // cols
        c = idx % cols
        axes[r, c].imshow(patch)
        axes[r, c].set_title(f"{positions[idx][0]} {positions[idx][1]}")

    # Hide any unused subplots
    for i in range(num_patches, rows * cols):
        r = i // cols
        c = i % cols

    plt.tight_layout()


    # (Optional) show a blending mask separately
    plt.figure(figsize=(4, 4))
    plt.imshow(masks[0], cmap='gray')
    plt.title('Sample Blending Mask')
    plt.show()

def merge_patches(patches, masks, positions, image_size, patch_size=(256, 256), padding_size=32):
    """
    Merge patches back into a full image using blending masks.

    Args:
        patches (list of np.ndarray): List of image patches.
        masks (list of np.ndarray): List of blending masks.
        positions (list of (x, y)): List of patch centers relative to original (unpadded) image.
        image_size (tuple): (height, width) of the original image (before padding).
        patch_size (tuple): (height, width) of the patch (without padding).
        padding_size (int): Size of padding used when extracting patches.

    Returns:
        merged_image (np.ndarray): The reconstructed full image.
    """
    img_h, img_w = image_size
    out_h = img_h + 2 * padding_size
    out_w = img_w + 2 * padding_size
    patch_h, patch_w = patch_size
    patch_h = patch_h + padding_size * 2
    patch_w = patch_w + padding_size * 2

    # Initialize accumulation arrays
    merged_image = np.zeros((out_h, out_w, patches[0].shape[2]), dtype=np.float32)
    weight_sum = np.zeros((out_h, out_w, 1), dtype=np.float32)

    for patch, mask, (center_x, center_y) in zip(patches, masks, positions):
        # Recover the top-left corner of patch (relative to padded image)
        top_left_x = center_x - patch_w // 2 + padding_size
        top_left_y = center_y - patch_h // 2 + padding_size

        # Add patch weighted by mask
        merged_image[top_left_y:top_left_y+patch_h, top_left_x:top_left_x+patch_w] += patch * mask[..., None]
        weight_sum[top_left_y:top_left_y+patch_h, top_left_x:top_left_x+patch_w] += mask[..., None]

    # Avoid division by zero
    weight_sum = np.clip(weight_sum, 1e-6, None)
    merged_image /= weight_sum

    # Remove padding to get back to original image size
    merged_image = merged_image[padding_size:padding_size+img_h, padding_size:padding_size+img_w]
    merged_image = merged_image.astype(np.uint8)

    return merged_image


# Example usage
if __name__ == "__main__":
    # Load an image
    img = cv2.imread('./data/div_000005.png')  # Replace with your image path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_size = [512, 512]
    img = cv2.resize(img, image_size)

    # Parameters
    patch_size = [256] * 2
    stride = patch_size[0] // 2
    padding_size = 32   # independent padding size
    blending_pad = patch_size[0] // 2   # blending area size

    patches, masks, positions = split_image_into_patches(img, patch_size, stride, padding_size, blending_pad)

    print(f"Number of patches: {len(patches)}")
    print(f"First patch position (relative to original image): {positions[0]}")
    print(f"Patch size: {patches[0].shape}, Mask size: {masks[0].shape}")

    merged_image = merge_patches(patches, masks, positions, image_size, patch_size, padding_size)
    
    # Visualization
    visualize_patches(img, patches, masks, positions, patch_size, stride, padding_size)
    # (Optional) show a blending mask separately
    plt.figure(figsize=(4, 4))
    plt.imshow(merged_image)
    plt.title('Merged image')
    plt.show()
