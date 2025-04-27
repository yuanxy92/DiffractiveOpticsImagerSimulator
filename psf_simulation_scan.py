import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.fft import fftshift, fft2, ifft2, ifftshift
from torchvision import transforms
from matplotlib import pyplot as plt
import cv2
import scipy.io as sio
import math

from waveoptics.optical_units import *
from waveoptics.metalens_param import *
params['padding_dim'] = 2300

def compute_center_energy(psf_img, xc, yc, center_size):
    total_intensity = np.sum(psf_img)
    psf_size = psf_img.shape[0]
    center_size_psf = psf_size // 2
    center_intensities = []
    xc = round(xc)
    yc = round(yc)

    for idx in range(len(center_size)):
        center_size_half = center_size[idx] // 2
        center_img = psf_img[yc - center_size_half:yc + center_size_half, xc - center_size_half:xc + center_size_half]
        center_intensity = np.sum(center_img)
        intensity_percentage = center_intensity / total_intensity
        center_intensities.append({'center_size': center_size[idx], 'pixel_size': 350e-9, 'xc':xc, 'yc':yc, 'intensity_percentage': intensity_percentage})

    return center_intensities

def compute_and_crop_center_position(psf_img):
    psf_img = psf_img / np.max(psf_img)
    thresh_val, psf_img2 = cv2.threshold(psf_img, 0.5, 1, cv2.THRESH_TOZERO)
    # Get the dimensions of the image
    height, width = psf_img2.shape
    # Create a grid of coordinates
    y_indices, x_indices = np.indices((height, width))
    # Calculate the sum of all pixel values
    total_intensity = psf_img2.sum()
    # Calculate the weighted sum of coordinates
    x_weighted_sum = (x_indices * psf_img2).sum()
    y_weighted_sum = (y_indices * psf_img2).sum()
    # Calculate the weighted center (centroid)
    x_center = x_weighted_sum / total_intensity
    y_center = y_weighted_sum / total_intensity
    return x_center, y_center

def compute_and_crop_center_position_w_angle(psf_img, theta, phi, distance, pixel_size, rindex, crop_half_size):
    psf_img = psf_img / np.max(psf_img)
    theta_out = math.asin(1 / rindex * math.sin(math.radians(theta)))
    phi_out = math.asin(1 / rindex * math.sin(math.radians(phi)))
    # compute pixel shift
    pixel_shift_theta = distance * math.tan(theta_out) / pixel_size
    pixel_shift_phi = distance * math.tan(phi_out) / pixel_size
    # Calculate the weighted center (centroid)
    x_center = psf_img.shape[1] // 2 - pixel_shift_theta
    y_center = psf_img.shape[0] // 2 - pixel_shift_phi

    # crop blocks from image
    x_lf = x_center - crop_half_size
    if x_lf < 0:
        psf_img2 = np.copy(psf_img)
        cut_pos = psf_img2.shape[1] // 2
        psf_img2 = np.roll(psf_img2, [0, cut_pos])
        x_center2 = x_center + cut_pos
    else:
        psf_img2 = psf_img
        x_center2 = x_center

    xc = math.floor(x_center2)
    yc = math.floor(y_center)
    output_mat_crop = psf_img2[yc - crop_half_size:yc + crop_half_size + 1, xc - crop_half_size:xc + crop_half_size + 1]

    return x_center, y_center, output_mat_crop, psf_img2

def wavelength_to_rgb(wavelength):
    """Convert a wavelength in the range 380-780 nm to an RGB color value."""

    gamma = 0.8
    intensity_max = 255
    factor = 0.0
    R = G = B = 0

    if 380 <= wavelength <= 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 < wavelength <= 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 < wavelength <= 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 < wavelength <= 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 < wavelength <= 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 < wavelength <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0.0  # Wavelength out of range

    # Let the intensity fall off near the vision limits
    if 380 <= wavelength <= 420:
        factor = 0.3 + 0.7*(wavelength - 380) / (420 - 380)
    elif 420 < wavelength <= 645:
        factor = 1.0
    elif 645 < wavelength <= 780:
        factor = 0.3 + 0.7*(780 - wavelength) / (780 - 645)
    else:
        factor = 0.0

    def adjust(color, factor):
        if color == 0.0:
            return 0
        else:
            return int(intensity_max * ((color * factor) ** gamma))

    R = adjust(R, factor)
    G = adjust(G, factor)
    B = adjust(B, factor)

    return (R, G, B)

def intensity_to_rgb(intensity, color_rgb_val):
    rgb_mat = np.zeros((intensity.shape[0], intensity.shape[1], 3), np.float32)
    rgb_mat[:, :, 0] = color_rgb_val[2] * intensity
    rgb_mat[:, :, 1] = color_rgb_val[1] * intensity
    rgb_mat[:, :, 2] = color_rgb_val[0] * intensity
    return rgb_mat

def main():
    device = torch.device('cuda:0')

    data_folder_path = 'D:/Code/Aurora_papers/data/240729_Best_results'
    output_folder_path = './results/240729_Best_results'
    duty_names = ['270723_Ep0+200_Dataset46_BS4_init0_theta28_lr4_D200_GPU5']
    # duty_names = ['240722_Ep0+30_Dataset2650_BS16_init0_theta28_lr4_D200_try3', '240722_Ep0+50_Dataset1491_BS16_init0_theta28_lr4_D200_try4',
    # '240724_Ep0+50_Dataset1491_BS16_init0_theta28_lr4_D200_GPU4',
    # '240725_Ep0+50_Dataset1491_BS16_init0_theta28_lr4_D200_GPU4',
    # '240726_Ep0+30_Dataset2650_BS16_init0_theta28_lr4_D200_GPU0',
    # '240727_Ep0+50_Dataset1491_BS16_init0_theta28_lr4_D200_GPU1',
    # '240727_Ep0+50_Dataset1491_BS16_init0_theta28_lr4_D200_GPU2',
    # '270723_Ep0+200_Dataset46_BS4_init0_theta28_lr4_D200_GPU5']
    # data_folder_path = './data/240731_Best_results'
    # output_folder_path = './results/240731_Best_results'
    # duty_names = ['240731_Ep0+50_Dataset1491_BS16_init0_theta28_lr4_D200_GPU3']

    for duty_idx in range(len(duty_names)):
        duty_name = duty_names[duty_idx]
        numpy_filename = f'{data_folder_path}/{duty_name}/final_params_best/duty.npy'
        output_dir = f'{output_folder_path}/{duty_name}'
        os.makedirs(output_dir, exist_ok=True)

        # set parameter
        aperture_diameter = 0.2e-3
        # Wavelengths and field angles
        # lambda_base = [625.0, 530.0, 460.0]
        lambda_base = []
        for idx in range(31):
            lambda_base.append(400 + idx * 10)
        channel_idx = [2] * len(lambda_base)
        theta_base = [0.0, 10.0, 20.0, 30.0]
        phi = 0.0 
        refractive_index = 1.45
        # refractive_index = 1.45
        # lens-sensor distance
        prop_dist = []
        for idx in range(5):
            prop_dist.append((1.110 + idx*0.02) * 1e-3)
        prop_dist.append(1.05 * 1e-3)
        prop_dist.append(1.25 * 1e-3)

        # psf cropping size
        crop_size = 201
        crop_size_half = crop_size // 2

        # load duty of metalens 
        duty = np.load(numpy_filename)
        duty = np.clip(duty, 0.3, 0.82)
        padding_values = ((0, 0), (0, 0))
        padding_size_duty_ = (2285 - duty.shape[0]) // 2
        padding_size_duty = ((padding_size_duty_, padding_size_duty_), (padding_size_duty_, padding_size_duty_))
        duty = np.pad(duty, padding_size_duty, 'constant', constant_values=padding_values)

        # set aperture
        params['pixels_aperture'] = aperture_diameter / params['pitch']
        aperture_mask1_ = 1 - ((x_mesh ** 2 + y_mesh ** 2) < (params['pixels_aperture'] * params['Lx'] / 2.0) ** 2)
        aperture_mask2_ = 1 - ((x_mesh ** 2 + y_mesh ** 2) > (19 * params['Lx'] / 2.0) ** 2)
        aperture_mask_ = 1 - (aperture_mask1_ + aperture_mask2_)

        if aperture_diameter > 0.38e-3:
            aperture_mask_inner_ = ((x_mesh ** 2 + y_mesh ** 2) < (0.38e-3 / params['pitch'] * params['Lx'] / 2.0) ** 2)
            aperture_mask_ = (aperture_mask_ + aperture_mask_inner_ * 1) / 2
        # padding
        padding_size = ((params['padding_dim'], params['padding_dim']), (params['padding_dim'], params['padding_dim']))
        padding_values = ((0, 0), (0, 0))
        aperture_mask = np.pad(aperture_mask_, padding_size, 'constant', constant_values=padding_values)

        lambda_shifts = [0]
        center_sizes = [21, 41, 61, 81, 101, 121, 141, 161, 181, 201]
        for dist in prop_dist:
            output_dir_in = f'{output_dir}/{dist*1000:1.3f}'
            os.makedirs(output_dir_in, exist_ok=True)
            output_dir_in_mat = f'{output_dir_in}/energy_stat'
            os.makedirs(output_dir_in_mat, exist_ok=True)
            for theta in theta_base:
                for lambda_idx in range(len(lambda_base)):
                    psf_img_crop = np.zeros((crop_size, crop_size), dtype=np.float32)
                    psf_img = np.zeros((params['whole_dim'] + 2 * params['padding_dim'], params['whole_dim'] + 2 * params['padding_dim']), dtype=np.float32)
                    xcs = 0
                    ycs = 0
                    for lambda_shift_idx in range(len(lambda_shifts)):
                        lambda_ = lambda_base[lambda_idx] + lambda_shifts[lambda_shift_idx]
                        params['lam0'] = lambda_ * params['nanometers']
                        prop = AngSpecProp(params['whole_dim'] + 2 * params['padding_dim'], params['pitch'], dist, params['lam0'], refractive_index)
                        prop.to(device=device)

                        image_amp = aperture_mask
                        init_phase = generate_phase_wavefront(params['whole_dim'], params['whole_dim'], params['lam0'], theta, phi, params['pitch'])
                        phase = phase_from_duty_and_lambda(duty, params)
                        # image_amp = np.ones(phase.shape, dtype=np.float32)
                        image_phase = init_phase + phase
                        image_phase = np.pad(image_phase, padding_size, 'constant', constant_values=padding_values)
                        image_complex = image_amp * np.cos(image_phase) + 1j * image_amp * np.sin(image_phase)
                        image_complex_tensor = torch.from_numpy(image_complex).cuda()
                        x = prop(image_complex_tensor)
                        output_int = x.abs() ** 2
                        input = image_complex_tensor.abs().cpu().detach().numpy()
                        output_mat = output_int.abs().cpu().detach().numpy()[::-1, ::-1]
                        output_mat = output_mat / np.max(output_mat) * 255
                        output_mat = output_mat.astype(np.uint8)
                        # compute center
                        xc, yc, output_mat_crop, psf_ = compute_and_crop_center_position_w_angle(output_mat, theta, phi, dist, params['pitch'], refractive_index, crop_size_half)
                        psf_img_crop = psf_img_crop + output_mat_crop
                        psf_img = psf_img + psf_
                        xcs = xcs + xc
                        ycs = ycs + yc

                    output_rgb_crop = np.zeros((psf_img_crop.shape[0], psf_img_crop.shape[1], 3), dtype=np.uint8)
                    output_rgb = np.zeros((psf_img.shape[0], psf_img.shape[1], 3), dtype=np.uint8)
                    xcs = xcs / len(lambda_shifts)
                    ycs = ycs / len(lambda_shifts)

                    color_rgb_val = wavelength_to_rgb(lambda_base[lambda_idx])
                    output_rgb_crop = intensity_to_rgb(psf_img_crop / len(lambda_shifts), color_rgb_val)
                    output_rgb = intensity_to_rgb(psf_img / len(lambda_shifts), color_rgb_val)

                    energy_percentage = compute_center_energy(psf_img / len(lambda_shifts), xcs, ycs, center_size=center_sizes)

                    print(f'lambda_{lambda_base[lambda_idx]:03.1f}_dist_{dist*1000:01.3f}_theta_{theta:02.1f}_n_{refractive_index}.png')
                    cv2.imwrite(f'{output_dir_in}/lambda_{lambda_base[lambda_idx]:03.1f}_dist_{dist*1000:01.3f}_theta_{theta:02.1f}_phi_{phi:02.1f}_n_{refractive_index}_crop.png', output_rgb_crop)
                    # cv2.imwrite(f'{output_dir}/lambda_{lambda_base[lambda_idx]:03.1f}_dist_{dist*1000:01.3f}_theta_{theta:02.1f}_phi_{phi:02.1f}_n_{refractive_index}.png', output_rgb)
                    np.save(f'{output_dir_in_mat}/lambda_{lambda_base[lambda_idx]:03.1f}_dist_{dist*1000:01.3f}_theta_{theta:02.1f}_phi_{phi:02.1f}_n_{refractive_index}.npy', energy_percentage)
                    sio.savemat(f'{output_dir_in_mat}/lambda_{lambda_base[lambda_idx]:03.1f}_dist_{dist*1000:01.3f}_theta_{theta:02.1f}_phi_{phi:02.1f}_n_{refractive_index}.mat', {'energy_percentage': energy_percentage})

if __name__ == "__main__":
    main()
