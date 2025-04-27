import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import numpy as np
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import lpips
import cv2

from unet import *

if __name__ == '__main__':
    input_image = torch.rand((1,3,400,400))
    model = UNet(3,3)
    output = model(input_image)
    print(output.size())

    if is_linux():
        DATASET_DIR = '/data/xiaoyun/6946_pair_data_video/frames_raw'
    else:
        DATASET_DIR = 'C:/Projects/dataset/6946_pair_data/'
    MODEL_DIR = './results/unet'
    OUTPUT_DIR = f'/data/xiaoyun/6946_pair_data_video/frames_raw_unet'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    epoch = 30
    model_path = f'{MODEL_DIR}/my_checkpoint_epoch{epoch}.pth'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        num_workers = torch.cuda.device_count() * 4

    BATCH_SIZE = 16

    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    torch.cuda.empty_cache()

    test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    with torch.no_grad():
        for idx in range(1, 3343):
            name = f'frame_{idx:05d}.png'
            img = Image.open(f'{DATASET_DIR}/{name}').convert("RGB")
            img = test_transform(img)
            img = img.to(device)[None, :, :, :]
            y_pred = model(img)[0, :, :, :]
            pred_h = y_pred.cpu().detach().numpy() * 255
            pred_h = np.clip(pred_h, 0, 255)
            pred_h = np.moveaxis(pred_h, 0, -1).astype(np.uint8)
            img_out = cv2.cvtColor(pred_h, cv2.COLOR_RGB2BGR)
            print(f'Process {name} ...')
            cv2.imwrite(f'{OUTPUT_DIR}/{name}', img_out)
            


