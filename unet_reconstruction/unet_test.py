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
        DATASET_DIR = '/data/xiaoyun/6946_pair_data2/train_image_and_label/'
    else:
        DATASET_DIR = 'C:/Projects/dataset/6946_pair_data/'
    MODEL_DIR = './results/unet'
    OUTPUT_DIR = f'{DATASET_DIR}/test_resutls'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    epoch = 30
    model_path = f'{MODEL_DIR}/my_checkpoint_epoch{epoch}.pth'

    train_dataset = OV6946MetalensDataset(DATASET_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        num_workers = torch.cuda.device_count() * 4

    BATCH_SIZE = 16

    train_dataloader = DataLoader(dataset=train_dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    torch.cuda.empty_cache()

    with torch.no_grad():
        for idx, imgs in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = imgs[0].float().to(device)
            label = imgs[1].float().to(device)
            names = imgs[2]
            y_pred = model(img)
            a = 1
            # save reults
            img_h = img.cpu().detach().numpy() * 255
            label_h = label.cpu().detach().numpy() * 255
            pred_h = y_pred.cpu().detach().numpy() * 255
            img_h = np.clip(img_h, 0, 255)
            label_h = np.clip(label_h, 0, 255)
            pred_h = np.clip(pred_h, 0, 255)
            img_h = np.moveaxis(img_h, 1, -1).astype(np.uint8)
            label_h = np.moveaxis(label_h, 1, -1).astype(np.uint8)
            pred_h = np.moveaxis(pred_h, 1, -1).astype(np.uint8)
            for save_idx in range(BATCH_SIZE):
                img = np.zeros((512, 512 * 3, 3), np.float32)
                img[:, 0:512, :] = img_h[save_idx, :, :, :]
                img[:, 512:1024, :] = label_h[save_idx, :, :, :]
                img[:, 1024:1536, :] = pred_h[save_idx, :, :, :]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                name = os.path.basename(names[save_idx])
                cv2.imwrite(f'{OUTPUT_DIR}/{name}.jpg', img)
            


