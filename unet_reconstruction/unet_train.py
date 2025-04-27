import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import lpips

from unet import *

if __name__ == '__main__':
    input_image = torch.rand((1,3,400,400))
    model = UNet(3,3)
    output = model(input_image)
    print(output.size())

    DATASET_DIR = '/data/xiaoyun/6946_pair_data2/train_image_and_label/'
    WORKING_DIR = './results/unet2'
    train_dataset = OV6946MetalensDataset(DATASET_DIR)
    generator = torch.Generator().manual_seed(25)

    train_dataset, test_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        num_workers = torch.cuda.device_count() * 4

    LEARNING_RATE = 3e-4
    BATCH_SIZE = 16

    train_dataloader = DataLoader(dataset=train_dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_channels=3, out_channels=3).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    criterion = custom_loss

    torch.cuda.empty_cache()

    EPOCHS = 100

    train_losses = []
    train_psnrs = []
    val_losses = []
    val_psnrs = []

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        train_running_psnr = 0
        
        for idx, imgs in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = imgs[0].float().to(device)
            label = imgs[1].float().to(device)
            
            y_pred = model(img)
            optimizer.zero_grad()
            
            psnr = psnr_loss(y_pred, label)
            loss = criterion(y_pred, label)
            
            train_running_loss += loss.item()
            train_running_psnr += psnr.item()

            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)
        train_psnr = train_running_psnr / (idx + 1)
        
        train_losses.append(train_loss)
        train_psnrs.append(train_psnr)

        model.eval()
        val_running_loss = 0
        val_running_psnr = 0
        
        with torch.no_grad():
            for idx, imgs in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                img = imgs[0].float().to(device)
                label = imgs[1].float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, label)
                psnr = psnr_loss(y_pred, label)
                
                val_running_loss += loss.item()
                val_running_psnr += psnr.item()

            val_loss = val_running_loss / (idx + 1)
            val_psnr = val_running_psnr / (idx + 1)
        
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)

        scheduler.step(val_loss)

        print("-" * 30)
        print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Training PSNR EPOCH {epoch + 1}: {train_psnr:.4f}")
        print("\n")
        print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print(f"Validation PSNR EPOCH {epoch + 1}: {val_psnr:.4f}")
        print("-" * 30)

        # Saving the model
        torch.save(model.state_dict(), f'{WORKING_DIR}/ckpt_epoch{epoch}.pth')
