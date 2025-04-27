import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage import io
import PIL
import os
import mimetypes
import torchvision.transforms as transforms
import glob
from skimage.io import imread
from natsort import natsorted
import re
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage

def split_tensor(tensor, tile_size=256):
    mask = torch.ones_like(tensor)
    # use torch.nn.Unfold
    stride  = tile_size//2
    unfold  = torch.nn.Unfold(kernel_size=(tile_size, tile_size), stride=stride)
    # Apply to mask and original image
    mask_p  = unfold(mask)
    patches = unfold(tensor)
	
    patches = patches.reshape(3, tile_size, tile_size, -1).permute(3, 0, 1, 2)
    if tensor.is_cuda:
        patches_base = torch.zeros(patches.size(), device=tensor.get_device())
    else: 
        patches_base = torch.zeros(patches.size())
	
    tiles = []
    for t in range(patches.size(0)):
        tiles.append(patches[[t], :, :, :])
    return tiles, mask_p, patches_base, (tensor.size(2), tensor.size(3))

def rebuild_tensor(tensor_list, mask_t, base_tensor, t_size, tile_size=256):
    stride  = tile_size//2  
    # base_tensor here is used as a container

    for t, tile in enumerate(tensor_list):
        print(tile.size())
        base_tensor[[t], :, :] = tile  
	
    base_tensor = base_tensor.permute(1, 2, 3, 0).reshape(3*tile_size*tile_size, base_tensor.size(0)).unsqueeze(0)
    fold = torch.nn.Fold(output_size=(t_size[0], t_size[1]), kernel_size=(tile_size, tile_size), stride=stride)
    # https://discuss.pytorch.org/t/seemlessly-blending-tensors-together/65235/2?u=bowenroom
    output_tensor = fold(base_tensor)/fold(mask_t)
    # output_tensor = fold(base_tensor)
    return output_tensor