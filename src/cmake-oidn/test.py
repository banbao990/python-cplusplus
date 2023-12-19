# for oidn_example
import os
os.add_dll_directory(R"{}\oidn\oidn-2.1.0.x64.windows\bin".format(os.path.dirname(os.path.abspath(__file__))))

# for utils
import sys
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../".format(path))
from utils.images import read_exr, tonemap_aces, read_png

import oidn_example

import cv2 as cv
import numpy as np
import torch
import ctypes


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../".format(CURRENT_DIR))

oidn_example.init()

# img_with_noise = read_exr(os.path.join(CURRENT_DIR, "../../assets/images/100spp.exr"))
img_with_noise = read_png(os.path.join(CURRENT_DIR, "../../assets/images/cbox.png"))
# only use first 3 channels
if img_with_noise.shape[2] > 3:
    img_with_noise = img_with_noise[:, :, :3]

print("original image size: {}".format(img_with_noise.shape))

crops = 5
for i in range(crops):
    # generate random crop of image_with_noise_original
    SEG = 300
    x_start = np.random.randint(0, max(img_with_noise.shape[0] - SEG, 0))
    x_end = np.random.randint(min(x_start + SEG, img_with_noise.shape[0]), img_with_noise.shape[0])
    y_start = np.random.randint(0, max(img_with_noise.shape[1] - SEG, 0))
    y_end = np.random.randint(min(y_start + SEG, img_with_noise.shape[1]), img_with_noise.shape[1])

    sub_image = img_with_noise[x_start:x_end, y_start:y_end, :].clone()
    image_clean = torch.zeros_like(sub_image).to("cuda")

    print("image size: {}".format(sub_image.shape))

    oidn_example.denoise(sub_image, image_clean, sub_image.shape[1], sub_image.shape[0], sub_image.shape[2])

    cv.imshow("noise", tonemap_aces(sub_image).cpu().numpy())
    cv.imshow("denoised", tonemap_aces(image_clean).cpu().numpy())
    
    if(cv.waitKey(0)):
        cv.destroyAllWindows()