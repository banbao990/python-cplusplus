import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

BIN_DIR = ""
if sys.platform == "win32":
    BIN_DIR = "{}/oidn/oidn-2.1.0.x64.windows/bin".format(CURRENT_DIR)
elif sys.platform == "linux":
    BIN_DIR = "{}/oidn/oidn-2.1.0.x86_64.linux/lib".format(CURRENT_DIR)

if os.name == "nt":
    os.add_dll_directory(BIN_DIR)
elif os.name == "posix":
    os.environ["PATH"] += os.pathsep + BIN_DIR
    os.environ["LD_LIBRARY_PATH"] += os.pathsep + BIN_DIR
    sys.path.append(BIN_DIR)

# for utils
sys.path.append("{}/../".format(CURRENT_DIR))
from utils.images import read_exr, tonemap_aces, read_png
from utils.pfm import read_pfm, write_pfm

import oidn_example

import cv2 as cv
import numpy as np
import torch
import ctypes

def pfm_test():
    NAME = "cbox"

    # read png
    img = cv.imread(os.path.join(CURRENT_DIR, "../../assets/images/{}.png".format(NAME)))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # vertival flip
    img = np.flipud(img)
    img = np.array(img, dtype=np.float32)/255.0

    # little endian
    write_pfm(os.path.join(CURRENT_DIR, "../../assets/images/{}.pfm".format(NAME)), img, 1, '<')

    # big endian
    img = img.byteswap()
    write_pfm(os.path.join(CURRENT_DIR, "../../assets/images/{}-b.pfm".format(NAME)), img, 1, '>')

def test_oidn():
    oidn_example.init()

    img_normal = None
    img_albedo = None
    # img_with_noise = read_png(os.path.join(CURRENT_DIR, "../../assets/images/cbox.png"))

    img_with_noise = read_exr(os.path.join(CURRENT_DIR, "../../assets/images/100spp.exr"))
    img_normal = read_exr(os.path.join(CURRENT_DIR, "../../assets/images/normal.exr"))
    img_normal = (img_normal * 2.0 - 1.0).clamp(-1.0, 1.0)
    img_albedo = read_exr(os.path.join(CURRENT_DIR, "../../assets/images/albedo.exr"))

    # only use first 3 channels
    if img_with_noise.shape[2] > 3:
        img_with_noise = img_with_noise[:, :, :3]
    if img_normal != None and img_normal.shape[2] > 3:
        img_normal = img_normal[:, :, :3]
    if img_albedo != None and img_albedo.shape[2] > 3:
        img_albedo = img_albedo[:, :, :3]

    print("original image size: {}".format(img_with_noise.shape))

    xxyy = []
    crops = 5
    for i in range(crops):
        # generate random crop of image_with_noise_original
        SEG = 300
        x_start = np.random.randint(0, max(img_with_noise.shape[0] - SEG, 0))
        x_end = np.random.randint(min(x_start + SEG, img_with_noise.shape[0]), img_with_noise.shape[0])
        y_start = np.random.randint(0, max(img_with_noise.shape[1] - SEG, 0))
        y_end = np.random.randint(min(y_start + SEG, img_with_noise.shape[1]), img_with_noise.shape[1])

        xxyy.append([x_start, x_end, y_start, y_end])

    # whole img
    xxyy.insert(0, [0, img_with_noise.shape[0], 0, img_with_noise.shape[1]])

    for i in xxyy:
        x_start, x_end, y_start, y_end = i
        sub_image = img_with_noise[x_start:x_end, y_start:y_end, :].clone()

        print("image size: {}".format(sub_image.shape))

        image_clean = torch.zeros_like(sub_image).to("cuda")
        oidn_example.denoise(sub_image, image_clean, sub_image.shape[1], sub_image.shape[0], sub_image.shape[2])

        if img_normal != None:
            image_clean_aux = torch.zeros_like(sub_image).to("cuda")
            img_normal_aux = img_normal[x_start:x_end, y_start:y_end, :].clone()
            img_albedo_aux = img_albedo[x_start:x_end, y_start:y_end, :].clone()
            oidn_example.denoise_with_normal_and_albedo(sub_image, img_normal_aux, img_albedo_aux, image_clean_aux, sub_image.shape[1], sub_image.shape[0], sub_image.shape[2])

        cv.imshow("noise", tonemap_aces(sub_image).cpu().numpy())
        cv.imshow("denoised", tonemap_aces(image_clean).cpu().numpy())

        if img_normal != None:
            cv.imshow("denoised(aux)", tonemap_aces(image_clean_aux).cpu().numpy())
        
        if(cv.waitKey(0)):
            cv.destroyAllWindows()

if __name__ == "__main__":
    test_oidn()