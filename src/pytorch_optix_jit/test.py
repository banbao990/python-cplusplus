import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from glob import glob
import os
import numpy as np
from typing import Tuple
import torchvision
import sys

# add module path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../".format(CURRENT_DIR))
from utils.images import *

from config import _C as cfg


def compile():
    os.environ['PATH'] = os.environ['PATH'] + os.pathsep + cfg.CL_PATH
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join("build")
    Debug = False  # compile with debug flag
    verbose = True  # show compile command
    # source files # TODO: how to deal with many cpp files
    cpp_files = glob("bind.cpp", root_dir=CURRENT_DIR)
    cpp_files = [os.path.join(CURRENT_DIR, file) for file in cpp_files]
    # include directories
    include_dirs = [os.path.join(CURRENT_DIR, "../include")]
    include_dirs.append(cfg.OPTIX_INCLUDE_PATH)
    include_dirs.append(cfg.CUDA_INCLUDE_PATH)
    print(include_dirs)

    # compile flags
    cflags = []
    # cflags.append("--extended-lambda --expt-relaxed-constexpr") # nvcc flags

    # link flags
    ldflags = []

    if sys.platform == "win32":
        if Debug:
            cflags.extend(["/DEBUG:FULL", "/Od"])
        else:
            cflags.extend(["/DEBUG:NONE", "/O2"])
    elif sys.platform == "linux":
        if Debug:
            cflags.append("-G -g -O0")
        else:
            cflags.append("-O3")

    if sys.platform == "win32":
        ldflags.append("/NODEFAULTLIB:LIBCMT")
        ldflags.append("/LIBPATH:{}/lib/x64/".format(cfg.CUDA_PATH))
        ldflags.append('cuda.lib')
        ldflags.append('cudart_static.lib')
    elif sys.platform == "linux":
        ldflags.append("-L{}/lib64/stubs/".format(cfg.CUDA_PATH))
        ldflags.append("-lcuda")

    demo = load(
        name="pytorch_optix_demo",  # name can not have '-'
        sources=cpp_files,
        extra_include_paths=include_dirs,
        extra_cflags=cflags,
        extra_ldflags=ldflags,
        verbose=verbose,
        with_cuda=False,
    )

    return demo


# main function
if __name__ == "__main__":

    demo = compile()

    # read exr image and convert to torch tensor, get first 3 channels
    # img_with_noise = gen_noise((720, 720), 1000, True)
    img_with_noise = read_exr(os.path.join(
        CURRENT_DIR, "../../assets/images/100spp.exr"))
    # img_with_noise = read_png(os.path.join(CURRENT_DIR, "../../assets/images/cbox.png"))
    # img_with_noise_tm = read_exr(os.path.join(CURRENT_DIR, "../../assets/images/100spp-tm.exr"))
    print("original image size: {}".format(img_with_noise.shape))

    window_names = ["image with noise", "image clean",
                    "image with noise(tm)", "image clean(tm)"]

    crops = 5
    for i in range(crops):
        # generate random crop of image_with_noise_original
        SEG = 300
        x_start = np.random.randint(0, max(img_with_noise.shape[0] - SEG, 0))
        x_end = np.random.randint(
            min(x_start + SEG, img_with_noise.shape[0]), img_with_noise.shape[0])
        y_start = np.random.randint(0, max(img_with_noise.shape[1] - SEG, 0))
        y_end = np.random.randint(
            min(y_start + SEG, img_with_noise.shape[1]), img_with_noise.shape[1])

        sub_image = img_with_noise[x_start:x_end, y_start:y_end, :].clone()

        print("image size: {}".format(sub_image.shape))

        image_clean = demo.optix_denoise(sub_image)

        # sub_image_tm = img_with_noise_tm[x_start:x_end, y_start:y_end, :].clone()
        # image_clean_tm = demo.optix_denoise(sub_image_tm)

        cv.imshow(window_names[0], tonemap_aces(sub_image).cpu().numpy())
        cv.imshow(window_names[1], tonemap_aces(image_clean).cpu().numpy())
        # the following methods are not recommended, because denoiser must be trained with original images(linear space)
        # cv.imshow(window_names[2], sub_image_tm.cpu().numpy())
        # cv.imshow(window_names[3], (image_clean_tm).cpu().numpy())

        if (cv.waitKey(0)):
            cv.destroyAllWindows()
