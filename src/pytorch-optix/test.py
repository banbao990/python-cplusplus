import torch
from torch.utils.cpp_extension import load
from glob import glob
import os
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv

from config import _C as cfg

def compile():
    os.environ['PATH'] = os.environ['PATH'] + ";" + cfg.CL_PATH
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join("build")
    Debug = False # compile with debug flag
    verbose = True # show compile command
    cuda_files = glob("bind.cpp") # source files
    include_dirs = ["../include"] # include directories
    include_dirs.append(cfg.OPTIX_INCLUDE_PATH)
    include_dirs.append(cfg.CUDA_INCLUDE_PATH)
    cflags = ["--extended-lambda --expt-relaxed-constexpr"] # nvcc flags
    if Debug:
        cflags.append("-G -g -O0")
    else:
        cflags.append("-O3")

    # link flags
    ldflags = ["/NODEFAULTLIB:LIBCMT"]
    ldflags.append("/LIBPATH:{}/lib/x64/".format(cfg.CUDA_PATH))
    ldflags.append('cuda.lib')
    ldflags.append('cudart_static.lib')

    demo = load(
        name="pytorch_optix_demo", # name can not have '-'
        sources=cuda_files,
        extra_include_paths=include_dirs,  
        extra_cflags=cflags,
        extra_ldflags=ldflags,
        verbose=verbose,
        with_cuda=False,
    )

    return demo


def read_exr(path: str) -> torch.Tensor:
    """
    Read exr image and convert to torch tensor.
    """

    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    img = torch.from_numpy(img)
    img = img.to(torch.float32)
    img = img.to("cuda")
    return img
   
# main function
if __name__ == "__main__":

    demo = compile()

    # read exr image and convert to torch tensor, get first 3 channels
    img_with_noise = read_exr("images/100spp.exr")
    img_with_noise = img_with_noise
    print(img_with_noise.shape)

    image_clean = demo.optix_denoiser(img_with_noise, img_with_noise)
    print(image_clean.shape)
    # axis 0 <-> axis 1

    cv.imshow("image with noise", img_with_noise.cpu().numpy())
    # close window
    cv.waitKey(0)
    cv.imshow("image clean", image_clean.cpu().numpy())
    cv.waitKey(0)
