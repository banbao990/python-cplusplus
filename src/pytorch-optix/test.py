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
    cpp_files = glob("bind.cpp") # source files # TODO: how to deal with many cpp files
    include_dirs = ["../include"] # include directories
    include_dirs.append(cfg.OPTIX_INCLUDE_PATH)
    include_dirs.append(cfg.CUDA_INCLUDE_PATH)

    cflags = []
    # cflags.append("--extended-lambda --expt-relaxed-constexpr") # nvcc flags
    if Debug:
        # cflags.append("-G -g -O0")
        cflags.extend(["/DEBUG:FULL", "/Od"])
    else:
        # cflags.append("-O3")
        cflags.extend(["/DEBUG:NONE", "/O2"])

    # link flags
    ldflags = ["/NODEFAULTLIB:LIBCMT"]
    ldflags.append("/LIBPATH:{}/lib/x64/".format(cfg.CUDA_PATH))
    ldflags.append('cuda.lib')
    ldflags.append('cudart_static.lib')

    demo = load(
        name="pytorch_optix_demo", # name can not have '-'
        sources=cpp_files,
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
    img_with_noise = read_exr("../../assets/images/100spp.exr")
    print("original image size: {}".format(img_with_noise.shape))

    window_names = ["image with noise", "image clean"]

    for i in range(5):
        # generate random crop of image_with_noise_original
        SEG = 300
        x_start = np.random.randint(0, max(img_with_noise.shape[0] - SEG, 0))
        x_end = np.random.randint(min(x_start + SEG, img_with_noise.shape[0]), img_with_noise.shape[0])
        y_start = np.random.randint(0, max(img_with_noise.shape[1] - SEG, 0))
        y_end = np.random.randint(min(y_start + SEG, img_with_noise.shape[1]), img_with_noise.shape[1])

        sub_image = img_with_noise[x_start:x_end, y_start:y_end, :].clone()
        print("image size: {}".format(sub_image.shape))

        image_clean = demo.optix_denoise(sub_image)

        cv.imshow(window_names[0], sub_image.cpu().numpy())
        cv.imshow(window_names[1], image_clean.cpu().numpy())
        
        if(cv.waitKey(0)):
            cv.destroyAllWindows()
