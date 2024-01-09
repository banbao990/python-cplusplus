from yacs.config import CfgNode as CN
import os
import sys

_C = CN()

############################# Personal Settings ########################
OPTIX_SDK_PATH = ""
CUDA_PATH = ""
if sys.platform == "linux":
    user_home = os.path.expanduser('~')
    # gcc
    OPTIX_SDK_PATH = "{}/data/jhj/soft/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/".format(
        user_home)
    CUDA_PATH = "/usr/local/cuda-12.1"

############################# Personal Settings ########################


if (OPTIX_SDK_PATH == ""):
    OPTIX_SDK_PATH = os.environ["OptiX_INSTALL_DIR"]
if (CUDA_PATH == ""):
    CUDA_PATH = os.environ["CUDA_PATH"]

INCLUDE_PATHS = []
INCLUDE_PATHS.append(os.path.join(OPTIX_SDK_PATH, "include"))
INCLUDE_PATHS.append(os.path.join(CUDA_PATH, "include"))

_C.OPTIX_INCLUDE_PATHS = ";".join(INCLUDE_PATHS)
_C.OPTIX_INCLUDE_PATHS = _C.OPTIX_INCLUDE_PATHS.replace("\\", "/")


def check_optix_path():
    print(
        "\033[93mPlease check your optix settings in the {}\033[00m".format(__file__))
    print("OptiX SDK Path: {}".format(OPTIX_SDK_PATH))
    print("Cuda Path: {}".format(CUDA_PATH))
    print()
