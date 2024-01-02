from yacs.config import CfgNode as CN
import os
import sys

_C = CN()

_C.CL_PATH = ""
# can be read from environment variable
_C.OPTIX_SDK_PATH = ""
_C.CUDA_PATH = ""

if sys.platform == "win32":
    _C.CL_PATH = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.38.33130/bin/Hostx64/x64"
    if (_C.OPTIX_SDK_PATH == ""):
        _C.OPTIX_SDK_PATH = os.environ["OptiX_INSTALL_DIR"]
    if (_C.CUDA_PATH == ""):
        _C.CUDA_PATH = os.environ["CUDA_PATH"]
elif sys.platform == "linux":
    user_home = os.path.expanduser('~')
    # gcc
    _C.CL_PATH = "/usr/bin"
    _C.OPTIX_SDK_PATH = "{}/data/jhj/soft/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/".format(
        user_home)
    _C.CUDA_PATH = "/usr/local/cuda-12.1"

_C.OPTIX_INCLUDE_PATH = os.path.join(_C.OPTIX_SDK_PATH, "include")
_C.CUDA_INCLUDE_PATH = os.path.join(_C.CUDA_PATH, "include")
