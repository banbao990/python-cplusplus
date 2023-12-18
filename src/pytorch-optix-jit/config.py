from yacs.config import CfgNode as CN
import os

_C = CN()

# _C.CL_PATH = r"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.37.32822/bin/Hostx86/x86"
_C.CL_PATH = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.38.33130/bin/Hostx64/x64"

# can be read from environment variable
_C.OPTIX_SDK_PATH = ""
_C.CUDA_PATH = ""


if(_C.OPTIX_SDK_PATH == ""):
    _C.OPTIX_SDK_PATH = os.environ["OptiX_INSTALL_DIR"]

if(_C.CUDA_PATH == ""):
    _C.CUDA_PATH = os.environ["CUDA_PATH"]
print(_C.CUDA_PATH)

_C.OPTIX_INCLUDE_PATH = os.path.join(_C.OPTIX_SDK_PATH, "include")
_C.CUDA_INCLUDE_PATH = os.path.join(_C.CUDA_PATH, "include")