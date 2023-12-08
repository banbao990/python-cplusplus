from yacs.config import CfgNode as CN
import os

_C = CN()

# _C.CL_PATH = r"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.37.32822/bin/Hostx86/x86"
_C.CL_PATH = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.37.32822/bin/Hostx64/x64"
_C.OPTIX_SDK_PATH = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0"
_C.CUDA_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3"


_C.OPTIX_INCLUDE_PATH = os.path.join(_C.OPTIX_SDK_PATH, "include")
_C.CUDA_INCLUDE_PATH = os.path.join(_C.CUDA_PATH, "include")