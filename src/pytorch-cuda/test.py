import torch
from torch.utils.cpp_extension import load
from glob import glob
import os

# CL_PATH = r"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.37.32822/bin/Hostx86/x86"
CL_PATH = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.37.32822/bin/Hostx64/x64"
print("add your 'cl.exe' to path, example:\n\t{}".format(CL_PATH))
os.environ['PATH'] = os.environ['PATH'] + ";" + CL_PATH

os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join("build")
Debug = False # compile with debug flag
verbose = True # show compile command
cuda_files = glob("bind.cu") # source files
include_dirs = ["../include"] # include directories
cflags = "--extended-lambda --expt-relaxed-constexpr " # nvcc flags
if Debug:
    cflags += "-G -g -O0 "
else:
    cflags += "-O3 "
cuda_module = load(
    name="cuda_module",
    sources=cuda_files,
    extra_include_paths=include_dirs,  
    extra_cflags=[cflags],
    verbose=verbose,
)

N = 10000
a = torch.arange(N, device="cuda", dtype=torch.float32)
b = torch.arange(N, device="cuda", dtype=torch.float32)

c = cuda_module.custom_cuda_func(a, b)

print(c)

assert torch.allclose(c, a + b)

