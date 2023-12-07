# python-cplucplus

# mi3

+ windows

```bash
nvcc --version  
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Fri_Nov__3_17:51:05_Pacific_Daylight_Time_2023
# Cuda compilation tools, release 12.3, V12.3.103
# Build cuda_12.3.r12.3/compiler.33492891_0
```

+ python 如下

```bash
# env
conda create -n mi3 python=3.10
```

```bash
# pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# mitsuba
pip install mitsuba

# imgui
pip install imgui

# glfw
pip install glfw

# opengl
pip install PyOpenGL PyOpenGL_accelerate

# tinycudann
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd bindings/torch
python setup.py install

# matplotlib, tqdm, tensorboard
pip install matplotlib
pip install tqdm
pip install tensorboard

# torch_scatter
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html

# ninja
pip install ninja
```

