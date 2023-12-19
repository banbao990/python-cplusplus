# python-cplucplus

```
git clone git@github.com:banbao990/python-cplusplus.git
git submodule update --init --recursive
```

# mi3 环境

## windows

+ cuda

```bash
nvcc --version  
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Fri_Nov__3_17:51:05_Pacific_Daylight_Time_2023
# Cuda compilation tools, release 12.3, V12.3.103
# Build cuda_12.3.r12.3/compiler.33492891_0
```

+ optix

```bash
# SDK 8.0.0
```

+ cmake

```bash
cmake --version
# cmake version 3.25.1
```


## python

+ 只需要如下配置，即可执行本工程中的代码

```bash
# env
conda create -n mi3 python=3.10
```

```bash
# pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# ninja, opencv, yacs
pip install ninja opencv-python yacs pybind11
```



## 其他

+ mi 环境中的其他库

```bash
# mitsuba
pip install mitsuba

# glfw, imgui
pip install imgui
pip install glfw

# opengl, cuda
pip install PyOpenGL PyOpenGL_accelerate
pip install cuda-python
pip install pycuda # 会报错, 解决方案见下面

# tinycudann
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd bindings/torch
python setup.py install

# tqdm, tensorboard
pip install tqdm
pip install tensorboard

# torch_scatter
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html

# matplotlib, openexr
pip install matplotlib
pip install OpenEXR
```



### cuda 问题

+ 直接 `pip install pycuda` 报错
  +  `PyCUDA was compiled without GL extension support`
+ [解决方案](https://github.com/harskish/ganspace/issues/43)

```txt
I've actually fixed this one. If you are on a windows device, you should pip install pipwin, then use pipwin to install pycuda. And then it installs it correctly.
```

```bash
pip install pipwin
pipwin install pycuda
```



