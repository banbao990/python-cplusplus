# python-cpp-extension

```
git clone git@github.com:banbao990/python-cplusplus.git
git submodule update --init --recursive
```

+ **<span style="color:red">MUST DO THIS</span>**
```bash
python prepare.py
# if complies error, do the following cmd instead
# python prepare.py --clean --all
```

+ `src/config.py` 文件是**<span style="color:red">环境配置</span>**，需要根据自己的环境进行配置



# mi3 环境

## 例子

+ 基本例子

|        module         |    window    |    Linux     |                备注                 |
| :-------------------: | :----------: | :----------: | :---------------------------------: |
|   pytorch_cuda_jit    | $\checkmark$ | $\checkmark$ |              直接执行               |
|   pytorch_optix_jit   | $\checkmark$ | $\checkmark$ |              直接执行               |
| python_cpp_setuptools | $\checkmark$ | $\checkmark$ |              安装执行               |
|   python_cpp_cmake    | $\checkmark$ | $\checkmark$ |              安装执行               |
|      cmake_oidn       | $\checkmark$ | $\checkmark$ | 安装执行<br />（更新见 setup 版本） |

+ 其他例子
  + 实现功能 optix（`albedo+normal`、`temporal`）

  + cmake_optix 问题：linux 运行报错


```txt
undefined symbol:
  _ZN3c106detail14torchCheckFailEPKcS2_jRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

|   module    |    window    |    Linux     |                备注                 |
| :---------: | :----------: | :----------: | :---------------------------------: |
| cmake_optix | $\checkmark$ |              | 安装执行<br />（更新见 setup 版本） |
| setup_optix | $\checkmark$ | $\checkmark$ |              安装执行               |
| setup-oidn  | $\checkmark$ |              |              安装执行               |

+ 测试环境
  + windows
    + cuda 12.3、optix SDK 8.0.0、cmake 3.25.1
  + linux
    + cuda 12.1、optix SDK 8.0.0、cmake 3.25.1
+ Optix SDK：[Link](https://developer.nvidia.com/designworks/optix/download)


```txt
https://developer.nvidia.com/downloads/designworks/optix/secure/8.0.0/nvidia-optix-sdk-8.0.0-win64.exe
https://developer.nvidia.com/downloads/designworks/optix/secure/8.0.0/nvidia-optix-sdk-8.0.0-linux64-x86_64.sh
```

+ 直接执行（例子）

```bash
python src/pytorch_cuda_jit/test.py
```

+ 安装执行（例子）

```bash
# install
python src/python_cpp_setuptools/install.py
# run
python src/python_cpp_setuptools/test.py
```



### setup_optix

+ 运行失败报错 `libGL error: MESA-LOADER: failed to open swrast`，虚拟环境中安装 `gcc`

```bash
 conda install -c conda-forge gcc
```

+ 注意如果是 `setup_optix` 想要在 `GPU-UI` 模式下运行，执行如下命令
  + 其中 `CUDA_VISIBLE_DEVICES=0` 表示有多张显卡，选择使用 `id=0` 的
  + 不加环境，直接运行会报错 `CUDA_ERROR`

```bash
# GPU
CUDA_VISIBLE_DEVICES=0 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python src/utils/ui.py --force_gpu_ui

# CPU
python src/utils/ui.py
```




## python

+ 只需要如下配置，即可执行本工程中的代码

```bash
# env
conda create -n mi3 python=3.10
```

```bash
# pytorch(查看官网给的命令)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# ninja, opencv, yacs
pip install ninja opencv-python yacs pybind11

# ui
pip install imgui glfw cuda-python PyOpenGL PyOpenGL_accelerate

# mitsuba
pip install mitsuba
```



### 其他库

+ mi 环境中的其他库

```bash
# tinycudann
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd bindings/torch
python setup.py install

# tqdm, tensorboard
pip install tqdm tensorboard

# torch_scatter
# https://github.com/rusty1s/pytorch_scatter/issues/186
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html

# matplotlib
pip install matplotlib

# openexr: linux 必须先执行第一步
conda install openexr-python --channel conda-forge
pip install OpenEXR

# oidn (CPU version)
pip install oidn
```



## 其他问题

### pycuda(<span style="color:red">deprecated</span>)

```bash
pip install pycuda # 会报错, 解决方案见下面
```

+ **这个库能够完全被 `cuda-python` 库取代，现在也不用了**
+ 直接 `pip install pycuda` 报错
  +  `PyCUDA was compiled without GL extension support`

#### windows

+ [解决方案](https://github.com/harskish/ganspace/issues/43)

```txt
I've actually fixed this one. If you are on a windows device, you should pip install pipwin, then use pipwin to install pycuda. And then it installs it correctly.
```

```bash
pip install pipwin
pipwin install pycuda
```

#### linux

+ 从源码安装：[code](https://github.com/inducer/pycuda)
