# README

+ [HOME](../../README.md)



## 实现功能

+ 滤波函数

```python
class KernelType(Enum):
    NONE = 0
    AVERAGE = 1
    GAUSSIAN = 2
    MEDIAN = 3
    NIS = 4
    BILATERAL = 5
    Depth = 6
```

+ 前几个如其名
  + 不处理、均值滤波、高斯滤波、中值滤波、NIS、双边滤波、显示深度值（归一化）
+ 其中 `NIS` 是 [Nvidia Image Scale](https://github.com/NVIDIAGameWorks/NVIDIAImageScaling)
  + OpenGL 版本



## NIS 细节

+ 生成 shader 流程如下，具体可以查看[文件](prepare_shaders.py)
  + `glslc` 生成 OpenGL 版本的 `spv` 文件
  + `spirv-cross` 生成 OpenGL 的 `compute shader`
