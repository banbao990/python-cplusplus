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
```

+ 前几个如其名
  + 不处理、均值滤波、高斯滤波、中值滤波
+ 其中 `NIS` 是 [Nvidia Image Scale](https://github.com/NVIDIAGameWorks/NVIDIAImageScaling)
  + OpenGL 版本



## NIS 细节

