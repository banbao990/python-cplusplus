# README

+ 在 `config.py` 文件中进行配置
+ 最好按照下面的方式运行
  + 因为 `torch.utils.cpp_extensions` 自动添加了 `/showIncludes`，会有大段的输出


```bash
python test.py > log.txt
```

+ 如果出现 `cv.imshow()` 的窗口说明运行成功了



# INFO

+ 细节不再更新，之后的更新在 `src/pytorch-optix-cmake` 示例中
