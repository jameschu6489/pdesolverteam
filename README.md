## 问题描述

二维Acoustic Wave方程

![image](https://github.com/jameschu6489/pdesolverteam_t15_2dwave/blob/main/equation_images/2d_wave.jpg)

其中定义域为

![image](https://github.com/jameschu6489/pdesolverteam_t15_2dwave/blob/main/equation_images/domain.jpg)

## PINNs方法求解Acoustic Wave方程

与论文要求一致。

## 模型结构

与论文要求一致。

## 数据集

与论文要求一致。

## 运行环境要求

计算硬件：Ascend 计算芯片

计算框架：Mindspore 1.7.0，numpy 1.21.2，matplotlib 3.5.1，scipy 1.5.4



## 代码框架

```
.
└─PINNforAcoustic2d
  ├─README.md
  ├─requirements.txt
  ├─solve.py                          # train
  ├─event1                            # seismic data
  ├─src
    ├──config.py                      # parameter configuration
    ├──dataset.py                     # dataset
    ├──model.py                       # network structure

```

## 模型训练

可以直接使用solve.py文件进行PINNs模型训练和求解Acoustic Wave方程。在训练过程中，模型的参数和训练过程也会被自动保存

```
python solve.py
```

模型的损失值会实时展示出来，变化如下：

```
start training...
it: 100, loss: 1.855e+00, time: 107
it: 200, loss: 1.665e+00, time: 18
it: 300, loss: 1.690e+00, time: 18
it: 400, loss: 1.626e+00, time: 18
it: 500, loss: 1.609e+00, time: 18
it: 600, loss: 1.600e+00, time: 18
it: 700, loss: 1.621e+00, time: 18
it: 800, loss: 1.610e+00, time: 18
it: 900, loss: 1.605e+00, time: 18
it: 1000, loss: 1.594e+00, time: 18
...
it: 499800, loss: 9.947e-04, time: 18
it: 499900, loss: 9.036e-04, time: 18
it: 500000, loss: 1.273e-03, time: 18
```

## MindScience官网

可以访问官网以获取更多信息：https://gitee.com/mindspore/mindscience
