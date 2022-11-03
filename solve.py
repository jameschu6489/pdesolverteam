import os
import numpy as np
import time
import matplotlib.pyplot as plt

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import context
from mindspore.common import set_seed

from src.model import DNN, GradFirst, GradSec, PINNAcoustic2d, MyTrainOneStep
from src.dataset import DatasetAcoustic2d
from src.config import OptionsWaveInverse

import time


def train():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    # 初始化参数器
    args = OptionsWaveInverse().parse()

    # 初始化训练数据
    dataset = DatasetAcoustic2d(args.res_batch, args.bcs_batch)
    X_pde, X_init, X_S, X_BC_t, U_ini1x, U_ini1z, U_ini2x, U_ini2z, Sx, Sz = dataset.init_data()

    # 初始化网络模型
    acoustic_net = DNN(args.acoustic_layers, args.weight_init)
    wavespeed_net = DNN(args.wavespeed_layers, args.weight_init)

    # 初始化梯度算子
    acoustic_gradfn_first = GradFirst(acoustic_net)
    acoustic_gradfn_second_res = GradSec(acoustic_net, args.res_batch)
    acoustic_gradfn_second_bcs = GradSec(acoustic_net, args.bcs_batch)

    # 初始化PINN
    pinn = PINNAcoustic2d(acoustic_net, wavespeed_net, acoustic_gradfn_first, acoustic_gradfn_second_res, acoustic_gradfn_second_bcs)
    pinn.to_float(ms.float16)

    # 初始化优化器
    optimizer = nn.Adam(pinn.trainable_params(), learning_rate=args.lr)

    # 初始化训练器
    train_net = MyTrainOneStep(pinn, optimizer)

    # 训练
    pinn.set_train(mode=True)
    with open('train_infos.txt', 'w') as f:
        pass
    last_time = time.time()
    for i in range(args.epochs_Adam):
        loss = train_net(X_pde, X_init, X_S, X_BC_t, U_ini1x, U_ini1z, U_ini2x, U_ini2z, Sx, Sz)
        if (i + 1) % 100 == 0:
            current_time = time.time()
            used_time = current_time - last_time
            infos = ('iter: %.d, loss: %.3e, time: %d' % (i + 1, loss, used_time))
            with open('./train_infos.txt', 'a') as f:
                f.write(infos + '\n')
            print(infos)
            last_time = current_time
            X_pde = dataset.get_batch_res()
            X_BC_t = dataset.get_batch_bcs()
            if (i + 1) % 1000 == 0:
                ms.save_checkpoint(pinn.acoustic_net, "./acoustic_net.ckpt")
                ms.save_checkpoint(pinn.wavespeed_net, "./wavespeed_net.ckpt")


if __name__ == '__main__':
    train()
