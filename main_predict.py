import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import time
import joblib

import torch
from fc import FC
from net import Model
from torch import nn
from torch.nn import init
from utils import get_norm
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler


def predict(net, input, label):
    # return net.model(input), label
    print(input)
    x = net.model.fc[0](input)
    print(x)
    x = net.model.fc[1](x)
    print(x)
    x = net.model.fc[2](x)

    return x, label


def cal(pred, target):
    n = 0
    for i in range(len(pred[0])):
        err = abs(pred[0][i] - target[0][i])
        if err < 0.05:
            print(i, "********")
            n += 1
        # elif err > 1:
        #     print(i, "*")
    print("num:", n)


if __name__ == '__main__':
    num_inputs, num_outputs, num_hiddens = 926, 618, 256
    batch_size, num_epochs = 64, 40
    scale = StandardScaler()

    root_path = "/home/rr/Downloads/nsm_data/bone_gating_WalkTrain/"
    loss_func = torch.nn.MSELoss()

    inputs_list = os.listdir(root_path + "Input/")
    inputs_list.sort(key=lambda x: int(x[:-4]))

    net = Model(num_inputs, num_hiddens, num_outputs, batch_size)

    net.model.load_state_dict(
        torch.load(os.path.join("../models", "fcn_30.pth"), map_location=torch.device("cuda:0")))
    net.optimizer.load_state_dict(
        torch.load(os.path.join("../models", "fcn_opt_30.pth"), map_location=torch.device("cuda:0")))
    net.model.eval()

    input_data = pd.read_csv(root_path + "Input/" + "1.txt", sep=' ', header=None, dtype=float)
    label_data = pd.read_csv(root_path + "Label/" + "1.txt", sep=' ', header=None, dtype=float)

    # scale 标准化
    # scale = scale.fit(input_data)
    # input_data = torch.Tensor(scale.transform(input_data))

    # 手动 标准化
    input_mean, input_std = get_norm("/home/rr/Downloads/nsm_data/bone_gating_WalkTrain/InputNorm.txt")
    input_mean, input_std = get_norm("/home/rr/Downloads/nsm_data/utils/inputNorm.txt")
    input_mean, input_std = input_mean[0:926], input_std[0:926]
    input_data = torch.Tensor((np.array(input_data).astype('float32') - input_mean) / input_std)

    # input_data = torch.Tensor(np.array(input_data))
    label_data = torch.Tensor(np.array(label_data))

    input_data = Variable(input_data.type(torch.FloatTensor).to(torch.device("cuda:0")))
    label_data = Variable(label_data.type(torch.FloatTensor).to(torch.device("cuda:0")))

    pred, target = predict(net, input_data[-1:], label_data[-1:])
    loss = loss_func(pred, target).sum()

    print(loss)

