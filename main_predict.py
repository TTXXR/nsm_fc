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
from utils.utils import get_norm
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler


def predict(net, input, label, seq_flag=False):
    if not seq_flag:
        return net.model(input), label
    else:
        out = []
        x = input[0]
        for i in range(len(input)):
            x = net.model(x)
            out.append(x)
            print(x)

        return out, label


if __name__ == '__main__':
    root_path = "/home/rr/Downloads/nsm_data/bone_gating_WalkTrain/"
    loss_func = torch.nn.MSELoss()
    num_inputs, num_outputs, num_hiddens = 926, 618, 256
    batch_size, num_epochs = 64, 40

    inputs_list = os.listdir(root_path + "Input/")
    inputs_list.sort(key=lambda x: int(x[:-4]))

    net = Model(num_inputs, num_hiddens, num_outputs, batch_size)
    net.model.load_state_dict(
        torch.load(os.path.join("models", "fcn_fixedScale_30.pth"), map_location=torch.device("cuda:0")))
    net.optimizer.load_state_dict(
        torch.load(os.path.join("models", "fcn_fixedScale_opt_30.pth"), map_location=torch.device("cuda:0")))
    net.model.eval()

    input_data = pd.read_csv(root_path + "Input/" + "1.txt", sep=' ', header=None, dtype=float)
    label_data = pd.read_csv(root_path + "Label/" + "1.txt", sep=' ', header=None, dtype=float)

    # scale 标准化
    # scale = StandardScaler()
    # scale = scale.fit(input_data)
    # input_data = torch.Tensor(scale.transform(input_data))

    # 手动 标准化
    input_mean, input_std = get_norm("/home/rr/Downloads/nsm_data/utils/inputNorm.txt")
    input_mean, input_std = input_mean[0:926], input_std[0:926]
    input_data = torch.Tensor((np.array(input_data).astype('float32') - input_mean) / input_std)

    label_data = torch.Tensor(np.array(label_data))
    input_data = Variable(input_data.type(torch.FloatTensor).to(torch.device("cuda:0")))
    label_data = Variable(label_data.type(torch.FloatTensor).to(torch.device("cuda:0")))

    # single test
    pred, target = predict(net, input_data[-10:], label_data[-10:])
    loss = loss_func(pred, target).sum()
    print(loss)

    # sequence test
    pred, target = predict(net, input_data[-10:], label_data[-10:], True)
    loss = loss_func(pred, target).sum()
    print(loss)
