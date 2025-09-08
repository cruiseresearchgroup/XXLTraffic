'''
STGCN
'''
import math
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x



class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(temporal_conv_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)


class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c, Lk):
        super(spatio_conv_layer, self).__init__()
        self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        return torch.relu(x_gc + x)


class st_conv_block(nn.Module):
    def __init__(self, ks, kt, n, c, p, Lk):
        super(st_conv_block, self).__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
        self.sconv = spatio_conv_layer(ks, c[1], Lk)
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])

        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)


class output_layer(nn.Module):
    def __init__(self, c, T, n, TIMESTEP_OUT):
        super(output_layer, self).__init__()
        self.tconv1 = temporal_conv_layer(T, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        # 一次输出12 * 3个
        self.fc = nn.Conv2d(c, TIMESTEP_OUT, 1)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)


class Model(nn.Module):
    def __init__(self, configs, ks=3, kt=3, bs= [[1, 16, 64], [64, 16, 64]], T=96, p=0):
        super(Model, self).__init__()
        self.n = configs.enc_in
        self.TIMESTEP_OUT = configs.pred_len
        adj = pd.read_csv(configs.adj_path).values
        W = weight_matrix(adj)
        L = scaled_laplacian(W)
        Lk = cheb_poly(L, ks)
        self.Lk = torch.Tensor(Lk.astype(np.float32)).cuda()
        self.st_conv1 = st_conv_block(ks, kt, self.n, bs[0], p, self.Lk)
        self.st_conv2 = st_conv_block(ks, kt, self.n, bs[1], p, self.Lk)
        self.output = output_layer(bs[1][2], T - 4 * (kt - 1), self.n, self.TIMESTEP_OUT)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = x.unsqueeze(-1)#btn>btnc
        # print('x shape ', x.shape)
        # x : b,t,n,c
        x = x.permute(0, 3, 1, 2)
        x_st1 = self.st_conv1(x)
        x_st2 = self.st_conv2(x_st1)
        output_1 = self.output(x_st2)
        # print("output1 shape is:", output_1.shape)
        output_2 = output_1.reshape(-1, self.TIMESTEP_OUT, 1, output_1.shape[3]).permute(0, 1, 3, 2)
        # print("output2 shape is:", output_2.shape)
        output_2 = output_2.squeeze(-1)
        # print('out shape ', output_2.shape)
        return output_2  # [b,t=36,c=1,n] -> [b,t=12,n,q=3]  torch.reshape


def weight_matrix(W, sigma2=0.1, epsilon=0.5):
    '''
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    n = W.shape[0]
    W = W / 100000
    W[W == 0] = np.inf
    W2 = W * W
    W_mask = (np.ones([n, n]) - np.identity(n))
    return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask


def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)


def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)

