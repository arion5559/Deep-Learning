import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import preprocessing
from joblib import dump, load
import numpy as np


def train_machine(training_set, nb_id):
    nv = len(training_set[0])
    nh = 200
    batch_size = 50
    rbm = RestrictedBoltzmannMachine(nv, nh)

    nb_epoch = 50
    for epoch in range(1, nb_epoch + 1):
        training_loss = 0
        s = 0.0
        for id_ in range(0, nb_id - batch_size, batch_size):
            vk = training_set[id_:id_ + batch_size]
            v0 = vk
            ph0, _ = rbm.sample_h(v0)
            for k in range(20):
                _, hk = rbm.sample_h(vk)
                _, vk = rbm.sample_v(hk)
                vk[v0 < 0] = v0[v0 < 0]
            phk, _ = rbm.sample_h(vk)
            rbm.train(v0, vk, ph0, phk)
            training_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
            s += 1.0
        print(f"Epoch: {epoch}, loss: {training_loss/s}")
    torch.save(rbm, r"RestrictedBoltzmannMachine.pt")


class RestrictedBoltzmannMachine:
    def __init__(self, nv, nh):
        self.w = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)

    def sample_h(self, x):  # x = mini_batch_size * nv
        wx = torch.mm(x, self.w.t())  # mini_batch_size * nh
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, x):
        wx = torch.mm(x, self.w)
        activation = wx + self.b.expand_as(wx)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        self.w += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)


def test_machine(training_set, test_set, nb_id):
    rbm = torch.load(r"RestrictedBoltzmannMachine.pt")
    testing_loss = 0
    s = 0.0
    final_v = []
    final_vt = []
    for id_ in range(0, nb_id):
        v = training_set[id_:id_ + 1]
        vt = test_set[id_:id_ + 1]
        if len(vt[vt >= 0]) > 0:
            _, h = rbm.sample_h(v)
            _, v = rbm.sample_v(h)
            testing_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.0
        final_v.append(v[0])
        final_vt.append(vt[0])
    print(f"Testing loss: {testing_loss/s}")
    print(final_v)
    print(final_vt)
