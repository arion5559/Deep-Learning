import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np


def train_encoder(train_set, test_set, nb_id, nb_data_per_id):
    sae = StackedAutoEncoder(nb_data_per_id)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(sae.parameters(), lr=0.02, weight_decay=0.25)
    nb_epoch = 800
    for epoch in range(1, nb_epoch + 1):
        train_loss = 0
        s = 0.
        for id_ in range(nb_id):
            input_id = Variable(train_set[id_]).unsqueeze(0)
            target = input_id.clone()
            if torch.sum(target.data > 0) > 0:
                output = sae.forward(input_id)
                target.require_grad = False
                output[target == 0] = 0
                loss = criterion(output, target)
                mean_corrector = nb_data_per_id/float(torch.sum(target.data > 0) + 1e-10)
                loss.backward()
                train_loss += np.sqrt(loss.data*mean_corrector)
                s += 1.
                optimizer.step()
        print(f"Epoch: {epoch}, loss: {train_loss/s}")
    torch.save(sae, r"StackedAutoEncoder.pt")


class StackedAutoEncoder(nn.Module):
    def __init__(self, nb):
        super(StackedAutoEncoder, self).__init__()
        self.fc1 = nn.Linear(nb, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 20)
        self.fc5 = nn.Linear(20, 40)
        self.fc6 = nn.Linear(40, nb)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.fc6(x)
        return x


def test_encoder(training_set, test_set, nb_id, nb_data_per_id):
    sae = torch.load(r"StackedAutoEncoder.pt")
    test_loss = 0
    s = 0.
    criterion = nn.MSELoss()
    for id_ in range(nb_id):
        input_id = Variable(training_set[id_]).unsqueeze(0)
        target = Variable(test_set[id_]).unsqueeze(0)
        if torch.sum(target.data > 0) > 0:
            output = sae.forward(input_id)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_data_per_id / float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            test_loss += np.sqrt(loss.data * mean_corrector)
            s += 1.
    print(f"Train loss: {test_loss / s}")
