from torch.nn import functional as F
from torch import nn
from cProfile import label
from statistics import mode
from urllib.request import proxy_bypass
import pandas
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils import *


def load_data(path="./dataclassif.csv"):
    """
    The load_data function loads the data from a CSV file and returns four arrays:
        tt - an array of treatement values (1 or 0)
        Y_f - an array of factual outcomes (0 or 1)  
        Y_cf - an array of  counterfactuals outcomes (0 or 1)
        pi_f - probability of factual outcome Pr(Y_f=1)
        pi_cf - probability of counterfactual outcome Pr(Y_cf=1)

    :param path="./dataclassif.csv": Used to indicate the path of the file to load.
    :return: a tuple of four numpy arrays.
        """
    df = pd.read_csv(path)
    df = reduce_mem_usage(df)
    tt = df['tt'].values
    Y_f = df['Y_f'].values
    Y_cf = df['Y_cf'].values
    pi_f = df['pi_f'].values
    pi_cf = df['pi_cf'].values
    df = df.drop(['tt', 'Y_cf', 'Y_f', 'Y_0', 'Y_1',
                 'pi_f', 'pi_cf', 'pi_0', 'pi_1'], 1)
    X = df.values
    return X, tt, Y_f, Y_cf, pi_f, pi_cf

#sc = StandardScaler()
#x = sc.fit_transform(x)

# defining dataset class


class dataset(Dataset):
    def __init__(self, x, y, tt):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.tt = torch.tensor(tt, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.tt[idx]

    def __len__(self):
        return self.length


x, tt, Y_f, Y_cf, pi_f, pi_cf = load_data()
trainset = dataset(x, Y_f, tt)
# DataLoader
trainloader = DataLoader(trainset, batch_size=64, shuffle=False)
# defining the network


class Net(nn.Module):
    def __init__(self, in_features, encoded_features):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(in_features, 32),  nn.LeakyReLU(),
            nn.Linear(32, 32),  nn.ReLU(),
            nn.Linear(32, 28),  nn.LeakyReLU(),
            nn.Linear(28, encoded_features),
        )

        self.psi = nn.Sequential(
            nn.Linear(encoded_features + 1, 128), nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 50),  nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, x, tt):
        tt = tt.reshape(-1, 1)
        phi = self.phi(x)
        x_t = torch.cat((phi, tt), 1)
        return nn.Sigmoid()(self.psi(x_t))


# hyper parameters
learning_rate = 0.01
epochs = 100
# Model , Optimizer, Loss
model = Net(in_features=x.shape[1], encoded_features=5)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()


# forward loop
losses = []
accur = []
accur_cf = []
accur_f = []
kl_list = []
for i in range(epochs):
    for j, (x_train, y_train, tt_train) in enumerate(trainloader):
        # calculate output
        output = model(x_train, tt_train)
        # calculate loss
        loss = loss_fn(output, y_train.reshape(-1, 1))

        def pred(model, x, y, tt, pi):
            pi_pred = model(torch.tensor(x, dtype=torch.float32),
                            torch.tensor(tt, dtype=torch.float32))
            y_pred = pi_pred.reshape(-1).detach().round()
            acc = (pi_pred.reshape(-1).detach().numpy().round() == y).mean()
            pi_pred_cc = pi_pred.clone().detach().float().squeeze()
            pi_pred_cc = np.asarray(pi_pred_cc, dtype=np.float64)
            pi_cc = np.asarray(pi.squeeze(), dtype=np.float64)
            pi_pred_cc = pi_pred_cc/pi_pred_cc.sum()
            pi_cc = pi_cc/pi_cc.sum()
            KL = np.sum(np.where(pi_pred_cc * pi_cc > 0,
                        pi_pred_cc*np.log(pi_pred_cc/pi_cc), 0))
            return pi_pred, y_pred, acc, KL

        pi_pred_f, y_pred_f, acc_f, KL_f = pred(model, x, Y_f, tt, pi_f)
        pi_pred_cf, y_pred_cf, acc_cf, KL_cf = pred(
            model, x, Y_cf, 1-tt, pi_cf)
        # np.sum(pi_f_cc*np.log(pi_pred_cc/pi_f_cc))
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i % 1 == 0:
        losses.append(loss)
        accur.append((acc_f+acc_cf)/2)
        accur_cf.append(acc_cf)
        accur_f.append(acc_f)
        kl_list.append((KL_f+KL_cf)/2)
        print("epoch {}\tloss : {}\t accuracy : {},{}".format(
            i, loss, acc_f, acc_cf))
        print("KL : {},{}".format(KL_f, KL_cf))

# detach losses from tensor
losses = torch.stack(losses).detach().numpy()
# plot losses
plt.plot(losses, label='loss')
plt.plot(kl_list, label='KL')
plt.plot(accur, label='accuracy')
plt.plot(accur_cf, label='accuracy_cf')
plt.plot(accur_f, label='accuracy_f')
plt.legend()
plt.show()
