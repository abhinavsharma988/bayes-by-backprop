import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("/Users/abhinav/Documents/GitHub/bayes-by-backprop/BBB")

from BNNLayer import BNNLayer
from BNN import BNN

import torch
#from torch.autograd import Variable

plt.style.use('seaborn-v0_8-paper')

x = np.random.uniform(-4, 4, size=200).reshape((-1, 1))
noise = np.random.normal(0, 9, size=200).reshape((-1, 1))
y = x ** 3 + noise

#Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype))
#X = Var(x)
#Y = Var(y)

X = torch.from_numpy(x).float()#.requires_grad_(True)
Y = torch.from_numpy(y).float()

# Initialize network
bnn = BNN(BNNLayer(1, 50, activation='tanh', prior_mean=0, prior_rho=0),
          BNNLayer(50, 1, activation='none', prior_mean=0, prior_rho=0))

optim = torch.optim.Adam(bnn.parameters(), lr=1e-1)
loss_history = []
# Main training loop
for i_ep in range(2000):
    print(i_ep)
    kl, lg_lklh = bnn.Forward(X, Y, 1, 'Gaussian')
    loss = BNN.loss_fn(kl, lg_lklh, 1)
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    loss_history.append(loss.item())

# Plotting
plt.scatter(x, y, c='navy', label='target')
x_ = np.linspace(-5, 5)
y_ = x_ ** 3
# Replace Var(x_) with proper tensor conversion
X_ = torch.from_numpy(x_).float().unsqueeze(1)
# Replace .data.numpy() with .detach().numpy()
with torch.no_grad():
    pred_lst = [
        bnn.forward(X_, mode='MC').detach().numpy().squeeze(1)
        for _ in range(2000)
    ]

pred = np.array(pred_lst).T
pred_mean = pred.mean(axis=1)
pred_std = pred.std(axis=1)
plt.plot(x_, pred_mean, c='royalblue', label='mean pred')
plt.fill_between(
    x_,
    pred_mean - 3 * pred_std,
    pred_mean + 3 * pred_std,
    color='cornflowerblue',
    alpha=0.5,
    label='+/- 3 std'
)

plt.plot(x_, y_, c='grey', label='truth')
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()   # create new plot window
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
