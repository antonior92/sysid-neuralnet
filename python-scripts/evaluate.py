# %% Initialization
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tcn import TCN
from data_generation import chen_example

args = {'ksize': 7,
        'lr': 0.001,
        'cuda': False,
        'dropout': 0.8,
        'seed': 1111,
        'optim': 'Adam',
        'batch_size': 3,
        'log_interval': 10,
        'ar': True,
        'epochs': 1000,
        'plot': True,
        'plotly': True}

torch.manual_seed(args['seed'])
if torch.cuda.is_available():
    if not args['cuda']:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# %% Prepare
# Problem specifications
nu = 1
ny = 1

# Producing data
u_train, y_train = chen_example(1000, 10)
u_val, y_val = chen_example(1000, 10)
u_test, y_test = chen_example(1000, 2)

# Convert to pytorch tensors
u_train, y_train = torch.tensor(u_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float)
u_val, y_val = torch.tensor(u_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
u_test, y_test = torch.tensor(u_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)
if args['cuda']:
    u_train = u_train.cuda()
    y_train = y_train.cuda()
    u_val = u_test.cuda()
    y_val = y_val.cuda()
    u_test = u_test.cuda()
    y_test = y_test.cuda()

if args['ar']:  # Generate autoregressive model
    y_train = torch.cat((y_train[:, :, 1:], torch.zeros_like(y_train[:, :, 0:1])), -1)
    x_train = torch.cat((u_train, y_train), 1)
    nx = nu + ny
else: # Generate "FIR" model
    x_train = u_train
    nx = nu

# Neural network
n_channels = [2, 4, 8, 16]
kernel_size = args['ksize']
dropout = args['dropout']
model = TCN(nx, ny, n_channels, kernel_size=kernel_size, dropout=dropout)
if args['cuda']:
    model.cuda()


# Optimization parameters
lr = args['lr']
batch_size = args['batch_size']
epochs = args['epochs']
optimizer = getattr(optim, args['optim'])(model.parameters(), lr=lr)


# %% Train
def train(epoch):
    model.train()
    total_loss = 0
    batch_idx = 0
    for i in range(0, x_train.size()[0], batch_size):
        if i + batch_size > x_train.size()[0]:
            x, y = x_train[i:], y_train[i:]
        else:
            x, y = x_train[i:(i + batch_size)], y_train[i:(i + batch_size)]
        optimizer.zero_grad()

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        processed = min(i + batch_size, x_train.size()[0])
        total_loss = i / processed * total_loss + x.size()[0] / processed * loss.item()

        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, x_train.size()[0], 100. * processed / x_train.size()[0], lr, total_loss))
            batch_idx = 0

    return total_loss


def one_step_ahead(u, y):
    model.eval()

    if args['ar']:
        x = torch.cat((u, y), 1)
    else:
        x = u

    y_pred = model(x)
    return y_pred

for epoch in range(1, epochs+1):
    train(epoch)


# %% Test
if args['plot']:
    import matplotlib.pyplot as plt
    for i in range(0, u_test.size()[0]):
        fig, ax = plt.subplots()
        y_pred = one_step_ahead(u_test, y_test)
        ax.plot(y_test[i, 0, :].detach().numpy(), color='b', label='y true')
        ax.plot(y_pred[i, 0, :].detach().numpy(), color='g', label='y_pred')

        if args['plotly']:
            import plotly.tools as tls
            import plotly.offline as py
            plotly_fig = tls.mpl_to_plotly(fig)
            py.plot(plotly_fig)
        else:
            plt.show()


