# %% Initialization
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tcn import TCN
from data_generation import chen_example
from model_eval import get_input

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

# Convert to pytorch tensors
u_train, y_train = torch.tensor(u_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float)
u_val, y_val = torch.tensor(u_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

if args['cuda']:
    u_train = u_train.cuda()
    y_train = y_train.cuda()
    u_val = u_val.cuda()
    y_val = y_val.cuda()

x_train = get_input(u_train, y_train, args['ar'])
nx = x_train.size()[1]

# Neural network
n_channels = [16, 32, 64, 128]
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


all_losses = []
best_vloss = 1e10
for epoch in range(1, epochs+1):
    loss = train(epoch)
    all_losses += [loss]
    if loss < best_vloss:  # Use validation loss here instead of training loss
        torch.save(model, 'best_model.pt')
        best_vloss = loss
