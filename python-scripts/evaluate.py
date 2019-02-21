# %% Initialization
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tcn import TCN
from data_generation import chen_example

args = {'ksize': 7,
        'lr': 0.01,
        'cuda': False,
        'dropout': 0.8,
        'seed': 1111,
        'optim': 'Adam',
        'batch_size': 3,
        'log_interval': 10,
        'ar': True,
        'epochs': 1000}

torch.manual_seed(args['seed'])
if torch.cuda.is_available():
    if not args['cuda']:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# %% Prepare
# Problem specifications
nu = 1
ny = 1

# Producing data
U_train, Y_train = chen_example(1000, 10)
U_val, Y_val = chen_example(1000, 10)
U_test, Y_test = chen_example(1000, 10)

# Convert to pytorch tensors
U_train, Y_train = torch.tensor(U_train, dtype=torch.float), torch.tensor(Y_train, dtype=torch.float)
U_val, Y_val = torch.tensor(U_val, dtype=torch.float), torch.tensor(Y_val, dtype=torch.float)
U_test, Y_test = torch.tensor(U_test, dtype=torch.float), torch.tensor(Y_test, dtype=torch.float)


if args['ar']:  # Generate autoregressive model
    X_train = torch.cat((U_train, Y_train), 1)
    X_val = torch.cat((U_val, Y_val), 1)
    X_test = torch.cat((U_test, Y_test), 1)
    nx = nu + ny
else: # Generate "FIR" model
    X_train = U_train
    X_val = U_val
    X_test = U_test
    nx = nu

# Neural network
n_channels = [2, 4, 8, 16]
kernel_size = args['ksize']
dropout = args['dropout']
model = TCN(nx, ny, n_channels, kernel_size=kernel_size, dropout=dropout)

if args['cuda']:
    model.cuda()
    X_train = X_train.cuda()
    Y_train = Y_train.cuda()
    X_val = X_test.cuda()
    Y_val = Y_val.cuda()
    X_test = X_test.cuda()
    Y_test = Y_test.cuda()

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
    for i in range(0, X_train.size()[0], batch_size):
        if i + batch_size > X_train.size()[0]:
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i+batch_size)], Y_train[i:(i+batch_size)]
        optimizer.zero_grad()

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        processed = min(i + batch_size, X_train.size()[0])
        total_loss = i / processed * total_loss + x.size()[0] / processed * loss.item()

        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_train.size()[0], 100.*processed/X_train.size()[0], lr, total_loss))
            batch_idx = 0


for epoch in range(1, epochs+1):
    train(epoch)
