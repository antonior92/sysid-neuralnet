# %% Initialization
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tcn import TCN

args = {'ksize': 7,
        'lr': 0.01,
        'cuda': False,
        'dropout': 0.8,
        'seed': 1111,
        'optim': 'Adam',
        'batch_size': 1,
        'log_interval': 10}

torch.manual_seed(args['seed'])
if torch.cuda.is_available():
    if not args['cuda']:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# %% Prepare
# Problem specifications
num_inputs = 1
num_outputs = 1

# Producing data
X_train, Y_train = torch.rand(5, num_inputs, 1000), torch.rand(5, num_outputs, 1000)
X_val, Y_val = torch.rand(2, num_inputs, 100), torch.rand(2, num_outputs, 100)
X_test, Y_test = torch.rand(2, num_inputs, 100), torch.rand(3, num_outputs, 100)

# Neural network
num_channels = [2, 4, 8, 16]
kernel_size = args['ksize']
dropout = args['dropout']
model = TCN(num_inputs, num_outputs, num_channels, kernel_size=kernel_size, dropout=dropout)

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


def train(epoch):
    model.train()
    total_loss = 0
    for i in range(0, X_train.size()[0], batch_size):
        if i + batch_size > X_train.size()[0]:
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i+batch_size)], Y_train[i:(i+batch_size)]
        optimizer.zero_grad()

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        processed = min(i + batch_size, X_train.size()[0])
        total_loss = i /processed * total_loss + x.size()[0] / processed *loss.item()

        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_train.size()[0], 100.*processed/X_train.size()[0], lr, total_loss))


for epoch in range(1, epochs+1):
    train(epoch)
    test()
