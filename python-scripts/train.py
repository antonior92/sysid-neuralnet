# %% Initialization
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model.tcn import TCN
from model.lstm import LSTM
from model.dynamic_model import DynamicModel
from data_generation.data_generator import DataLoaderExt
from data_generation.chen_example import ChenDataset
from data_generation.silver_box import SilverBoxDataset


args = {'lr': 0.001,
        'cuda': False,
        'seed': 1111,
        'optim': 'Adam',
        'batch_size': 3,
        'eval_batch_size': 10,
        'log_interval': 1,
        'ar': True,
        'epochs': 1000,
        'plot': True,
        'plotly': True,
        'lr_scheduler_nepochs': 10,
        'lr_scheduler_factor': 10,
        'dataset': "SilverBox",
        'model': 'tcn',
        'tcn_options':
        {
        'num_channels': [16, 32],
        'dilation_sizes': [1, 1],
        'kernel_size': 3,
        'dropout': 0.8,
        },
        'lstm_options':
        {
            'hidden_size': 5
        },
        'chen_options':
        {
            'seq_len': 1000,
            'train': {
                      'ntotbatch': 10,
                      'seed': 1
                     },
            'valid': {
                      'ntotbatch': 10,
                      'seed': 2
                     }
        },
        'silverbox_options': {'seq_len': 1000},
        'min_lr': 1e-6}

torch.manual_seed(args['seed'])
if torch.cuda.is_available():
    if not args['cuda']:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# %% Prepare

# Specifying datasets
if args['dataset'] == 'Chen':
    options = args['chen_options']
    loader_train = DataLoaderExt(ChenDataset(seq_len=options['seq_len'], **options['train']),
                                 batch_size=args["batch_size"], shuffle=True, num_workers=4)
    loader_valid = DataLoaderExt(ChenDataset(seq_len=options['seq_len'], **options['valid']),
                                 batch_size=args["eval_batch_size"], shuffle=False, num_workers=4)
elif args['dataset'] == 'SilverBox':
    options = args['silverbox_options']
    loader_train = DataLoaderExt(SilverBoxDataset(**options, split='train'),
                                 batch_size=args["batch_size"], shuffle=False, num_workers=4)
    loader_valid = DataLoaderExt(SilverBoxDataset(**options, split='valid'),
                                 batch_size=args["eval_batch_size"], shuffle=False, num_workers=4)
else:
    raise Exception("Dataset not implemented: {}".format(args['dataset']))

# Problem specifications
nu = loader_train.data_shape[0][0]  # first dimension of u, ie. # channels of u
ny = loader_train.data_shape[1][0]  # first dimension of y, ie. # channels of y

# Neural network
if args['model'] == 'lstm':
    model = LSTM
    model_options = args['lstm_options']
elif args['model'] == 'tcn':
    model = TCN
    model_options = args['tcn_options']
else:
    raise Exception("Model not implemented: {}".format(args['model']))
model = DynamicModel(model, nu, ny, args['ar'], **model_options)

if args['cuda']:
    model.cuda()


# Optimization parameters
lr = args['lr']
batch_size = args['batch_size']
epochs = args['epochs']
optimizer = getattr(optim, args['optim'])(model.parameters(), lr=lr)


# %% Train
def validate():
    model.eval()
    total_vloss = 0
    total_batches = 0
    for i, (u, y) in enumerate(loader_valid):
        if args['cuda']:
            u = u.cuda()
            y = y.cuda()
        output = model(u, y)
        vloss = nn.MSELoss()(output, y)
        total_batches += u.size()[0]
        total_vloss += u.size()[0]*vloss.item()

    return total_vloss/total_batches


def train(epoch):
    model.train()
    total_loss = 0
    total_batches = 0
    for i, (u, y) in enumerate(loader_train):
        if args['cuda']:
            u = u.cuda()
            y = y.cuda()
        optimizer.zero_grad()

        output = model(u, y)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()

        total_batches += u.size()[0]
        total_loss += u.size()[0] * loss.item()

        if i % args['log_interval'] == 0:
            print('Train Epoch: {:5d} [{:6d}/{:6d} ({:3.0f}%)]\tLearning rate: {:.6f}\tLoss: {:.6f}'.format(
                epoch, i, len(loader_train), 100. * i / len(loader_train), lr, total_loss/total_batches))

    return total_loss/total_batches


all_losses = []
all_vlosses = []
best_vloss = 1e10
for epoch in range(1, epochs+1):
    # Train and validate
    loss = train(epoch)
    vloss = validate()
    # Save losses
    all_losses += [loss]
    all_vlosses += [vloss]
    if vloss < best_vloss:
        torch.save(model, 'best_model.pt')
        best_vloss = vloss
    # Print validation results
    print('-'*100)
    print('Train Epoch: {:5d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.6f}\tLoss: {:.6f}\tVal Loss: {:.6f}'.format(
            epoch, len(loader_train), len(loader_train), 100., lr, loss, vloss))
    print('-'*100)
    # lr scheduler
    if epoch > args['lr_scheduler_nepochs'] and vloss > max(all_vlosses[-args['lr_scheduler_nepochs']-1:-1]):
        lr = lr / args['lr_scheduler_factor']
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # Early stoping
    if lr < args['min_lr']:
        break
