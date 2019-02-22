# %% Initialization
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tcn import TCN
from data_generation.data_generator import DataLoaderExt
from data_generation.chen_example import ChenDataset
from data_generation.silver_box import SilverBoxDataset


from model_eval import get_input, one_step_ahead

args = {'ksize': 3,
        'lr': 0.001,
        'cuda': False,
        'dropout': 0.8,
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
# Problem specifications


# Producing data


if args['dataset'] == 'Chen':
    options = args['chen_options']
    loader_train = DataLoaderExt(ChenDataset(seq_len=options['seq_len'], **options['train']),
                                 batch_size=args["batch_size"], shuffle=True, num_workers=4)
    loader_valid = DataLoaderExt(ChenDataset(seq_len=options['seq_len'], **options['valid']),
                                 batch_size=args["eval_batch_size"], shuffle=False, num_workers=4)
elif args['dataset'] == 'SilverBox':
    options = args['silverbox_options']
    loader_train = DataLoaderExt(SilverBoxDataset(**options, split = 'train'),
                                 batch_size=args["batch_size"], shuffle=False, num_workers=4)
    loader_valid = DataLoaderExt(SilverBoxDataset(**options, split='valid'),
                                 batch_size=args["eval_batch_size"], shuffle=False, num_workers=4)
else:
    raise Exception("Dataset not implemented: {}".format(args['dataset']))

nu = loader_train.data_shape[0][0] # first dimension of u
ny = loader_train.data_shape[1][0] # first dimension of y

if args["ar"]:
    nx = nu + ny
else:
    nx = nu

# Neural network
n_channels = [16, 32]
dilation_sizes = [1, 1, 1]
kernel_size = args['ksize']
dropout = args['dropout']
model = TCN(nx, ny, n_channels, kernel_size=kernel_size, dilation_sizes=dilation_sizes, dropout=dropout)
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
    for i, (u, y) in enumerate(loader_valid):
        if args['cuda']:
            u = u.cuda()
            y = y.cuda()

        x = get_input(u, y, args['ar'])

        output = model(x)
        vloss = nn.MSELoss()(output, y)
        processed = min(i + batch_size, len(loader_valid.dataset))
        total_vloss = i / processed * total_vloss + x.size()[0] / processed * vloss.item()

    return total_vloss


def train(epoch):
    model.train()
    total_loss = 0
    batch_idx = 0
    for i, (u, y) in enumerate(loader_train):
        if args['cuda']:
            u = u.cuda()
            y = y.cuda()

        x = get_input(u, y, args['ar'])

        optimizer.zero_grad()

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()


        processed = min(i + batch_size, len(loader_train.dataset))
        total_loss = i / processed * total_loss + x.size()[0] / processed * loss.item()

        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {:5d} [{:6d}/{:6d} ({:3.0f}%)]\tLearning rate: {:.6f}\tLoss: {:.6f}'.format(
                epoch, processed, len(loader_train.dataset), 100. * processed / len(loader_train.dataset), lr, total_loss))
        batch_idx += 1

    return total_loss


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
            epoch, len(loader_train.dataset), len(loader_train.dataset), 100., lr, loss, vloss))
    print('-'*100)
    # lr scheduler
    if epoch > args['lr_scheduler_nepochs'] and vloss > max(all_vlosses[-args['lr_scheduler_nepochs']-1:-1]):
        print('bfaf') ## ????
        lr = lr / args['lr_scheduler_factor']
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # Early stoping
    if lr < args['min_lr']:
        break
