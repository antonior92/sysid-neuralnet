# %% Initialization
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path

def run_train(start_epoch, cuda, modelstate, logdir, loader_train, loader_valid, train_options ):

    # %% Train
    def validate():
        modelstate.model.eval()
        total_vloss = 0
        total_batches = 0
        for i, (u, y) in enumerate(loader_valid):
            if cuda:
                u = u.cuda()
                y = y.cuda()
            output = modelstate.model(u, y)
            vloss = nn.MSELoss()(output, y)
            total_batches += u.size()[0]
            total_vloss += u.size()[0]*vloss.item()

        return total_vloss/total_batches


    def train(epoch):
        modelstate.model.train()
        total_loss = 0
        total_batches = 0
        for i, (u, y) in enumerate(loader_train):
            if cuda:
                u = u.cuda()
                y = y.cuda()
            modelstate.optimizer.zero_grad()

            output = modelstate.model(u, y)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            modelstate.optimizer.step()

            total_batches += u.size()[0]
            total_loss += u.size()[0] * loss.item()

            #Extract learning rate
            lr = modelstate.optimizer.param_groups[0]["lr"]

            if i % train_options['log_interval'] == 0:
                print('Train Epoch: {:5d} [{:6d}/{:6d} ({:3.0f}%)]\tLearning rate: {:.6f}\tLoss: {:.6f}'.format(
                    epoch, i, len(loader_train), 100. * i / len(loader_train),
                    lr, total_loss/total_batches))

        return total_loss/total_batches


    all_losses = []
    all_vlosses = []
    best_vloss = 1e10
    epoch = start_epoch
    for epoch in range(start_epoch, start_epoch + train_options["epochs"]+1):
        # Train and validate
        loss = train(epoch)
        vloss = validate()
        # Save losses
        all_losses += [loss]
        all_vlosses += [vloss]
        if vloss < best_vloss:
            modelstate.save_model(epoch, logdir, 'best_model.pt')
            best_vloss = vloss

        #Extract learning rate
        lr = modelstate.optimizer.param_groups[0]["lr"]

        # Print validation results
        print('-'*100)
        print('Train Epoch: {:5d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.6f}\tLoss: {:.6f}\tVal Loss: {:.6f}'.
              format(epoch, len(loader_train), len(loader_train), 100., lr, loss, vloss))
        print('-'*100)
        # lr scheduler
        if len(all_vlosses) > train_options['lr_scheduler_nepochs'] and \
                vloss > max(all_vlosses[-train_options['lr_scheduler_nepochs']-1:-1]):
            lr = lr / train_options['lr_scheduler_factor']
            for param_group in modelstate.optimizer.param_groups:
                param_group['lr'] = lr
        # Early stoping
        if lr < train_options['min_lr']:
            break

    modelstate.save_model(epoch, logdir, 'final_model.pt')
