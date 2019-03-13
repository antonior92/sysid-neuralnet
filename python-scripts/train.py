# %% Initialization
import torch.nn as nn
import torch.nn.functional as F
import torch
import time


def run_train(start_epoch, cuda, modelstate, logdir, loader_train, loader_valid, train_options):
    def validate(loader):
        modelstate.model.eval()
        total_vloss = 0
        total_batches = 0
        with torch.no_grad():
            for i, (u, y) in enumerate(loader):
                if cuda:
                    u = u.cuda()
                    y = y.cuda()
                output = modelstate.model(u, y)
                vloss = F.mse_loss(output, y)
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
            loss = F.mse_loss(output, y)
            loss.backward()
            modelstate.optimizer.step()

            total_batches += u.size()[0]
            total_loss += u.size()[0] * loss.item()

            # Extract learning rate
            lr = modelstate.optimizer.param_groups[0]["lr"]

            if i % train_options['log_interval'] == 0:
                print('Train Epoch: {:5d} [{:6d}/{:6d} ({:3.0f}%)]\tLearning rate: {:.6f}\tLoss: {:.6f}'.format(
                    epoch, i, len(loader_train), 100. * i / len(loader_train),
                    lr, total_loss/total_batches))

        return total_loss/total_batches

    if torch.cuda.is_available():
        if not cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")



    try:
        modelstate.model.set_mode(train_options['training_mode'])
        # Train
        epoch = start_epoch
        vloss = validate(loader_valid)
        all_losses = []
        all_vlosses = []
        best_vloss = vloss
        start_time = time.clock()
        for epoch in range(start_epoch, start_epoch + train_options["epochs"]+1):
            # Train and validate
            train(epoch)
            vloss = validate(loader_valid)
            loss = validate(loader_train)
            # Save losses
            all_losses += [loss]
            all_vlosses += [vloss]
            if vloss < best_vloss:
                best_vloss = vloss
                modelstate.save_model(epoch, best_vloss, time.clock() - start_time, logdir, 'best_model.pt')

            # Extract learning rate
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
        modelstate.save_model(epoch, vloss, time.clock() - start_time, logdir, 'final_model.pt')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        modelstate.save_model(epoch, vloss, time.clock() - start_time, logdir, 'interrupted_model.pt')
        print('-' * 89)
