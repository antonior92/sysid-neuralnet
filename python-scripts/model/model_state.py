import torch

from model.lstm import LSTM
from model.tcn import TCN
from model.dynamic_model import DynamicModel
import torch.optim as optim
import os.path


class ModelState:
    """
    Container for all model related parameters and optimizer

    model
    optimizer
    """

    def __init__(self, seed, cuda, nu, ny, optimizer, init_lr, model, model_options):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            if not cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        if model == 'lstm':
            nn_model = LSTM
        elif model == 'tcn':
            nn_model = TCN
        else:
            raise Exception("Model not implemented: {}".format(model))
        self.model = DynamicModel(nn_model, nu, ny, **model_options)

        if cuda:
            self.model.cuda()

        # Optimization parameters
        self.optimizer = getattr(optim, optimizer['optim'])(self.model.parameters(), lr=init_lr)

    def load_model(self, path, name='model.pth'):

        try:
            file = os.path.join(path, name)
            ckpt = torch.load(file)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            epoch = ckpt['epoch']
        except Exception as e:
            print(e)
            try:
                file = path
                ckpt = torch.load(file)
                self.model.load_state_dict(ckpt["model"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                epoch = ckpt['epoch']
            except Exception as e:
                raise Exception("Could not find model: " + path)
        return epoch

    def save_model(self, epoch, path, name='model.pth'):

        torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
            os.path.join(path, name))


