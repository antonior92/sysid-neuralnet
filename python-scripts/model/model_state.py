import torch

from model import LSTM, MLP, TCN, DynamicModel
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

        self.model = DynamicModel(model, nu, ny, **model_options)

        if cuda:
            self.model.cuda()

        # Optimization parameters
        self.optimizer = getattr(optim, optimizer['optim'])(self.model.parameters(), lr=init_lr)

    def load_model(self, path, name='model.pt'):
        try:
            file = os.path.join(path, name)
            ckpt = torch.load(file, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            epoch = ckpt['epoch']
        except NotADirectoryError as e:
            try:
                file = path
                ckpt = torch.load(file, map_location=lambda storage, loc: storage)
                self.model.load_state_dict(ckpt["model"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                epoch = ckpt['epoch']
            except NotADirectoryError as e:
                raise Exception("Could not find model: " + path)
        return epoch

    def save_model(self, epoch, vloss, elapsed_time,  path, name='model.pt'):

        torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'vloss': vloss,
                'elapsed_time': elapsed_time,
            },
            os.path.join(path, name))

