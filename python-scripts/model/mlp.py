import torch.nn as nn
from .utils import RunMode
from collections import OrderedDict


class MLP(nn.Module):
    """Shallow 2-layer neural network with sigmoidal activation function."""
    def __init__(self, num_inputs, num_outputs, hidden_size, max_past_input):
        super(MLP, self).__init__()
        self.receptive_field = max_past_input

        self.pad = nn.ConstantPad1d((max_past_input-1, 0), 0)

        conv1 = nn.Conv1d(num_inputs, hidden_size, kernel_size=max_past_input, padding=0)

        sigmoid = nn.Sigmoid()
        conv2 = nn.Conv1d(hidden_size, num_outputs, 1)

        self.net = nn.Sequential(OrderedDict([('0', conv1),
                                             ("2", sigmoid),
                                             ("3", conv2)]))

    def set_mode(self, mode):
        if mode == RunMode.FREE_RUN_SIMULATION:
            self.pad.padding = (0, 0)
        else:
            self.pad.padding = (self.receptive_field-1, 0)

    def forward(self, x):
        x = self.pad(x)
        return self.net(x)
