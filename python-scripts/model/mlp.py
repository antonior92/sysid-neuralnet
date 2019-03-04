import torch.nn as nn
from .utils import RunMode, DynamicModule
from collections import OrderedDict


class MLP(DynamicModule):
    """Shallow 2-layer neural network with sigmoidal activation function."""
    def __init__(self, num_inputs, num_outputs, hidden_size, max_past_input):
        super(MLP, self).__init__()
        self.receptive_field = max_past_input

        self.pad = nn.ConstantPad1d((self.receptive_field - 1, 0), 0)

        conv1 = nn.Conv1d(num_inputs, hidden_size, kernel_size=max_past_input, padding=0)

        sigmoid = nn.Sigmoid()
        conv2 = nn.Conv1d(hidden_size, num_outputs, 1)

        # Due to backward compatibility the modules are named as follows
        self.net = nn.Sequential(OrderedDict([('0', conv1),
                                             ("2", sigmoid),
                                             ("3", conv2)]))

    def set_mode(self, mode):
        self.mode = mode
        if mode == RunMode.ONE_STEP_AHEAD:
            self.pad.padding = (self.receptive_field - 1, 0)

    def forward(self, x):
        if self.mode == RunMode.FREE_RUN_SIMULATION:
            seq_len = x.size()[-1]
            padding = max(self.receptive_field - seq_len, 0)
            self.pad.padding = (padding, 0)
        x = self.pad(x)
        return self.net(x)
