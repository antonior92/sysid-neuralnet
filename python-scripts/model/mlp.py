import torch.nn as nn
from .utils import Chomp1d
from collections import OrderedDict


class MLP(nn.Module):
    """Shallow 2-layer neural network with sigmoidal activation function."""
    def __init__(self, num_inputs, num_outputs, hidden_size, max_past_input, padding='same'):
        super(MLP, self).__init__()
        self.receptive_field = max_past_input

        if padding == 'same':
            conv1 = nn.Conv1d(num_inputs, hidden_size, kernel_size=max_past_input, padding=max_past_input-1)
            chomp = Chomp1d(max_past_input - 1)
        else:
            conv1 = nn.Conv1d(num_inputs, hidden_size, kernel_size=max_past_input, padding=0)

        sigmoid = nn.Sigmoid()
        conv2 = nn.Conv1d(hidden_size, num_outputs, 1)

        if max_past_input > 1 and padding == 'same':
            self.net = nn.Sequential(OrderedDict([('conv1', conv1),
                                                  ('chomp', chomp),
                                                  ('sigmoid', sigmoid),
                                                  ('conv2', conv2)]))
        else:
            self.net = nn.Sequential(OrderedDict([('conv1', conv1),
                                                  ('sigmoid', sigmoid),
                                                  ('conv2', conv2)]))

    def forward(self, x):
        return self.net(x)
