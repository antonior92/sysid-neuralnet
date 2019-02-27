import torch.nn as nn
from .utils import Chomp1d


class MLP(nn.Module):
    """Shallow 2-layer neural network with sigmoidal activation function."""
    def __init__(self, num_inputs, num_outputs, hidden_size, max_past_input):
        super(MLP, self).__init__()

        conv1 = nn.Conv1d(num_inputs, hidden_size, kernel_size=max_past_input, padding=max_past_input-1)
        chomp = Chomp1d(max_past_input-1)
        sigmoid = nn.Sigmoid()
        conv2 = nn.Conv1d(hidden_size, num_outputs, 1)

        if max_past_input > 1:
            self.net = nn.Sequential(conv1, chomp, sigmoid, conv2)
        else:
            self.net = nn.Sequential(conv1, sigmoid, conv2)

    def forward(self, x):
        return self.net(x)
