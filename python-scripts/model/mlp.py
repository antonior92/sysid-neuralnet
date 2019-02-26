import torch.nn as nn
from .utils import Chomp1d


class MLP(nn.Module):
    """Shallow 2-layer neural network with sigmoidal activation function."""
    def __init__(self, num_inputs, num_outputs, hidden_size, max_past_input):
        super(MLP, self).__init__()

        self.conv1 = nn.Conv1d(num_inputs, hidden_size, kernel_size=max_past_input, padding=max_past_input-1)
        self.chomp = Chomp1d(max_past_input-1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv1d(hidden_size, num_outputs, 1)

        self.net = nn.Sequential()

    def forward(self, x):
        return self.net(x)
