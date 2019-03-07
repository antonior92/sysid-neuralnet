import torch.nn as nn
from .base import CausalConv, CausalConvNet
from collections import OrderedDict


class MLP(CausalConvNet):
    """Shallow 2-layer neural network with sigmoidal activation function."""
    def __init__(self, num_inputs, num_outputs, hidden_size, max_past_input, activation_fn):
        super(MLP, self).__init__()

        self.conv1 = CausalConv(num_inputs, hidden_size, max_past_input)

        if activation_fn == "sigmoid":
            self.fn = nn.Sigmoid()
        elif activation_fn == "relu":
            self.fn = nn.ReLU()
        elif activation_fn == "elu":
            self.fn = nn.ELU()
        else:
            raise Exception("Activation function not implemented: " + activation_fn)

        self.conv2 = nn.Conv1d(hidden_size, num_outputs, 1)

        # Due to backward compatibility the modules are named as follows
        self.net = nn.Sequential(OrderedDict([("0", self.conv1),
                                              ("2", self.fn),
                                              ("3", self.conv2)]))

        self.set_causal_conv_list([self.conv1])

    def forward(self, x):
        return self.net(x)
