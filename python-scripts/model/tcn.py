# MIT License
#
# Copyright (c) 2018 CMU Locus Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch.nn as nn
from torch.nn.utils import weight_norm
from .base import CausalConv, CausalConvNet


class TemporalBlock(CausalConvNet):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2, mode='dilation'):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(CausalConv(n_inputs, n_outputs, kernel_size, subsampl=dilation, mode=mode))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(CausalConv(n_outputs, n_outputs, kernel_size, subsampl=dilation, mode=mode))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.pad1, self.conv1, self.relu1, self.dropout1,
                                 self.pad2, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self.dynamic_module_list = [self.conv1, self.conv2]  # Important! Look at CausalConvNet to see why

    def set_mode(self, mode):
        self.conv1.mode = mode
        self.conv2.mode = mode

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.net(x)
        return self.relu(out + res)


class TCN(CausalConvNet):
    def __init__(self, num_inputs, num_outputs, n_channels, dilation_sizes=None, ksize=16, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(n_channels)
        if dilation_sizes is None:
            dilation_sizes = [2 ** i for i in reversed(range(num_levels))]
        for i in range(num_levels):
            dilation_size = dilation_sizes[i]
            in_channels = num_inputs if i == 0 else n_channels[i-1]
            out_channels = n_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, ksize,
                                     dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(n_channels[-1], num_outputs, 1)

        self.dynamic_module_list = layers  # Important! Look at CausalConvNet to see why

    def forward(self, x):
        y = self.network(x)
        return self.final_conv(y)
