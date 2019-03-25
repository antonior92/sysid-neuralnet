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
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2, normalization='batch_norm'):
        super(TemporalBlock, self).__init__()
        bias = False if normalization == 'batch_norm' else True
        conv1 = CausalConv(n_inputs, n_outputs, kernel_size, subsampl=dilation, bias=bias)
        if normalization == 'batch_norm':
            bn1 = nn.BatchNorm1d(n_outputs)
        elif normalization == 'weight_norm':
            conv1.conv = weight_norm(conv1.conv)
        relu1 = nn.ReLU()
        dropout1 = nn.Dropout(dropout)

        conv2 = CausalConv(n_outputs, n_outputs, kernel_size, subsampl=dilation)
        if normalization == 'batch_norm':
            bn2 = nn.BatchNorm1d(n_outputs)
        elif normalization == 'weight_norm':
            conv2.conv = weight_norm(conv2.conv)
        relu2 = nn.ReLU()
        dropout2 = nn.Dropout(dropout)

        if normalization == 'batch_norm':
            self.net = nn.Sequential(conv1, bn1, relu1, dropout1, conv2, bn2, relu2, dropout2)
        else:
            self.net = nn.Sequential(conv1, relu1, dropout1, conv2, relu2, dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self.set_causal_conv_list([conv1, conv2])

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.net(x)
        if res.size()[2] != out.size()[2]:
            res = res[:, :, -out.size()[2]:]
        return self.relu(out + res)


class TCN(CausalConvNet):
    def __init__(self, num_inputs, num_outputs, n_channels, dilation_sizes=None, ksize=16, dropout=0.2, normalization='batch_norm'):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(n_channels)
        if dilation_sizes is None:
            dilation_sizes = [2 ** i for i in range(num_levels)]
        for i in range(num_levels):
            dilation_size = dilation_sizes[i]
            in_channels = num_inputs if i == 0 else n_channels[i-1]
            out_channels = n_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, ksize,
                                     dilation=dilation_size, dropout=dropout,
                                     normalization=normalization)]
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(n_channels[-1], num_outputs, 1)

        self.set_causal_conv_list(layers)

    def forward(self, x):
        y = self.network(x)
        return self.final_conv(y)
