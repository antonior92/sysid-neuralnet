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
from .utils import RunMode, DynamicModule


class TemporalBlock(DynamicModule):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # The padding of one conv and the receptive field of the entire module
        self.padding = (kernel_size - 1) * dilation
        self.receptive_field = 2 * self.padding + 1
        self.requested_output = None

        self.pad1 = nn.ConstantPad1d((self.padding, 0), 0)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=0, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.pad2 = nn.ConstantPad1d((self.padding, 0), 0)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=0, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.pad1, self.conv1, self.relu1, self.dropout1,
                                 self.pad2, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)

        if self.mode == RunMode.FREE_RUN_SIMULATION:
            # Calculate needed padding size
            seq_len1 = x.size()[-1]
            req_input1 = min(seq_len1, self.requested_output + self.padding) + self.padding
            padding1 = req_input1 - seq_len1
            self.pad1.padding = (padding1, 0)
            assert(padding1 >= 0)

            seq_len2 = seq_len1 + padding1 - self.padding
            req_input2 = min(seq_len2, self.requested_output) + self.padding
            padding2 = req_input2 - seq_len2
            self.pad2.padding = (padding2, 0)
            assert (padding2 >= 0)

            res = res[..., -self.requested_output:]

        out = self.net(x)

        return self.relu(out + res)

    def set_mode(self, mode):
        self.mode = mode
        if mode == RunMode.ONE_STEP_AHEAD:
            self.pad1.padding = (self.padding, 0)
            self.pad2.padding = (self.padding, 0)


class TCN(DynamicModule):
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
            layers += [TemporalBlock(in_channels, out_channels, ksize, stride=1,
                                     dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(n_channels[-1], num_outputs, 1)

        requested_output = 1
        for temporal_module in reversed(self.network):
            temporal_module.requested_output = requested_output
            requested_output += temporal_module.receptive_field - 1
        self.receptive_field = requested_output

    def forward(self, x):
        y = self.network(x)
        return self.final_conv(y)

    def set_mode(self, mode):
        self.mode = mode
        for temporal_module in self.network:
            temporal_module.set_mode(mode)
