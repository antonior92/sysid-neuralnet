import torch
import torch.nn as nn
from enum import Enum


class RunMode(str, Enum):
    FREE_RUN_SIMULATION = 'free-run-simulation'
    ONE_STEP_AHEAD = 'one-step-ahead'


class Normalizer1D(nn.Module):
    _epsilon = 1e-16

    def __init__(self, scale, offset):
        super(Normalizer1D, self).__init__()
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32) + self._epsilon)
        self.register_buffer('offset', torch.tensor(offset, dtype=torch.float32))

    def normalize(self, x):
        x = x.permute(0, 2, 1)
        x = (x-self.offset)/self.scale
        return x.permute(0, 2, 1)

    def unnormalize(self, x):
        x = x.permute(0, 2, 1)
        x = x*self.scale + self.offset
        return x.permute(0, 2, 1)


class DynamicModule(nn.Module):
    def __init__(self):
        super(DynamicModule, self).__init__()
        self.has_internal_state = False

    def get_requested_input(self, requested_output='internal'):
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError

    def init_hidden(self, batch_size):
        if self.has_internal_state:
            raise NotImplementedError


class CausalConv(DynamicModule):
    def __init__(self, in_channels, out_channels, kernel_size,
                 subsampl=1, bias=True, groups=1, mode='dilation'):
        super(CausalConv, self).__init__()
        self.kernel_size = kernel_size
        self.subsampl = subsampl
        self.mode = mode
        padding = (kernel_size - 1) * subsampl
        self.pad = nn.ConstantPad1d((padding, 0), 0)
        if mode == 'dilation':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                  dilation=subsampl, groups=groups, bias=bias)
        elif mode == 'stride':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                  groups=groups, bias=bias)
        self.requested_output = None

    def set_requested_output(self, requested_output):
        self.requested_output = requested_output

    def get_requested_output(self):
        return self.requested_output

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'dilation':
            self.conv.dilation = self.subsampl
        else:
            self.conv.dilation = 1

    def get_requested_input(self, requested_output='internal'):
        if requested_output is None:
            return None
        if requested_output == 'same':
            return 'same'
        if requested_output == 'internal':
            requested_output = self.requested_output
        if self.mode == 'stride':
            requested_output = self.subsampl * (requested_output-1) + 1
        return requested_output + (self.kernel_size - 1) * self.subsampl

    def forward(self, x):
        seq_len = x.size()[-1]
        if self.requested_output is not None:
            requested_output = self.requested_output if self.requested_output != 'same' else seq_len
            requested_input = self.get_requested_input(requested_output)
            padding = max(requested_input - seq_len, 0)
            self.pad.padding = (padding, 0)
            x = self.pad(x)
        if self.mode == 'stride':
            x = x[:, :, (seq_len-1)%self.subsampl::self.subsampl]
        return self.conv(x)


class CausalConvNet(DynamicModule):
    def __init__(self):
        super(CausalConvNet, self).__init__()
        self.dynamic_module_list = None

    def set_dynamic_module_list(self, l):
        self.dynamic_module_list = l

    def get_requested_input(self, requested_output='internal'):
        requested = requested_output
        for temporal_module in reversed(self.dynamic_module_list):
            requested = temporal_module.requested_output(requested)
        return requested

    def get_requested_output(self):
        return self.dynamic_module_list[-1].requested_output

    def set_requested_output(self, requested_output):
        requested = requested_output
        for temporal_module in reversed(self.dynamic_module_list):
            temporal_module.requested_output = requested
            requested = temporal_module.requested_input(requested)

    def set_mode(self, mode):
        for temporal_module in reversed(self.dynamic_module_list):
            if isinstance(o, (CausalConvNet, CausalConv)):
                temporal_module.set_mode(mode)

    def forward(self, *input):
        raise NotImplementedError

