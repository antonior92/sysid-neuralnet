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
        self.register_buffer('scale', torch.tensor(scale + self._epsilon, dtype=torch.float32))
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
        self.mode = RunMode.ONE_STEP_AHEAD
        self.receptive_field = None
        self.has_internal_state = None

    def set_mode(self, mode):
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError

    def init_hidden(self, batch_size):
        if self.has_internal_state:
            raise NotImplementedError


def copy_module_params(src, dest):
    for name, param in src.named_parameters():
        attrib = dest
        modules = name.split('.')
        try:
            for i in range(len(modules) - 1):
                attrib = getattr(attrib, modules[i])
            setattr(attrib, modules[-1], param)
        except:
            pass
    return dest
