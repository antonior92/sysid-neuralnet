import torch.nn as nn
from enum import Enum


class RunMode(str, Enum):
    FREE_RUN_SIMULATION = 'free-run-simulation'
    ONE_STEP_AHEAD = 'one-step-ahead'


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


class DynamicModule(nn.Module):
    def __init__(self):
        super(DynamicModule,self).__init__()
        self.mode = RunMode.ONE_STEP_AHEAD
        self.receptive_field = None

    def set_mode(self, mode):
        raise NotImplementedError
