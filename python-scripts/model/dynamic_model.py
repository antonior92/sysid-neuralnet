import torch
import torch.nn as nn
import torch.nn.functional as F
from . import MLP
from model.utils import RunMode
from .utils import copy_module_params
import time


class DynamicModel(nn.Module):
    def __init__(self, model, num_inputs, num_outputs, ar, io_delay, *args, **kwargs):
        super(DynamicModel, self).__init__()
        # Save parameters
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.args = args
        self.kwargs = kwargs
        self.ar = ar
        self.io_delay = io_delay
        self.is_cuda = False
        # Initialize model
        self.mode = RunMode.ONE_STEP_AHEAD
        if model == 'mlp':
            self.m = MLP(self.num_model_inputs, self.num_outputs, *self.args, **self.kwargs)
            self.m.set_mode(self.mode)
        else:
            raise Exception("Unimplemented model")

    def cuda(self, device=None):
        self.is_cuda = True
        super(DynamicModel, self).cuda(device)

    def cpu(self):
        self.is_cuda = False
        super(DynamicModel, self).cpu()

    @property
    def num_model_inputs(self):
        return self.num_inputs + self.num_outputs if self.ar else self.num_inputs

    def set_mode(self, mode):
        self.mode = mode
        self.m.set_mode(mode)

    def one_step_ahead(self, u, y=None):

        u_delayed = DynamicModel._get_u_delayed(u, self.io_delay)

        if self.ar:
            y_delayed = F.pad(y[:, :, :-1], [1, 0])
            x = torch.cat((u_delayed, y_delayed), 1)
        else:
            x = u_delayed
        y_pred = self.m(x)
        return y_pred

    def free_run_simulation(self, u, y=None):
        if self.ar:
            rf = self.m.receptive_field
            num_batches, _, seq_len = u.size()
            y_sim = torch.zeros(num_batches, self.num_outputs, seq_len)
            if self.is_cuda:
                y_sim = y_sim.cuda()
            u_delayed = DynamicModel._get_u_delayed(u, self.io_delay)
            for i in range(seq_len):
                if i < rf:
                    y_in = F.pad(y_sim[:, :, :i], [rf-i, 0])
                    u_in = F.pad(u_delayed[:, :, :i+1], [rf-i-1, 0])
                else:
                    y_in = y_sim[:, :, i-rf:i]
                    u_in = u_delayed[:, :, i-rf+1:i+1]
                x = torch.cat((u_in, y_in), 1)
                print(x.size())
                y_sim[:, :, i] = self.m(x)[:, :, -1]
        else:
            y_sim = self.one_step_ahead(u, y)
        return y_sim

    @staticmethod
    def _get_u_delayed(u, io_delay):
        if io_delay > 0:
            u_delayed = F.pad(u[:, :, :-io_delay], [io_delay, 0])
        elif io_delay < 0:
            u_delayed = F.pad(u[:, :, io_delay:], [0, io_delay])
        else:
            u_delayed = u

        return u_delayed

    def forward(self, *args):
        if self.mode == RunMode.ONE_STEP_AHEAD:
            return self.one_step_ahead(*args)
        elif self.mode == RunMode.FREE_RUN_SIMULATION:
            return self.free_run_simulation(*args)
        else:
            raise Exception("Not implemented mode")

