import torch
import torch.nn as nn
from . import MLP
from .utils import copy_module_params
import time


class DynamicModel(nn.Module):
    def __init__(self, model, num_inputs, num_outputs, ar, io_delay=0,
                 mode='one-step-ahead', *args, **kwargs):
        super(DynamicModel, self).__init__()
        # Save parameters
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.args = args
        self.kwargs = kwargs
        self.ar = ar
        self.io_delay = io_delay
        # Initialize model
        self.mode = None
        self.m = None
        self.receptive_field = None
        self.set_mode(mode)

    def set_mode(self, mode):
        if self.model == 'mlp':
            padding = 'none' if mode == 'free-run-simulation' else 'same'
            num_inputs = self.num_inputs + self.num_outputs if self.ar else self.num_inputs
            m = MLP(num_inputs, self.num_outputs, padding=padding,
                    *self.args, **self.kwargs)
            receptive_field = m.receptive_field
        else:
            raise Exception("Unimplemented model")
        if self.mode is not None:
            m = copy_module_params(self.m, m)
        self.mode = mode
        self.m = m
        self.receptive_field = receptive_field

    def one_step_ahead(self, u, y=None):
        io_delay = self.io_delay
        n_batches, n_inputs, seq_len = u.size()
        n_outputs = self.num_outputs
        u_delayed = torch.cat((torch.zeros((n_batches, n_inputs, io_delay)), u[:, :, :-io_delay],), -1) if io_delay > 0 else u
        if self.ar:
            y_delayed = torch.cat((torch.zeros((n_batches, n_outputs, 1)), y[:, :, :-1],), -1)
            x = torch.cat((u_delayed, y_delayed), 1)
        else:
            x = u_delayed
        y_pred = self.m(x)
        return y_pred

    def free_run_simulation(self, u, y=None):
        if self.ar:
            rf = self.receptive_field
            io_delay = self.io_delay
            n_batches, n_inputs, seq_len = u.size()
            n_outputs = self.num_outputs
            y_sim = torch.zeros(n_batches, n_outputs, seq_len)
            u_delayed = torch.cat((torch.zeros((n_batches, n_inputs, io_delay)), u[:, :, :-io_delay],), -1) if io_delay > 0 else u
            for i in range(seq_len):
                if i < rf:
                    y_in = torch.cat((torch.zeros((n_batches, n_outputs, rf-i)), y_sim[:, :, :i]), -1)
                    u_in = torch.cat((torch.zeros((n_batches, n_inputs, rf-i-1)), u_delayed[:, :, :i+1]), -1)
                else:
                    y_in = y_sim[:, :, i-rf:i]
                    u_in = u_delayed[:, :, i-rf+1:i+1]
                x = torch.cat((u_in, y_in), 1)
                y_sim[:, :, i] = self.m(x)[:, :, -1]
        else:
            y_sim = self.one_step_ahead(u, y)
        return y_sim

    def forward(self, *args):
        if self.mode == 'one-step-ahead':
            return self.one_step_ahead(*args)
        elif self.mode == 'free-run-simulation':
            return self.free_run_simulation(*args)
        else:
            raise Exception("Not implemented mode")

