import torch
import torch.nn as nn


class DynamicModel(nn.Module):
    def __init__(self, model, num_inputs, num_outputs, ar, *args, **kwargs):
        super(DynamicModel, self).__init__()
        if ar:
            num_inputs = num_inputs + num_outputs
        self.m = model(num_inputs, num_outputs, *args, **kwargs)
        self.ar = ar
        self.mode = 'one-step-ahead'

    def get_input(self, u, y):
        if self.ar:
            y_delayed = torch.cat((torch.zeros_like(y[:, :, 0:1]), y[:, :, :-1],), -1)
            x = torch.cat((u, y_delayed), 1)
        else:
            x = u
        return x

    def one_step_ahead(self, u, y):
        x = self.get_input(u, y)
        y_pred = self.m(x)
        return y_pred

    def free_run_simulation(self, u, y):
        if self.ar:
            y_sim = torch.zeros_like(y)
            for i in range(y.size(2)):
                x = self.get_input(u[:, :, 0:i + 1], y_sim[:, :, 0:i + 1])
                y_sim[:, :, 0:i + 1] = self.m(x)
        else:
            y_sim = self.one_step_ahead(u, y)
        return y_sim

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, u, y):
        if self.mode == 'one-step-ahead':
            return self.one_step_ahead(u, y)
        elif self.mode == 'free-run-simulation':
            return self.free_run_simulation(u, y)
        else:
            raise Exception("Not implemented mode")

