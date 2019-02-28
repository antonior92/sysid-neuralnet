import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


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
