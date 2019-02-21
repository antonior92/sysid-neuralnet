import torch


def get_input(u, y, ar):
    if ar:
        y_delayed = torch.cat((torch.zeros_like(y[:, :, 0:1]), y[:, :, :-1], ), -1)
        x = torch.cat((u, y_delayed), 1)
    else:
        x = u
    return x


def one_step_ahead(model, u, y, ar):
    model.eval()
    x = get_input(u, y, ar)
    y_pred = model(x)
    return y_pred


