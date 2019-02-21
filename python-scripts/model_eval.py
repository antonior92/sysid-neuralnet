import torch


def one_step_ahead(model, u, y, ar):
    model.eval()

    if ar:
        y_delayed = torch.cat((y[:, :, 1:], torch.zeros_like(y[:, :, 0:1])), -1)
        x = torch.cat((u, y_delayed), 1)
    else:
        x = u

    y_pred = model(x)
    return y_pred
