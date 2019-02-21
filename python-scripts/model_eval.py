import torch


def one_step_ahead(model, u, y, ar):
    model.eval()

    if ar:
        x = torch.cat((u, y), 1)
    else:
        x = u

    y_pred = model(x)
    return y_pred
