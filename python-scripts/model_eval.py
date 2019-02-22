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


def simulation(model, u, y, ar):
    y = torch.zeros_like(y)
    if ar:
        y_sim = torch.zeros_like(y)
        model.eval()
        for i in range(y.size(2)):
            x = get_input(u[:, :, 0:i+1], y_sim[:, :, 0:i+1], ar)
            y_sim[:, :, 0:i+1] = model(x)
    else:
        y_sim = one_step_ahead(model, u, y, ar)    
    return y_sim

