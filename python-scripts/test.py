# %% Test
import torch
from model_eval import one_step_ahead
from data_generation import chen_example
import matplotlib.pyplot as plt

args = {'plotly': True,
        'ar': True}

model = torch.load('best_model.pt')
u_test, y_test = chen_example(1000, 2)
u_test, y_test = torch.tensor(u_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)

# Test
for i in range(0, u_test.size()[0]):
    fig, ax = plt.subplots()
    y_pred = one_step_ahead(model, u_test, y_test, args['ar'])
    ax.plot(y_test[i, 0, :].detach().numpy(), color='b', label='y true')
    ax.plot(y_pred[i, 0, :].detach().numpy(), color='g', label='y_pred')

    if args['plotly']:
        import plotly.tools as tls
        import plotly.offline as py
        plotly_fig = tls.mpl_to_plotly(fig)
        py.plot(plotly_fig)
    else:
        plt.show()
