# %% Test
import torch
from data_generation import DataLoaderExt, ChenDataset
import matplotlib.pyplot as plt

args = {'plotly': True}

model = torch.load('best_model.pt')
loader_test = DataLoaderExt(ChenDataset(seq_len=1000, ntotbatch=2, seed=3), batch_size=1,
                            shuffle=False)
model.set_mode('free-run-simulation')

# Test
for i, (u, y) in enumerate(loader_test):
    y_pred = model(u, y)
    fig, ax = plt.subplots()
    ax.plot(y[0, 0, :].detach().numpy(), color='b', label='y true')
    ax.plot(y_pred[0, 0, :].detach().numpy(), color='g', label='y_pred')

    if args['plotly']:
        import plotly.tools as tls
        import plotly.offline as py
        plotly_fig = tls.mpl_to_plotly(fig)
        py.plot(plotly_fig)
    else:
        plt.show()
