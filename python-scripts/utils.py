import numpy as np

def show_fig(fig, plotly=False):
    if plotly:
        import plotly.tools as tls
        import plotly.offline as py
        plotly_fig = tls.mpl_to_plotly(fig)
        py.plot(plotly_fig)
    else:
        import matplotlib.pyplot as plt
        plt.show()

def compute_fit(y, yHat):
    fit = 100*(1 - np.linalg.norm(y - yHat)/np.linalg.norm(y - np.mean(y)))
    return fit
