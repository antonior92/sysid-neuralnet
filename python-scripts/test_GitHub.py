# %% Test
import torch
import numpy as np
from data_generation import DataLoaderExt, ChenDataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm

args = {'plotly': True}

model = torch.load('best_model.pt')
loader_test = DataLoaderExt(ChenDataset(seq_len=1000, ntotbatch=2, seed=3), batch_size=1,
                            shuffle=False)
model.set_mode('free-run-simulation')

def handlePlotly(fig, args):
    if args['plotly']:
        import plotly.tools as tls
        import plotly.offline as py
        plotly_fig = tls.mpl_to_plotly(fig)
        py.plot(plotly_fig)
    else:
        plt.show()

def compute_rmse(y_pred, y):
    y = y[0, 0, :].detach().numpy()
    y_pred = y_pred[0, 0, :].detach().numpy()
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    return rmse

# Test
for i, (u, y) in enumerate(loader_test):
    # True and predicted/simulated output
    y_pred = model(u, y)
    fig, ax = plt.subplots()
    ax.plot(y[0, 0, :].detach().numpy(), color='b', label='True')
    if model.mode == 'one-step-ahead':
        ax.plot(y_pred[0, 0, :].detach().numpy(), color='g', label='Predicted')
        ax.set_title('True and predicted output')
    elif model.mode == 'free-run-simulation':
        ax.plot(y_pred[0, 0, :].detach().numpy(), color='g', label='Simulated')
        ax.set_title('True and simulated output')
    else:
#        ax.set_title('True output')
        raise Exception("Model mode not implemented: {}".format(model.mode))
    ax.set_xlabel('Sample number')
    ax.set_ylabel('Amplitude (V)')
    handlePlotly(fig, args)

    # True output and error predicted/simulated output
    e_pred = y - y_pred
    fig, ax = plt.subplots()
    ax.plot(y[0, 0, :].detach().numpy(), color='b', label='True')
    if model.mode == 'one-step-ahead':
        ax.plot(e_pred[0, 0, :].detach().numpy(), color='r', label='Prediction error')
        ax.set_title('True output and error one-step-ahead prediction')
    elif model.mode == 'free-run-simulation':
        ax.plot(e_pred[0, 0, :].detach().numpy(), color='r', label='Simulation error')
        ax.set_title('True output and error free-run simulation')
    else:
#        ax.set_title('True output')
        raise Exception("Model mode not implemented: {}".format(model.mode))
    ax.set_xlabel('Sample number')
    ax.set_ylabel('Amplitude (V)')
    handlePlotly(fig, args)

    # Comparison with other methods Silverbox (mostly from PhD thesis Anne Van Mulders)
    if model.mode == 'free-run-simulation':
        model_rmse = compute_rmse(y_pred, y)
        model_number_of_parameters = 1000 # TODO: Needs to be computed
        otherApproachesSilverbox = {
            'Hjalmarsson and Schoukens (2004)': {
                    'num_parameters': 5,
                    'test_rmse': 0.96,
                    'abbreviation': 'NLFB relaxation'
            },
            'Paduart et al. (2004)': {
                    'num_parameters': 10,
                    'test_rmse': 0.38,
                    'abbreviation': 'NLFB'
            },
            'Ljung et al. (2004)': {
                    'num_parameters': 712,
                    'test_rmse': 0.30,
                    'abbreviation': 'NL ARX'
            },
            'Espinoza et al. (2004)': {
                    'num_parameters': 490, # Number of support vectors
                    'test_rmse': 0.32,
                    'abbreviation': 'LSSVM-NARX'
            },
            'Verdult (2004)': {
                    'num_parameters': 16,
                    'test_rmse': 1.3,
                    'abbreviation': 'LLSS'
            },
            'Paduart (2008)': {
                    'num_parameters': 37,
                    'test_rmse': 0.26,
                    'abbreviation': 'PNLSS'
            },
            'Paduart (2008) BLA': {
                    'num_parameters': 5,
                    'test_rmse': 13.7,
                    'abbreviation': 'BLA'
            },
            'Van Mulders et al. (2011)': {
                    'num_parameters': 11,
                    'test_rmse': 0.35,
                    'abbreviation': 'poly-LFR'
            },
            'Marconato et al. (2012)': {
                    'num_parameters': 23,
                    'test_rmse': 0.34,
                    'abbreviation': 'sigm-NLSS'
            },
            'Espinoza (2006)': {              # prediction mode? No, in Espinoza et al. 2005, this result appears in the simulation result row in the table
                    'num_parameters': 190,
                    'test_rmse': 0.27,
                    'abbreviation': 'PWL-LSSVM-NARX'
            },
            'Srager et al. (2004)': {
                    'num_parameters': 600,
                    'test_rmse': 7.8,
                    'abbreviation': 'MLP-ANN'
            },
            'Pepona et al. (2011)': {
                    'num_parameters': 18,
                    'test_rmse': 4.08,
                    'abbreviation': 'PWA-LFR'
            }
        }
        colors = iter(cm.rainbow(np.linspace(0, 1, len(otherApproachesSilverbox))))
        fig, ax = plt.subplots()
        for key, value in otherApproachesSilverbox.items():
            ax.scatter(value['num_parameters'], value['test_rmse'], color=next(colors), label=value['abbreviation'])
        # Plot result current model here
        ax.scatter(model_number_of_parameters, model_rmse, color='k', label='This paper')
        ax.set_title('Free-run simulation results')
        ax.set_xlabel('Number of parameters')
        ax.set_ylabel('rms error on test data (mV)')
        handlePlotly(fig, args)
