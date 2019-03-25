# %% Imports
import torch
import run
import numpy as np
import matplotlib.pyplot as plt
from utils import show_fig
import scipy.io as io


# %% Load model, data and options
(model_tcn, loader, options) = run.run({"cuda": False},
                                        load_model="test/f16gvt_tcn/best_model.pt")
model_tcn.cpu()
model_tcn.eval()

(model_mlp, _, _) = run.run({"cuda": False},
                            load_model="test/f16gvt_mlp/best_model.pt")
model_mlp.cpu()
model_mlp.eval()

(model_lstm, _, _) = run.run({"cuda": False},
                            load_model="test/f16gvt_lstm/best_model.pt")
model_lstm.cpu()
model_lstm.eval()


#%% Get outputs and one-step-ahead predictions
model_tcn.set_mode('one-step-ahead')
model_mlp.set_mode('one-step-ahead')
model_lstm.set_mode('one-step-ahead')
y = []
u = []
one_step_ahead_tcn = []
one_step_ahead_mlp = []
one_step_ahead_lstm = []
for i, (ui, yi) in enumerate(loader["test"]):
    with torch.no_grad():
        one_step_ahead_tcn += [model_tcn(ui, yi).detach()]
        one_step_ahead_mlp += [model_mlp(ui, yi).detach()]
        one_step_ahead_lstm += [model_lstm(ui, yi).detach()]
        y += [yi]
        u += [ui]

y = np.squeeze(np.concatenate(y, 0))
u = np.squeeze(np.concatenate(u, 0))
one_step_ahead_tcn = np.squeeze(np.concatenate(one_step_ahead_tcn, 0))
one_step_ahead_mlp = np.squeeze(np.concatenate(one_step_ahead_mlp, 0))
one_step_ahead_lstm = np.squeeze(np.concatenate(one_step_ahead_lstm, 0))

for i in range(3):
    fig, ax = plt.subplots()
    plt.plot(y[i, :])
    plt.plot(one_step_ahead_tcn[i, :])
    plt.plot(one_step_ahead_mlp[i, :])
    plt.plot(one_step_ahead_lstm[i, :])
    show_fig(fig, True)


#%% Get free-run-simulation
model_tcn.set_mode('free-run-simulation')
model_mlp.set_mode('free-run-simulation')
model_lstm.set_mode('free-run-simulation')
free_run_simulation_tcn = []
free_run_simulation_mlp = []
free_run_simulation_lstm = []
for i, (ui, yi) in enumerate(loader["test"]):
    with torch.no_grad():
        free_run_simulation_tcn += [model_tcn(ui, yi).detach()]
        free_run_simulation_mlp += [model_mlp(ui, yi).detach()]
         free_run_simulation_lstm += [model_lstm(ui, yi).detach()]

free_run_simulation_tcn = np.squeeze(np.concatenate(free_run_simulation_tcn, 0))
free_run_simulation_mlp = np.squeeze(np.concatenate(free_run_simulation_mlp, 0))
free_run_simulation_lstm = np.squeeze(np.concatenate(free_run_simulation_lstm, 0))

for i in range(3):
    fig, ax = plt.subplots()
    plt.plot(y[i, :])
    plt.plot(free_run_simulation_tcn[i, :])
    plt.plot(free_run_simulation_mlp[i, :])
    plt.plot(free_run_simulation_lstm[i, :])
    show_fig(fig, True)

# %% Save results
io.savemat('f16gvt_outputs', {"y": y, "u": u,
                              "one_step_ahead_mlp": one_step_ahead_mlp,
                              "one_step_ahead_tcn": one_step_ahead_tcn,
                              "one_step_ahead_lstm": one_step_ahead_lstm,
                              "free_run_simulation_mlp": free_run_simulation_mlp,
                              "free_run_simulation_tcn": free_run_simulation_tcn,
                              "free_run_simulation_lstm": free_run_simulation_lstm
                              })