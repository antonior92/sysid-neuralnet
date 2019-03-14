# %% Prepare
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from utils import show_fig

# %% Read file
df1 = pd.read_csv('validate/val_code.csv')
df2 = pd.read_csv('validate/hydra8.csv')
df3 = pd.read_csv('validate/val_hydra3.csv')

df = pd.concat([df1, df2, df3], axis=0, ignore_index=True, sort=False)

# Get only results from chen example
results = df[df['dataset'] == 'f16gvt']

# Create one dataframe for each model type
results_mlp = results[results['model'] == 'mlp']
results_tcn = results[results['model'] == 'tcn']
results_lstm = results[results['model'] == 'lstm']

print('len tcn = {}'.format(len(results_tcn)))
print('len lstm = {}'.format(len(results_lstm)))
print('len mlp = {}'.format(len(results_mlp)))

# %% Get best result for each configuration

models = ['mlp', 'tcn', 'lstm']

mins = []

for i, m in enumerate(models):
    aux = results[(results['model'] == m)]
    mins += [aux.ix[aux["vloss"].idxmin()]["vloss"]]


df = pd.DataFrame(np.reshape(mins, (1, 3)), columns=models, index=[0])

print(df)

# %% (TCN) Dropout/nlayers effect

fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             hue='model_options_dropout',
             x='model_options_n_channels_len',
             data=results_tcn,
             legend='full',
             estimator='min',
             ci=None)
ax.set_yscale('log')
show_fig(fig, False)

# %% (TCN) Batch norm/nlayers effect

fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             hue='model_options_normalization',
             x='model_options_n_channels_len',
             data=results_tcn,
             legend='full',
             estimator='min',
             ci=None)
ax.set_yscale('log')
show_fig(fig, False)


# %% (TCN) n_layers, num_channels effect

fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             hue='model_options_n_channels_len',
             x='model_options_n_channels_0',
             style='model_options_dilation_sizes_1',
             data=results_tcn,
             legend='full',
             estimator='min',
             ci=None)
ax.set_yscale('log')
show_fig(fig, False)


# %% (TCN) kernel size vs depth

fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             x='model_options_n_channels_len',
             hue='model_options_ksize',
             data=results_tcn,
             legend='full',
             estimator='min',
             ci=None)
ax.set_yscale('log')
show_fig(fig, False)


# %% (TCN) kernel size vs num channels

fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             x='model_options_n_channels_0',
             hue='model_options_ksize',
             data=results_tcn,
             legend='full',
             estimator='min',
             ci=None)
ax.set_yscale('log')
show_fig(fig, False)


# %% (MLP) max past input / activation function

fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             x='model_options_max_past_input',
             hue='model_options_activation_fn',
             data=results_mlp,
             legend='full',
             estimator='min',
             ci=None)
ax.set_yscale('log')
show_fig(fig, False)

# %% (MLP) hidden size / activation function
fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             x='model_options_hidden_size',
             hue='model_options_activation_fn',
             data=results_mlp,
             legend='full',
             estimator='min',
             ci=None)
ax.set_yscale('log')
show_fig(fig, False)


# %% (LSTM) hidden size list
fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             x='model_options_hidden_size',
             hue='model_options_num_layers',
             data=results_lstm,
             legend='full',
             estimator='min',
             ci=None)
ax.set_yscale('log')
show_fig(fig, False)

# %% (LSTM) dropout
fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             x='model_options_hidden_size',
             hue='model_options_dropout',
             data=results_lstm,
             legend='full',
             estimator='min',
             ci=None)
ax.set_yscale('log')
show_fig(fig, False)