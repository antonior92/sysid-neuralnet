# %% Prepare
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from utils import show_fig

# %% Read file
df1 = pd.read_csv('validate/code.csv')
df2 = pd.read_csv('validate/hydra8.csv')

df = pd.concat([df1, df2], axis=0, ignore_index=True, sort=False)

# Get only results from chen example
results = df[df['dataset'] == 'chen']

results['rmse'] = results.apply(lambda x: np.sqrt(x['vloss']), axis=1)

# Create one dataframe for each model type
results_mlp = results[results['model'] == 'mlp']
results_tcn = results[results['model'] == 'tcn']
results_lstm = results[results['model'] == 'lstm']

# %% Get best result for each configuration

models = ['mlp', 'tcn', 'lstm']
noise_levels = [0, 0.3, 0.6]
n_batches = [5, 20, 80]
n_hyperparameters = [3, 3, 3]

mins = np.zeros([3, 3, 3])
hyperparameters = {m: np.zeros([n_hyperparameters[i], 3, 3], dtype=object) for i, m in enumerate(models)}

for i, m in enumerate(models):
    for j, sd_v in enumerate(noise_levels):
        for k, n_batch in enumerate(n_batches):
            aux = results[(results['model'] == m) &
                          (results['train_sd_v'] == sd_v) &
                          (results['train_ntotbatch'] == n_batch)]
            entry = aux.ix[aux["rmse"].idxmin()]

            mins[i, j, k] = entry["rmse"]


mins_da = xr.DataArray(mins,
                       coords={'model': models,
                               'noise_levels': noise_levels,
                               'n_batches': n_batches},
                       dims=['model', 'noise_levels', 'n_batches'])
mins_df = mins_da.to_dataframe('i').unstack().swaplevel(axis=0).unstack().sort_index(level=0)

print(mins_df)
mins_df.to_csv('optimal_models_chen.csv', float_format='%.3f')


# Hyperparameters mlp
hyperparameters_mlp = xr.DataArray(hyperparameters['mlp'],
                                   coords={'hyperparam': ['activation fun',
                                                          'model order',
                                                          'hidden size'],
                                           'noise_levels': noise_levels,
                                           'n_batches': n_batches},
                                  dims=['hyperparam', 'noise_levels', 'n_batches'])
hyperparameters_mlp_df \
    = hyperparameters_mlp.to_dataframe('i').unstack().swaplevel(axis=0).unstack().sort_index(level=0)
hyperparameters_mlp_df.to_csv('hyperparam_mlp_chen.csv', float_format='%.3f')


# Hyperparameters tcn
hyperparameters_tcn = xr.DataArray(hyperparameters['tcn'],
                                   coords={'hyperparam': ['hidden size',
                                                          'n layers',
                                                          #'kernel size',
                                                          #'normalization',
                                                          #'dilation',
                                                          'dropout'],
                                           'noise_levels': noise_levels,
                                           'n_batches': n_batches},
                                  dims=['hyperparam', 'noise_levels', 'n_batches'])
hyperparameters_tcn_df \
    = hyperparameters_tcn.to_dataframe('i').unstack().swaplevel(axis=0).unstack().sort_index(level=0)

hyperparameters_tcn_df.to_csv('hyperparam_tcn_chen.csv', float_format='%.3f')

# Hyperparameters lstm
hyperparameters_lstm = xr.DataArray(hyperparameters['lstm'],
                                   coords={'hyperparam': ['hidden size',
                                                          'n layers',
                                                          'dropout'],
                                           'noise_levels': noise_levels,
                                           'n_batches': n_batches},
                                  dims=['hyperparam', 'noise_levels', 'n_batches'])
hyperparameters_lstm_df \
    = hyperparameters_lstm.to_dataframe('i').unstack().swaplevel(axis=0).unstack().sort_index(level=0)

hyperparameters_lstm_df.to_csv('hyperparam_lstm_chen.csv', float_format='%.3f')

# %% (TCN) Dropout effect (global)

fig, ax = plt.subplots()
sns.boxplot(y='rmse',
             x='model_options_dropout',
             data=results_tcn[(results_tcn['train_sd_v'] == 0.3) &
                              (results_tcn['train_ntotbatch'] == 20)],
            )

ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('dropout rate', fontsize=16)
ax.set_ylabel('RMSE', fontsize=16)
plt.tight_layout()
plt.savefig('dropout_chen.png')
show_fig(fig, False)

# %% (TCN) batch norm, weight norm, nothing


fig, ax = plt.subplots()
sns.boxplot(y='rmse',
            x='model_options_normalization',
            data=results_tcn[(results_tcn['train_sd_v'] == 0.3) &
                              (results_tcn['train_ntotbatch'] == 20)],
            )

ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('', fontsize=16)
ax.set_ylabel('RMSE', fontsize=16)
plt.tight_layout()
plt.savefig('normalization_chen.png')

show_fig(fig, False)


# %% (TCN) Dilations effects


fig, ax = plt.subplots()
sns.boxplot(y='rmse',
            x='model_options_dilation_sizes_1',
            data=results_tcn[(results_tcn['train_sd_v'] == 0.3) &
                              (results_tcn['train_ntotbatch'] == 20)],
            )


ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('', fontsize=16)
ax.set_ylabel('RMSE', fontsize=16)
ax.set_xticklabels(['no dilations', 'with dilations'])
plt.tight_layout()
plt.savefig('dilations_chen.png')
show_fig(fig, False)



# %% (TCN) Effect of the number of n layers (global)
fig, ax = plt.subplots()
sns.boxplot(y='rmse',
            x='model_options_n_channels_len',
            data=results_tcn[(results_tcn['train_sd_v'] == 0.3) &
                              (results_tcn['train_ntotbatch'] == 20)],
            )


ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('number of residual blocks', fontsize=16)
ax.set_ylabel('RMSE', fontsize=16)
ax.set_xticklabels([1, 2, 4, 8])
plt.tight_layout()
plt.savefig('nlayers_chen.png')
show_fig(fig, False)

# %% (TCN) Effect of the number of channels/layers
fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             x='model_options_n_channels_0',
             hue='model_options_n_channels_len',
             style='train_sd_v',
             data=results_tcn[(results['train_ntotbatch'] == 5)],
             legend='full',
             estimator='min',
             ci=None)
show_fig(fig, False)

# %% (TCN) Effect of the number of channels/layers (with dropout)
fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             x='model_options_n_channels_0',
             hue='model_options_n_channels_len',
             style='train_sd_v',
             data=results_tcn[(results['train_ntotbatch'] == 20) &
                              (results['model_options_dropout'] == 0.8)],
             legend='full',
             estimator='min',
             ci=None)
ax.set_xscale('log')
show_fig(fig, False)

# %% (TCN) Effect of the number of channels/layers/norm
fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             x='model_options_n_channels_len',
             hue='train_ntotbatch',
             style='model_options_normalization',
             data=results_tcn[(results['train_sd_v'] == 0.3)],
             legend='full',
             estimator='min',
             ci=None)
show_fig(fig, False)


# %% (MLP) Effect of the activation function / number of hidden units
fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             x='model_options_hidden_size',
             hue='model_options_activation_fn',
             style='train_sd_v',
             data=results_mlp[(results['train_ntotbatch'] == 20)],
             legend='full',
             estimator='min',
             ci=None)
show_fig(fig, False)


# %% (LSTM) Effect of the dropout (global)

fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             x='model_options_dropout',
             hue='train_ntotbatch',
             style='train_sd_v',
             data=results_lstm,
             legend='full',
             estimator='min',
             ci=None)
show_fig(fig, False)


# %% (LSTM) Effect of the hidden_size/num_layes

fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             x='model_options_hidden_size',
             hue='model_options_num_layers',
             style='train_sd_v',
             data=results_lstm[(results['train_ntotbatch'] == 20)],
             legend='full',
             estimator='min',
             ci=None)
ax.set_xscale('log')
show_fig(fig, False)
