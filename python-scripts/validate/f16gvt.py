# %% Prepare
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from utils import show_fig

# %% Read file
df1 = pd.read_csv('validate/val2_code.csv')
df2 = pd.read_csv('validate/hydra8.csv')
df3 = pd.read_csv('validate/val2_hydra3.csv')

df = pd.concat([df1, df2, df3], axis=0, ignore_index=True, sort=False)

# Get only results from chen example
results = df[df['dataset'] == 'f16gvt']

results['rmse'] = results.apply(lambda x: np.sqrt(x['vloss']), axis=1)

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
logdir = []
for i, m in enumerate(models):
    aux = results[(results['model'] == m)]
    mins += [aux.ix[aux["vloss"].idxmin()]["vloss"]]
    logdir += [aux.ix[aux["vloss"].idxmin()]["logdir"]]


df = pd.DataFrame(np.reshape(mins, (1, 3)), columns=models, index=[0])

print(df)

# %% Get best models directories

for i in range(3):
    if sum(df1['logdir'] == logdir[i]):
        print('Best {0} model is on code:~/Documents/sysid-neuralnet/python-scripts/{1}'.format(models[i], logdir[i]))

for i in range(3):
    if sum(df2['logdir'] == logdir[i]):
        print('Best {0} model is on hydra8:~/Documents/sysid-neuralnet/python-scripts/{1}'.format(models[i], logdir[i]))

for i in range(3):
    if sum(df3['logdir'] == logdir[i]):
        print('Best {0} model is on hydra3:~/Documents/sysid-neuralnet/python-scripts/{1}'.format(models[i], logdir[i]))

# %% (TCN) Dropout effect

fig, ax = plt.subplots()
sns.boxplot(y='rmse',
             x='model_options_dropout',
             data=results_tcn,
            )

ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('dropout rate', fontsize=16)
ax.set_ylabel('RMSE', fontsize=16)
plt.tight_layout()
plt.savefig('dropout_f16gvt.png')
show_fig(fig, False)

# %% (TCN) Batch norm

fig, ax = plt.subplots()
sns.boxplot(y='rmse',
            x='model_options_normalization',
            data=results_tcn,
            order=['weight_norm', 'batch_norm', 'none']

            )

ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('', fontsize=16)
ax.set_ylabel('RMSE', fontsize=16)
plt.tight_layout()
plt.savefig('normalization_f16gvt.png')

show_fig(fig, False)





# %%
fig, ax = plt.subplots()
sns.boxplot(y='rmse',
            x='model_options_dilation_sizes_1',
            data=results_tcn,
            )


ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('', fontsize=16)
ax.set_ylabel('RMSE', fontsize=16)
ax.set_xticklabels(['no dilations', 'with dilations'])
plt.tight_layout()
plt.savefig('dilations_f16gvt.png')
show_fig(fig, False)

# %% (TCN) kernel size vs depth

fig, ax = plt.subplots()
sns.boxplot(y='rmse',
            x='model_options_n_channels_len',
            hue='model_options_n_channels_0',
            data=results_tcn,
            )


ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('number of residual blocks', fontsize=16)
ax.set_ylabel('RMSE', fontsize=16)
ax.set_xticklabels([1, 2, 4, 8])
plt.tight_layout()
plt.savefig('nlayers_chen.png')
show_fig(fig, False)

#%%
fig, ax = plt.subplots()
sns.boxplot(y='rmse',
            x='model_options_n_channels_len',
            data=results_tcn,
            )


ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('number of residual blocks', fontsize=16)
ax.set_ylabel('RMSE', fontsize=16)
ax.set_xticklabels([1, 2, 4, 8])
plt.tight_layout()
plt.savefig('nlayers_f16gvt.png')
show_fig(fig, False)

# %% (TCN) kernel size vs num channels

fig, ax = plt.subplots()
sns.lineplot(y='vloss',
             x='model_options_n_channels_0',
             #hue='model_options_ksize',
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