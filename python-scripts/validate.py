# %%
import pandas as pd
import os
import glob
import json
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def show_fig(fig, plotly = False):
    if plotly:
        import plotly.tools as tls
        import plotly.offline as py
        plotly_fig = tls.mpl_to_plotly(fig)
        py.plot(plotly_fig)
    else:
        plt.show()


# Get files names
folder_list = glob.glob(os.path.join('log/chen/mlp', 'train_*'))
folder_list += glob.glob(os.path.join('log/chen/tcn', 'train_*'))


# Parse dictionaries
def single_indexed_dict(in_dict, d=None, name=''):
    if d is None:
        d = {}
    for key, value in in_dict.items():
        if isinstance(value, dict):
            single_indexed_dict(value, d=d, name=key+'_')
        elif isinstance(value, list):
            d[name + key + '_len'] = len(value)
            for i in range(len(value)):
                d[name + key + '_' + str(i)] = value[i]
        else:
            d[name+key] = value
    return d

# Generate dataframe
df= []
i = 0
options_dict = {}
for folder in folder_list:
    options_file = os.path.join(folder, 'options.txt')
    model_file = os.path.join(folder, 'best_model.pt')
    try:
        with open(options_file, 'r') as f:
            options_dict = json.loads(f.read())
    except:
        options_dict = {}
    try:
        # Gets model info
        model_pth = torch.load(model_file,  map_location='cpu')
        model_dict = {'epoch': model_pth['epoch'], 'vloss': model_pth['vloss']}
    except:
        model_dict = {}
    # Generate dataframe
    d = single_indexed_dict(dict(options_dict, **model_dict))
    df += [pd.DataFrame(d, index=[i])]
    i += 1
results = pd.concat(df, sort=False)


# Get models that fail
failed_executions = results[np.isnan(results.vloss)]

# Replace NaNs
results.fillna({'train_sd_v': 0.1, 'train_sd_w': 0.5,
                'val_sd_v': 0.1, 'val_sd_w': 0.5,
                'test_sd_v': 0.1, 'test_sd_w': 0.5},
               inplace=True)

# Filter results
results_filtered_mlp = results[
    (results['model'] == 'mlp') &
    (results['model_options_io_delay'] == 1) &
    (results['model_options_max_past_input'] == 3)
    ]
results_filtered_tcn = results[
    (results['model'] == 'tcn')
    &
    (results['model_options_n_channels_len'] == 4)
    &
    (results['model_options_dropout'] == 0)
    ]


# Plot example
fig, ax = plt.subplots(1, 2)
sns.lineplot(hue='model_options_ksize',
              y='vloss',
              x='model_options_n_channels_0',
              style='train_sd_v',
              data=results_filtered_tcn,
              legend='full', ax=ax[0])
ax[0].set_ylim([0.001, 2.5])
ax[0].set_yscale('log')
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),
             ncol=2, fancybox=True, shadow=True)
sns.lineplot(y='vloss',
          x='model_options_hidden_size',
          hue='train_sd_v',
          data=results_filtered_mlp,
          legend='full', ax=ax[1])
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),
             ncol=1, fancybox=True, shadow=True)
ax[1].set_ylim([0.001, 2.5])
ax[1].set_yscale('log')
plt.tight_layout()
show_fig(fig, True)
