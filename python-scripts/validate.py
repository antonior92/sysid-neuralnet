# %%
import pandas as pd
import os
import glob
import json
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plotly = False
def show_fig(fig):
    if plotly:
        import plotly.tools as tls
        import plotly.offline as py
        plotly_fig = tls.mpl_to_plotly(fig)
        py.plot(plotly_fig)
    else:
        plt.show()


# Get files names
folder_list = glob.glob(os.path.join('log/mlp_networks', 'train_*'))
folder_list += glob.glob(os.path.join('log/mlp_networks_2', 'train_*'))

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
    with open(options_file, 'r') as f:
        options_dict = json.loads(f.read())
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


# Plot example
fig, ax = plt.subplots()
ax = sns.lineplot(hue='model_options_hidden_size', y='vloss', x='model_options_max_past_input',data=results[results.model_options_io_delay==0][results.model_options_max_past_input < 100][results.model_options_hidden_size < 100], legend='full')
plt.legend(bbox_to_anchor=(1.1, 1.05))
show_fig(fig)