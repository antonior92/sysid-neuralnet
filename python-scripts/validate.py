# %%
import pandas as pd
import os
import glob
import json
import torch
import seaborn as sns
import matplotlib.pyplot as plt

dir_path = 'log/mlp_networks'

# Get files names
folder_list = glob.glob(os.path.join(dir_path, 'train_*'))


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
    # Gets model info
    model_pth = torch.load(model_file)
    model_dict = {'epoch': model_pth['epoch'], 'vloss': model_pth['vloss']}
    # Generate dataframe
    d = single_indexed_dict(dict(options_dict, **model_dict))
    df += [pd.DataFrame(d, index=[i])]
    i += 1
results = pd.concat(df, sort=False)

# Plot example
ax = sns.lineplot(x='model_options_io_delay', y='vloss', hue='model_options_max_past_input', data=results)
plt.show()