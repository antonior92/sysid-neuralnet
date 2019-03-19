import glob
import os
import json
import torch
import pandas as pd


def get_results_frames(*folders):
    """
    Finds all training runs in folders and create an pandas frame containing the results and options

    :param folders:
    :return: pandas frame with results
    """
    folder_list = []
    for glob_folder in folders:
        folder_list += glob.glob(os.path.join(glob_folder, 'train_*'))

    def single_indexed_dict(in_dict, d=None, name=''):
        if d is None:
            d = {}
        for key, value in in_dict.items():
            if isinstance(value, dict):
                single_indexed_dict(value, d=d, name=key + '_')
            elif isinstance(value, list):
                d[name + key + '_len'] = len(value)
                for i in range(len(value)):
                    d[name + key + '_' + str(i)] = value[i]
            else:
                d[name + key] = value
        return d

    # Generate dataframe
    df = []
    i = 0
    options_dict = {}
    for folder in folder_list:
        options_file = os.path.join(folder, 'options.txt')
        model_file = os.path.join(folder, 'best_model.pt')
        with open(options_file, 'r') as f:
            options_dict = json.loads(f.read())
        try:
            # Gets model info
            model_pth = torch.load(model_file, map_location='cpu')
            model_dict = {'epoch': model_pth['epoch'], 'vloss': model_pth['vloss'], 'run_path': folder}
        except:
            model_dict = {}
        # Generate dataframe
        d = single_indexed_dict(dict(options_dict, **model_dict))
        df += [pd.DataFrame(d, index=[i])]
        i += 1
    results = pd.concat(df, sort=False)

    return results
