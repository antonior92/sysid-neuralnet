import argparse
import json
import copy
import time
import os.path

import data.loader as loader
from model.model_state import ModelState

from test import run_test
from train import run_train

default_options_lstm = {
    'hidden_size': 5,
    'ar': True,
}

default_options_tcn = {
    'ksize': 3,
    'dropout': 0.8,
    'n_channels': [16, 32],
    'dilation_sizes': [1, 1],
    'ar': True
}

default_options_chen = {
    'seq_len': 1000,
    'train': {
        'ntotbatch': 10,
        'seed': 1
     },
    'valid': {
        'ntotbatch': 10,
        'seed': 2
    },
    'test': {
        'ntotbatch': 10,
        'seed': 2
    }
}

default_options_silverbox = {'seq_len': 1000}

default_options_train = {
        'init_lr': 0.001,
        'min_lr': 1e-6,
        'batch_size': 3,
        'epochs': 1000,
        'lr_scheduler_nepochs': 10,
        'lr_scheduler_factor': 10,
        'log_interval': 1
}

default_options_optimizer = {
    'optim': 'Adam',
}

default_options_test = {
    'plot': True,
    'plotly': True,
    'batch_size': 10,
}

default_options = {
    'cuda': False,
    'seed': 1111,
    'logdir': None,
    'run_name': None,
    'load_model': None,
    'evaluate_model': False,
    'train_options': default_options_train,
    'test_options': default_options_test,
    'optimizer': default_options_optimizer,

    'dataset': "silverbox",
    'chen_options': default_options_chen,
    'silverbox_options': default_options_silverbox,

    'model': 'lstm',
    'tcn_options': default_options_tcn,
    'lstm_options': default_options_lstm
}

def recursive_merge(default_dict, new_dict, path=None):
    # Stack overflow : https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge/7205107#7205107
    if path is None:
        path = []
    for key in new_dict:
        if key in default_dict:
            if isinstance(default_dict[key], dict) and isinstance(new_dict[key], dict):
                recursive_merge(default_dict[key], new_dict[key], path + [str(key)])
            elif default_dict[key] == new_dict[key]:
                pass  # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            raise Exception('Default value not found at %s' % '.'.join(path + [str(key)]))
            # default_dict[key] = new_dict[key]
    return default_dict


def clean_options(options):
    # Remove unused options
    datasets = ["chen", 'silverbox']
    if options["dataset"] not in datasets:
        raise Exception("Unknown dataset: " + options["dataset"])
    dataset_options = options[options["dataset"] + "_options"]

    models = ["tcn", "lstm"]
    if options["model"] not in models:
        raise Exception("Unknown model: " + options["model"])
    model_options = options[options["model"] + "_options"]

    remove_options = [name + "_options" for name in datasets + models]
    for key in dict(options):
        if key in remove_options:
            del options[key]

    # Specify used dataset and model options
    options["dataset_options"] = dataset_options
    options["model_options"] = model_options
    return options


def get_commandline_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', type=str,
                        help='Directory for logs')

    parser.add_argument('--run_name', type=str,
                        help='The name of this run')

    parser.add_argument('--load_model', type=str, help='Path to a saved model')

    parser.add_argument('--evaluate_model', type=str2bool, const=True, nargs='?', help='Evaluate model')

    parser.add_argument('--cuda', type=str2bool, const=True, nargs='?',
                        help='Specify if the model is to be run on the GPU')

    parser.add_argument('--seed', type=int,
                        help='The seed used in pytorch')

    parser.add_argument('--option_file', type=str, default=None,
                        help='File containing Json dict with options (default(%s))')

    parser.add_argument('--option_dict', type=str, default='{}',
                        help='Json with options specified at commandline (default(%s))')

    args = vars(parser.parse_args())

    # Options file
    option_file = args['option_file']

    # Options dict from commandline
    option_dict = json.loads(args['option_dict'])

    commandline_options = {k: v for k, v in args.items() if v is not None and k != "option_file" and k != "option_dict"}

    return commandline_options, option_dict, option_file


def get_options(option_file=None, *option_dicts):
    merged_options = copy.deepcopy(default_options)

    # Options specified in file
    if option_file is not None:
        file = open(option_file, "r")
        options = json.loads(file.read())
        merged_options = recursive_merge(merged_options, options)

    # Options specified in commandline dict
    for option_dict in option_dicts:
        merged_options = recursive_merge(merged_options, option_dict)

    # Clear away unused fields
    options = clean_options(merged_options)

    ctime = time.strftime("%c")

    if options["logdir"] is None:
        options["logdir"] = "log"

    if options["run_name"] is None:
        if options["evaluate_model"]:
            options["run_name"] = "eval_"+ctime
        else:
            options["run_name"] = "train_"+ctime

    options["logdir"] = os.path.join(options["logdir"], options["run_name"])

    return options


def run(options=None, load_model=None, mode='interactive'):

    if options is None:
        options = {}

    if load_model is not None:
        file = open(os.path.join(os.path.dirname(load_model), 'options.txt'), "r")
        ckpt_options = json.loads(file.read())

        options["optimizer"] = ckpt_options["optimizer"]
        options["model"] = ckpt_options["model"]
        options["model_options"] = ckpt_options["model_options"]
        options["dataset"] = ckpt_options["dataset"]
        options["dataset_options"] = ckpt_options["dataset_options"]

        options = recursive_merge(ckpt_options, options)

    # Specifying datasets
    loaders = loader.load_dataset(dataset=options["dataset"],
                                  dataset_options=options["dataset_options"],
                                  train_batch_size=options["train_options"]["batch_size"],
                                  test_batch_size=options["test_options"]["batch_size"])

    # Define model
    modelstate = ModelState(seed=options["seed"],
                            cuda=options["cuda"],
                            nu=loaders["train"].nu, ny=loaders["train"].ny,
                            optimizer=options["optimizer"],
                            init_lr=options["train_options"]["init_lr"],
                            model=options["model"],
                            model_options=options["model_options"])

    # Restore model
    if load_model is not None:
        current_epoch = modelstate.load_model(load_model)
    else:
        current_epoch = 0

    if mode == 'script':
        # Write options used to file
        os.makedirs(os.path.dirname(options["logdir"] + "/options.txt"), exist_ok=True)
        with open(options["logdir"] + "/options.txt", "w+") as f:
            f.write(json.dumps(options, indent=1))
            print(json.dumps(options, indent=1))

        # Run model
        if options["evaluate_model"]:
            run_test(epoch=current_epoch,
                     logdir = options["logdir"],
                     loader_test=loaders["test"],
                     model=modelstate.model,
                     test_options=options["test_options"])
        else:
            run_train(start_epoch=current_epoch,
                      cuda=options["cuda"],
                      modelstate=modelstate,
                      logdir=options["logdir"],
                      loader_train=loaders["train"],
                      loader_valid=loaders["valid"],
                      train_options=options["train_options"])
    elif mode == 'interactive':
        return modelstate.model, loaders, options


if __name__ == "__main__":
    commandline_options, option_dict, option_file = get_commandline_args()

    run_options = get_options(option_file, option_dict, commandline_options)

    run(run_options, load_model=run_options["load_model"], mode='script')
