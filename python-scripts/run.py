import argparse
import json
import copy
import time
import os.path
import train
from logger import set_redirects
import data.loader as loader
from model.model_state import ModelState
import torch
from model.utils import RunMode

default_options_lstm = {
    'hidden_size': 5,
    'ar': True,
    'io_delay': 0
}

default_options_tcn = {
    'ksize': 3,
    'dropout': 0.2,
    'n_channels': [16, 32],
    'dilation_sizes': [1, 1],
    'ar': True,
    'io_delay': 0
}

default_options_mlp = {
    'hidden_size': 16,
    'max_past_input': 32,
    'ar': True,
    'io_delay': 0
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
        'seed': 3
    }
}

default_options_silverbox = {'seq_len': 1000, 'seq_len_eval': 1000}

default_options_train = {
        'init_lr': 0.001,
        'min_lr': 1e-6,
        'batch_size': 3,
        'epochs': 10000,
        'lr_scheduler_nepochs': 10,
        'lr_scheduler_factor': 10,
        'log_interval': 1,
        'training_mode': 'one-step-ahead'
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
    'normalize': False,
    'normalize_n_std': 1,
    'train_options': default_options_train,
    'test_options': default_options_test,
    'optimizer': default_options_optimizer,

    'dataset': "silverbox",
    'dataset_options': {},
    'chen_options': default_options_chen,
    'silverbox_options': default_options_silverbox,

    'model': 'mlp',
    'model_options':{} ,
    'tcn_options': default_options_tcn,
    'lstm_options': default_options_lstm,
    'mlp_options': default_options_mlp,

}


def recursive_merge(default_dict, new_dict, path=None, allow_new=False):
    # Stack overflow : https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge/7205107#7205107
    if path is None:
        path = []
    for key in new_dict:
        if key in default_dict:
            if isinstance(default_dict[key], dict) and isinstance(new_dict[key], dict):
                if key in ("model_options", "dataset_options"):
                    recursive_merge(default_dict[key], new_dict[key], path + [str(key)], allow_new=True)
                else:
                    recursive_merge(default_dict[key], new_dict[key], path + [str(key)], allow_new=allow_new)
            elif isinstance(default_dict[key], dict) or isinstance(new_dict[key], dict):
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
            else:
                default_dict[key] = new_dict[key]
        else:
            if allow_new:
                default_dict[key] = new_dict[key]
            else:
                raise Exception('Default value not found at %s' % '.'.join(path + [str(key)]))
    return default_dict


def clean_options(options):
    # Remove unused options
    datasets = ["chen", 'silverbox']
    if options["dataset"] not in datasets:
        raise Exception("Unknown dataset: " + options["dataset"])
    dataset_options = options[options["dataset"] + "_options"]

    models = ["tcn", "lstm", "mlp"]
    if options["model"] not in models:
        raise Exception("Unknown model: " + options["model"])
    model_options = options[options["model"] + "_options"]

    remove_options = [name + "_options" for name in datasets + models]
    for key in dict(options):
        if key in remove_options:
            del options[key]

    # Specify used dataset and model options
    options["dataset_options"] = recursive_merge(dataset_options, options["dataset_options"], allow_new=True)
    options["model_options"] = recursive_merge(options["model_options"], model_options, allow_new=True)
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


def create_full_options_dict(*option_dicts):
    """
    Merges multiple option dictionaries with the default dictionary and an optional option file specifying options in
    Json format.

    :param option_dicts: Any number of option dictionaries either in the form of a dictionary or a file containg Json dict
    :return: A merged option dictionary giving priority in the order of the input
    """

    merged_options = copy.deepcopy(default_options)

    # Options specified in file


    # Options specified in commandline dict
    for option_dict in reversed(option_dicts):
        if option_dict is not None:
            if isinstance(option_dict, str):
                with open(option_dict, "r") as file:
                    option_dict = json.loads(file.read())

            merged_options = recursive_merge(merged_options, option_dict)

    # Clear away unused fields and merge model options
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


def compute_mean(loader_train, cuda):
    total_batches = 0
    for i, (u, y) in enumerate(loader_train):
        total_batches += u.size()[0]
        if cuda:
            u = u.cuda()
            y = y.cuda()
        if i == 0:
            u_mean = torch.mean(u, dim=(0, 2))
            y_mean = torch.mean(y, dim=(0, 2))
            u_var = torch.mean(torch.var(u, dim=2, unbiased=False), dim=0)
            y_var = torch.mean(torch.var(y, dim=2, unbiased=False), dim=0)
        else:
            u_mean += torch.mean(u, dim=(0, 2))
            y_mean += torch.mean(y, dim=(0, 2))
            u_var += torch.mean(torch.var(u, dim=2, unbiased=False), dim=0)
            y_var += torch.mean(torch.var(y, dim=2, unbiased=False), dim=0)

    return (u_mean/total_batches, y_mean/total_batches,
           torch.sqrt(u_var/total_batches), torch.sqrt(y_var/total_batches))


def run(options=None, load_model=None, mode_interactive=True):

    if not mode_interactive:
        # Create folder
        os.makedirs(options["logdir"], exist_ok=True)
        # Set stdout to print to file and console
        set_redirects(options["logdir"])

    if load_model is not None:

        ckpt_options = create_full_options_dict(os.path.join(os.path.dirname(load_model), 'options.txt'))

        if options is None:
            options = {}

        options["model"] = ckpt_options["model"]
        options["dataset"] = ckpt_options["dataset"]

        options["optimizer"] = ckpt_options["optimizer"]
        options["model_options"] = ckpt_options["model_options"]
        options["dataset_options"] = ckpt_options["dataset_options"]

        options = recursive_merge(ckpt_options, options)
    elif options is None:
        options = create_full_options_dict()  # Default values

    # Specifying datasets
    loaders = loader.load_dataset(dataset=options["dataset"],
                                  dataset_options=options["dataset_options"],
                                  train_batch_size=options["train_options"]["batch_size"],
                                  test_batch_size=options["test_options"]["batch_size"])
    # Compute mean and var
    u_mean, y_mean, u_std, y_std = compute_mean(loaders['train'], options["cuda"])

    # Define model
    modelstate = ModelState(seed=options["seed"],
                            cuda=options["cuda"],
                            nu=loaders["train"].nu, ny=loaders["train"].ny,
                            optimizer=options["optimizer"],
                            init_lr=options["train_options"]["init_lr"],
                            model=options["model"],
                            model_options=options["model_options"],
                            normalize=options["normalize"],
                            normalize_n_std=options["normalize_n_std"],
                            u_mean=u_mean,
                            y_mean=y_mean,
                            u_std=u_std,
                            y_std=y_std)

    # Restore model
    if load_model is not None:
        current_epoch = modelstate.load_model(load_model)
    else:
        current_epoch = 0

    if not mode_interactive:
        with open(os.path.join(options["logdir"], "options.txt"), "w+") as f:
            f.write(json.dumps(options, indent=1))
            print(json.dumps(options, indent=1))

        # Run model
        train.run_train(start_epoch=current_epoch,
                        cuda=options["cuda"],
                        modelstate=modelstate,
                        logdir=options["logdir"],
                        loader_train=loaders["train"],
                        loader_valid=loaders["valid"],
                        train_options=options["train_options"])
    else:
        return modelstate.model, loaders, options


if __name__ == "__main__":
    # Get options
    commandline_options, option_dict, option_file = get_commandline_args()
    run_options = create_full_options_dict(commandline_options, option_dict, option_file)
    # Run
    run(run_options, load_model=run_options["load_model"], mode_interactive=False)
