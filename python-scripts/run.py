


import argparse
import json
import copy

default_options_lstm = {
    'hidden_size': 5
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
    }
}

default_options_tcn = {
    'ksize': 3,
    'dropout': 0.8,
    'n_channels': [16, 32],
    'dilation_sizes': [1, 1],
}

default_options_silverbox = {'seq_len': 1000}

default_options_training = {
        'optim': 'Adam',
        'lr': 0.001,
        'min_lr': 1e-6,
        'batch_size': 3,
        'epochs': 1000,
        'lr_scheduler_nepochs': 10,
        'lr_scheduler_factor': 10,
        'log_interval': 1
}

default_options_test = {
    'plot': True,
    'plotly': True,
    'eval_batch_size': 10,
}


default_options = {
        'cuda': False,
        'seed': 1111,
        'logdir': None,
        'load_model': None,
        'evaluate_model': False,

        'ar': True,
        'dataset': "SilverBox",
        'model': 'lstm',
        'training_options': default_options_training,
        'tcn_options': default_options_tcn,
        'lstm_options': default_options_lstm,
        'chen_options': default_options_chen,
        'silverbox_options': default_options_silverbox,
        'test_options': default_options_test
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
            #default_dict[key] = new_dict[key]
    return default_dict


def main():
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

    parser.add_argument('--load_ckpt', type=str, help='Path to a saved model, overrides ALL other options \\'
                                                      'except the evaluate model flag, cuda flag and seed')

    parser.add_argument('--evaluate_model', type=str2bool, const=True, nargs='?', help='Evaluate model')

    parser.add_argument('--cuda', type=str2bool, const=True, nargs='?',
                        help='Specify if the model is to be run on the GPU')

    parser.add_argument('--seed', type=int,
                        help='The seed used in pytorch')

    parser.add_argument('--option_file', type=str, default=None,
                        help='File containing Json dict with options (default(%s))')

    parser.add_argument('--option_dict', type=str, default='{}',
                        help='Json with options specified at commandline (default(%s))')


    merged_options = copy.deepcopy(default_options)

    args = vars(parser.parse_args())

    # Options specified in file
    if args["option_file"] is not None:
        file = open(args["option_file"], "r")
        options = json.loads(file.read())
        merged_options = recursive_merge(merged_options, options)

    # Options specified in commandline dict
    options = json.loads(args['option_dict'])
    merged_options = recursive_merge(merged_options, options)

    # Options specified at command line
    args = {k: v for k, v in args.items() if v is not None and k != "option_file" and k != "option_dict"}
    merged_options = {**merged_options, **args}

    options_copy = copy.deepcopy(merged_options)


    # train(used_options=options_copy, **options)














if __name__ == "__main__":

    main()