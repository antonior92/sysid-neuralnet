from multiprocessing import Process
import run
import time


def sub_run(dict):
    options = run.create_full_options_dict(dict)
    run.run(options, mode_interactive=False)


option_dicts = []


# TCN
io_delay = 1
ksize = 2
logdir = "log/chen_example/tcn_2"
channels_list = [16, 32, 64, 128, 256]
n_blocks_list = [1, 2, 4, 8]
ds_list = [2]
dropout_list = [0, 0.3, 0.5, 0.8]
noise_levels_list = [(0, 0), (0.3, 0.3), (0.6, 0.6)]
n_batches_list = [5, 20, 80]
normalization_list = ['none', 'batch_norm', 'weight_norm']
for n_batches in n_batches_list:
    for noise_levels in noise_levels_list:
        for n_blocks in n_blocks_list:
            for channels in channels_list:
                for ds in ds_list:
                    n_channels = n_blocks*[channels]
                    dilation_sizes = [ds**i for i in range(len(n_blocks))]
                    for dropout in dropout_list:
                        for normalization in normalization_list:
                            option_dicts.append({"logdir": logdir,
                                                 "cuda": True,
                                                 "dataset": "chen",
                                                 "model": "tcn",
                                                 "train_options": {"batch_size": 2},
                                                 "model_options": {"ksize": ksize,
                                                                   "n_channels": n_channels,
                                                                   "dilation_sizes": dilation_sizes,
                                                                   "dropout": dropout,
                                                                   "normalization": normalization
                                                                   },
                                                "dataset_options": {'seq_len': 100,
                                                                    'train': {'ntotbatch': n_batches,
                                                                              'sd_v': noise_levels[0],
                                                                              'sd_w': noise_levels[1]},
                                                                    'valid': {'ntotbatch': 2,
                                                                              'sd_v': noise_levels[0],
                                                                              'sd_w': noise_levels[1]},
                                                                    'test': {'ntotbatch': 10,
                                                                             'sd_v': 0.0,
                                                                             'sd_w': 0.0}}}
                                                )


# MLP
io_delay = 1
logdir = "log/chen_example/mlp_2"
hidden_size_list = [16, 32, 64, 128, 256]
max_past_input_list = [2, 3, 4]
noise_levels_list = [(0, 0), (0.3, 0.3), (0.6, 0.6)]
n_batches_list = [5, 20, 80]
activation_fn_list = []
for n_batches in n_batches_list:
    for noise_levels in noise_levels_list:
        for max_past_input in max_past_input_list:
            for hidden_size in hidden_size_list:
                for activation_fn in activation_fn_list:
                    option_dicts.append({"logdir": logdir,
                                         "cuda": True,
                                         "dataset": "chen",
                                         "model": "mlp",
                                         "train_options": {"batch_size": 2},
                                         "model_options": {"max_past_input": max_past_input,
                                                           "hidden_size": hidden_size,
                                                           "io_delay": io_delay,
                                                           "activation_fn": activation_fn
                                                           },
                                        "dataset_options": {'seq_len': 100,
                                                            'train': {'ntotbatch': n_batches,
                                                                      'sd_v': noise_levels[0],
                                                                      'sd_w': noise_levels[1]},
                                                            'valid': {'ntotbatch': 2,
                                                                      'sd_v': noise_levels[0],
                                                                      'sd_w': noise_levels[1]},
                                                            'test': {'ntotbatch': 10,
                                                                     'sd_v': 0.0,
                                                                     'sd_w': 0.0}}}
                                        )


# LSTM
io_delay = 1
logdir = "log/chen_example/lstm"
hidden_size_list = [16, 32, 64, 128]
num_layers_list = []  # [1, 2, 3]
dropout_list = [0, 0.3, 0.5, 0.8]
noise_levels_list = [(0, 0), (0.3, 0.3), (0.6, 0.6)]
n_batches_list = [5, 20, 80]
for n_batches in n_batches_list:
    for noise_levels in noise_levels_list:
        for num_layers in num_layers_list:
            for hidden_size in hidden_size_list:
                for dropout in dropout_list:
                    option_dicts.append({"logdir": logdir,
                                         "cuda": True,
                                         "dataset": "chen",
                                         "model": "lstm",
                                         "train_options": {"batch_size": 2},
                                         "model_options": {'hidden_size': hidden_size,
                                                           'io_delay': io_delay,
                                                           'num_layers': num_layers,
                                                           'dropout': dropout},
                                        "dataset_options": {'seq_len': 100,
                                                            'train': {'ntotbatch': n_batches,
                                                                      'sd_v': noise_levels[0],
                                                                      'sd_w': noise_levels[1]},
                                                            'valid': {'ntotbatch': 2,
                                                                      'sd_v': noise_levels[0],
                                                                      'sd_w': noise_levels[1]},
                                                            'test': {'ntotbatch': 10,
                                                                     'sd_v': 0.0,
                                                                     'sd_w': 0.0}}}
                                        )





num_processes = 8
processes = []
while len(option_dicts) > 0:
    opt_dict = option_dicts.pop()
    new_p = Process(target=sub_run, args=(opt_dict,))
    processes.append(new_p)
    new_p.start()
    time.sleep(1)
    while len(processes) >= num_processes:
        for p in processes:
            if not p.is_alive():
                p.join()
                processes.remove(p)
                break
        time.sleep(1)

# Join the last processes
for p in processes:
    p.join()






