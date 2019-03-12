from multiprocessing import Process
import run
import time


def sub_run(dict):
    options = run.create_full_options_dict(dict)
    run.run(options, mode_interactive=False)


option_dicts = []


# TCN
io_delay = 0
ksize_list = [2, 4, 8, 16]
logdir = "log/f16gvt/tcn"
channels_list = [16, 32, 64, 128]
n_blocks_list = [1, 2, 4, 8]
ds_list = [1, 2]
dropout_list = [0, 0.3, 0.5, 0.8]
normalization_list = ['none', 'batch_norm', 'weight_norm']

for normalization in normalization_list:
    for ksize in ksize_list:
        for n_blocks in n_blocks_list:
            for channels in channels_list:
                for ds in ds_list:
                    n_channels = n_blocks*[channels]
                    dilation_sizes = [ds**i for i in range(len(n_blocks))]
                    for dropout in dropout_list:
                        option_dicts.append({"logdir": logdir,
                                             "cuda": True,
                                             "dataset": "f16gvt",
                                             "model": "tcn",
                                             "train_options": {"batch_size": 2},
                                             "model_options": {"ksize": ksize,
                                                               "n_channels": n_channels,
                                                               "dilation_sizes": dilation_sizes,
                                                               "dropout": dropout,
                                                               "normalization": normalization
                                                               },
                                             "dataset_options": {'seq_len_train': 2048,
                                                                 'seq_len_val': 2048,
                                                                 'seq_len_test': None}}
                                            )


# MLP
io_delay = 0
logdir = "log/f16gvt/mlp"
hidden_size_list = [16, 32, 64, 128, 256]
max_past_input_list = [2, 4, 8, 16, 32, 64, 128]
activation_fn_list = ['relu', 'sigmoid']
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
                                "dataset_options": {'seq_len_train': 2048,
                                                    'seq_len_val': 2048,
                                                    'seq_len_test': None}}
                                )


# LSTM
io_delay = 0
logdir = "log/f16gvt/lstm"
hidden_size_list = [16, 32, 64, 128]
num_layers_list = [1, 2, 3]
dropout_list = [0, 0.3, 0.5, 0.8]
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
                                 "dataset_options": {'seq_len_train': 2048,
                                                     'seq_len_val': 2048,
                                                     'seq_len_test': None}}
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