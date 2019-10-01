from multiprocessing import Process
import run
import time


def sub_run(dict):
    options = run.create_full_options_dict(dict)
    run.run(options, mode_interactive=False)


option_dicts = []


# LSTM
io_delay = 0
logdir = "log/silverbox_schroeder/lstm"
hidden_size_list = [16, 32, 64, 128]
num_layers_list = [1, 2]
dropout_list = [0]
for num_layers in num_layers_list:
    for hidden_size in hidden_size_list:
        for dropout in dropout_list:
            option_dicts.append({"logdir": logdir,
                                 "cuda": False,
                                 "dataset": "silverbox_schroeder",
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

num_processes = 1
processes = []
while len(option_dicts) > 0:
    opt_dict = option_dicts.pop()
    sub_run(opt_dict)
    # new_p = Process(target=sub_run, args=(opt_dict,))
    # processes.append(new_p)
    # new_p.start()
    # time.sleep(1)
    # while len(processes) >= num_processes:
    #     for p in processes:
    #         if not p.is_alive():
    #             p.join()
    #             processes.remove(p)
    #             break
    #     time.sleep(1)

# Join the last processes
#for p in processes:
#    p.join()
