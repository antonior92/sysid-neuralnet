from multiprocessing import Process
import run
import time
import math

def sub_run(dict):
    options = run.create_full_options_dict(dict)
    run.run(options, mode_interactive=False)


option_dicts = []
mlp_max_past_input_list = [2**i for i in range(7)]
mlp_hidden_size_list = [4*2**i for i in range(7)]
io_delay_list = [0, 1, 2, 3]

tcn_layer1 = [4*2**i for i in range(4)]
tcn_layer2 = [4*2**i for i in range(4)]
tcn_layer3 = [4*2**i for i in range(4)]

tcn_layer2 += [None]
tcn_layer3 += [None]

dropout_list = [0, 0.05, 0.1, 0.2]

seqlen_list = [32*2**i for i in range(6)]

lr_list = [0.001*math.sqrt(0.1)**i for i in range(4)]

lstm_size_list = [4*i**2 for i in range(5)]
batchsize_list = [1*2**i for i in range(6)]


for max_past_input in mlp_max_past_input_list:
    for hidden_size in mlp_hidden_size_list:
            option_dicts.append({"logdir": "log/mlp_without_normalization_1", "cuda": True,
                                 "dataset": "silverbox", "model": "mlp",
                                 "normalize": False, "normalize_n_std": 1,
                                 "train_options": {"batch_size": 4, "lr_scheduler_factor": 2},
                                 "model_options": {'hidden_size': hidden_size,
                                                   'max_past_input': max_past_input,
                                                   'activation_fn': 'sigmoid'}
                                 }
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






