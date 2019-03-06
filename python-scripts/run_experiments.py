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


seqlen_list = [32*2**i for i in range(6)]
batchsize_list = [8*2**i for i in range(6)]
lr_list = [0.001*math.sqrt(0.1)**i for i in range(4)]

for layer1 in tcn_layer1:
    for layer2 in tcn_layer2:
        for layer3 in tcn_layer3:
            n_channels = [layer1]
            if layer2 is None:
                if layer3 is not None:
                    continue
            else:
                n_channels += [layer2]

            if layer3 is not None:
                n_channels += [layer3]

            option_dicts.append({"logdir": "log/tcn_1", "cuda": True,
                                 "dataset": "silverbox", "model": "tcn",
                                 "normalize": True, "normalize_n_std": 1,
                                 "model_options": {"n_channels": n_channels}
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






