from multiprocessing import Process
import run
import time
import math

def sub_run(dict):
    options = run.create_full_options_dict(dict)
    run.run(options, mode_interactive=False)

io_delay = 1
option_dicts = []
channels_list = [16, 32, 64]
n_blocks_list = [1, 2, 4]
ksize_list = [2, 4, 8]
dropout_list = [0, 0.3, 0.8]
for noise_levels in [(0, 0), (0.1, 0.5), (0.8, 0.8)]:
    for n_blocks in n_blocks_list:
        for channels in channels_list:
            n_channels = n_blocks*[channels]
            dilation_sizes = n_blocks*[]
            for ksize in ksize_list:
                for dropout in dropout_list:
                    option_dicts.append({"logdir": "log/chen_example/tcn", "cuda": True,
                                         "dataset": "chen",
                                         "model": "tcn",
                                         "model_options": {"ksize": ksize,
                                                           "n_channels": n_channels,
                                                           "dilation_size": dilation_sizes,
                                                           "dropout": dropout
                                                           },
                                        "dataset_options": {'train': {'sd_v': noise_levels[0],
                                                                      'sd_w': noise_levels[1]},
                                                            'valid': {'sd_v': noise_levels[0],
                                                                      'sd_w': noise_levels[1]},
                                                            'test': {'sd_v': noise_levels[0],
                                                                     'sd_w': noise_levels[1]}}}
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






