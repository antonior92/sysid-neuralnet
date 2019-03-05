from multiprocessing import Process
import run
import time
import math

def sub_run(dict):
    options = run.create_full_options_dict(dict)
    run.run(options, mode_interactive=False)


option_dicts = []
mlp_max_past_input_list = [i for i in range(10)]
mlp_hidden_size_list = [4*2**i for i in range(7)]
io_delay_list = [0, 1, 2]
for io_delay in io_delay_list:
    for max_past_input in mlp_max_past_input_list:
        for hidden_size in mlp_hidden_size_list:
            option_dicts.append({"logdir": "log/chen_example/mlp_networks_1", "cuda": True,
                                 "dataset": "chen",
                                 "model": "mlp",
                                 "model_options": {"max_past_input": max_past_input,
                                                   "hidden_size": hidden_size,
                                                   "io_delay": io_delay},
                                "dataset_options": {'train': {'sd_v': 0,
                                                              'sd_w': 0},
                                                    'valid': {'sd_v': 0,
                                                              'sd_w': 0},
                                                    'test': {'sd_v': 0,
                                                             'sd_w': 0}}}
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






