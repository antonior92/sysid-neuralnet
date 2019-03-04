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

seqlen_list = [32*2**i for i in range(6)]
batchsize_list = [8*2**i for i in range(6)]
lr_list = [0.001*math.sqrt(0.1)**i for i in range(4)]

for max_past_input in mlp_max_past_input_list:
    for hidden_size in mlp_hidden_size_list:
        option_dicts.append({"logdir": "log/mlp_with_normalization_2", "cuda": True,
                             "dataset": "silverbox", "model": "mlp",
                             "normalize": True, "normalize_n_std": 1,
                             "dataset_options": {"seq_len": 512, "seq_len_eval": 512},
                             "model_options": {"max_past_input": max_past_input, "hidden_size": hidden_size}
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






