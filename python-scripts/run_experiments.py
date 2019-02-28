from multiprocessing import Process
import run
import time
import math


def sub_run(dict):
    options = run.create_full_options_dict(dict)
    run.run(options, mode_interactive=False)


option_dicts = []
mlp_max_past_input_list = [32*2**i for i in range(5)]
mlp_hidden_size_list = [8*2**i for i in range(6)]
io_delay_list = [0, 1, 2, 3]

seqlen_list = [32*2**i for i in range(6)]
batchsize_list = [8*2**i for i in range(6)]
lr_list = [0.001*math.sqrt(0.1)**i for i in range(4)]

for seqlen in seqlen_list:
    for batch_size in batchsize_list:
        for lr in lr_list:
            option_dicts.append({"logdir": "log/batchsizes", "cuda": True,
                                 "dataset": "silverbox", "model": "mlp",
                                 "train_options": {"batch_size": batch_size, "init_lr": lr},
                                 "dataset_options": {"seq_len": seqlen}
                                 }
                                )


num_processes = 1
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






