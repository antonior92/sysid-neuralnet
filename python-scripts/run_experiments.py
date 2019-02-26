from multiprocessing import Process
import run
import time


class tcn_model:
    def __init__(self,  ksize = 3, dropout = 0.2, n_channels = [16, 32], dilation_sizes = [1, 1],  ar= True):
        self.ksize = ksize
        self.dropout = dropout
        self.n_channels = n_channels
        self.dilation_sizes = dilation_sizes
        self.ar = ar


class mlp_model:
    __dict__ = {'hidden_size': 10,
                'max_past_input': 3,
                'ar': True,
                'io_delay': 1}



def sub_run(dict):
    print("running")
    options = run.get_options(None, dict)
    run.run(options, mode='script')

option_dicts = []





mlp_max_past_input_list = [2**i for i in range(5)]
mlp_hidden_size_list = [8*2**i for i in range(6)]
io_delay_list  = [0, 1, 2, 3]

for mlp_hidden_size in mlp_hidden_size_list:
    for mlp_max_past_input in mlp_max_past_input_list:
        for io_delay in io_delay_list:
            option_dicts.append({"logdir": "log/mlp_networks", "cuda":True,
                                 "dataset": "silverbox", "model": "mlp",
                                 "train_options": {},
                                 "mlp_options": {"hidden_size":mlp_hidden_size,
                                                 "max_past_input":mlp_max_past_input,
                                                 "io_delay":io_delay}})


#seqlen_list    = [32*2**i for i in range(6)]
#batchsize_list = [8*2**i  for i in range(6)]


num_processes = 4

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






