
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


def sub_run(dict):
    print("running")
    options = run.get_options(None, dict)
    run.run(options, mode='script')

option_dicts = []

n_channels_list = [[4, 4], [8, 8], [16, 16], [32, 32]]
for i, n_channels in enumerate(n_channels_list):
    tc_options = tcn_model(n_channels=n_channels, dilation_sizes=None)

    option_dicts.append({"run_name":"tc_channels_"+str(i),
                         "dataset": "silverbox","model": "tcn", "tcn_options": vars(tc_options)})


processes = []
while len(option_dicts) > 0:
    opt_dict = option_dicts.pop()
    new_p = Process(target=sub_run, args=(opt_dict,))
    processes.append(new_p)
    new_p.start()
    time.sleep(1)
    while len(processes) > 3:
        for p in processes:
            if not p.is_alive():
                p.join()
                processes.remove(p)
                break
        time.sleep(1)

# Join the last processes
for p in processes:
    p.join()






