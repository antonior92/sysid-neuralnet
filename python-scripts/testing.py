import run
import numpy as np
import matplotlib.pyplot as plt
import data.loader as loader

(model, _, options) = run.run({"cuda":False, 'dataset_options': {'seq_len_eval': None}},
                              load_model="log/train_Wed Feb 27 14:54:48 2019/best_model.pt")

model.cpu()
model.set_mode("free-run-simulation")

loaders = loader.load_dataset("silverbox", {'seq_len': 1000, 'seq_len_eval': None}, 10, 10)

all_output = []
all_y = []
for i, (u, y) in enumerate(loaders["test"]):
    all_output += [model(u, y).detach()]
    all_y += [y]

all_output = np.concatenate(all_output, 0)
all_y = np.concatenate(all_y, 0)

plt.plot(all_y[-1, 0, -5000:])
plt.plot(all_output[-1, 0, -5000:])
plt.plot(all_output[-1, 0, :]-all_y[-1, 0, :])

plt.show()

print(np.sqrt(np.mean(np.square(all_output - all_y)))*1000)

