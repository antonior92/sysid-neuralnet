# %%
import torch
import run
import numpy as np
import matplotlib.pyplot as plt
from utils import show_fig

(model, loader, options) = run.run({"cuda": False},
                                     load_model="log/chen/mlp_2/train_Fri Mar  8 12:46:37 2019/best_model.pt")

model.cpu()
model.set_mode('free-run-simulation')
model.eval()


all_output = []
all_y = []
for i, (u, y) in enumerate(loader["test"]):
    with torch.no_grad():
        all_output += [model(u, y).detach()]
        all_y += [y]

all_output = np.concatenate(all_output, 0)
all_y = np.concatenate(all_y, 0)

fig, ax = plt.subplots()
plt.plot(all_y[-1, 0, :])
plt.plot(all_output[-1, 0, :])

show_fig(fig)

print(np.sqrt(np.mean(np.square(all_output - all_y))))


