# %%
import torch
import run
import numpy as np
import matplotlib.pyplot as plt
from utils import show_fig

(model, loader, options) = run.run({"cuda": False},
                                   load_model="log/chen/tcn_2/train_Sat Mar  9 02:50:38 2019/best_model.pt")

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

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(all_y[1, 0, :])
ax.plot(all_output[1, 0], ls='--')

show_fig(fig)

print(np.sqrt(np.mean(np.square(all_output - all_y))))


