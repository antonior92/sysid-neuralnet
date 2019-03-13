# %%
import torch
import run
import numpy as np
import matplotlib.pyplot as plt
from utils import show_fig

(model, loader, options) = run.run({"cuda": False},
                                   load_model="log/chen/tcn_2/train_Sat Mar  9 02:50:38 2019/best_model.pt")

model.cpu()
model.set_mode('one-step-ahead')
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
n = all_y.shape[2]
ax.plot(range(1, n+1), all_y[0, 0, :], label='true value')
ax.plot(range(1, n+1), all_output[0, 0], ls='--', label='simulated value')
ax.set_xlim([0, n])
ax.set_ylabel('$y$', fontsize=16)
ax.set_xlabel('$k$', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('output_chen.pdf', fontsize=16)
show_fig(fig)

rf = model.m.get_requested_input(1)
print(np.sqrt(np.mean(np.square(all_output[rf:] - all_y[rf:]))))


