import matplotlib.pyplot as plt
import numpy as np
from util import LoadData, Load, Save, DisplayPlot

load = np.load('eps0.001.npz')
lst = load.files
stats = {
        'train_ce': ...,
        'valid_ce': ...,
        'train_acc': ...,
        'valid_acc': ...
    }
for item in lst:
    stats[item] = load[item]
plt.figure(0)
plt.plot(stats['train_ce'][:, 0], stats['train_ce'][:, 1], 'b', label='Train')
plt.plot(stats['valid_ce'][:, 0], stats['valid_ce'][:, 1], 'g', label='Validation')
plt.figure(0).savefig("eps_0001_ce.png")
plt.figure(1)
plt.plot(stats['train_acc'][:, 0], stats['train_acc'][:, 1], 'b', label='Train')
plt.plot(stats['valid_acc'][:, 0], stats['valid_acc'][:, 1], 'g', label='Validation')
plt.figure(1).savefig("eps_0001_acc.png")
# plt.savefig("eps0001.png")