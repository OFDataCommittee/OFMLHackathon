import pickle
import torch
import glob
import sys
import matplotlib.pyplot as plt
import natsort
import numpy as np
import os

files = glob.glob(f"{sys.argv[1]}/obser*")

files = natsort.natsorted(files)

mean_r = []
std_r = []
for file_ in files:
    print(file_)
    with open (file_, 'rb') as f:
        data = pickle.load(f)
        s, a, r, p = data
        mean_r.append(torch.cat(r).mean())
        std_r.append(torch.cat(r).std())

ep = list(range(len(mean_r)))
y1 = list(np.subtract(mean_r,std_r))
y2 = list(np.add(mean_r,std_r))

plt.plot(ep, mean_r)
#plt.plot(ep, y1)
#plt.plot(ep, y2)
#plt.fill_between(mean_r, std_r)
plt.fill_between(ep, y1, y2, alpha=0.25)
plt.xlabel('Epoch')
plt.ylabel('Reward')

#name = str(os.getcwd()) + '.png'
#print(name)
plt.savefig('train.png')
