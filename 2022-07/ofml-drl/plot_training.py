from os.path import join
import pickle
import matplotlib.pyplot as plt

training_path = "test_training"

ep_start = 20
ep_end = 30
r_ep = []
for i in range(ep_start, ep_end):
    with open(join(training_path, f"observations_e{i}.pkl"), "rb") as f:
        obs = pickle.load(f)
        s, a, r, p = obs
        r_ep.append(p)

fig, axarr = plt.subplots(ep_end - ep_start, 1, figsize=(6, (ep_end - ep_start)*1.5))

for i in range(ep_end - ep_start):
    for j, r_j in enumerate(r_ep[i]):
        axarr[i].plot(range(len(r_j)), r_j, label=f"tr {i}")
        axarr[i].legend()
plt.show()