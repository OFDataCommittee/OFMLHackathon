from src.python.agent import FCPolicy
import torch as pt
import pickle

policy = FCPolicy(100, 1, 0.05, 0.05)
policy.load_state_dict(pt.load("test_training/policy_2.pt"))

with open("test_training/observations_e2.pkl", "rb") as f:
    s, a, r, p = pickle.load(f)

print(s[0].shape)

out = policy(s[0])
print(out)
