import torch

torch.set_num_threads(2)   # the models are tiny; many threads only contend on a shared machine
