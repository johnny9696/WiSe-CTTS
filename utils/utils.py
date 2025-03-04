import torch
import numpy as np
from torch import nn

def get_mask(length, max_length = None):
    if max_length == None:
        max_length = max(length)

    batch = len(length)
    mask = torch.ones((batch, max_length))
    for i in range(batch):
        mask[i][:length[i]] = torch.zeros(length[i])

    return mask