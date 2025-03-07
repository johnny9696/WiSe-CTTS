import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

def get_mask(length, max_length = None):
    if max_length == None:
        max_length = max(length)

    batch = len(length)
    mask = torch.ones((batch, max_length))
    for i in range(batch):
        mask[i][:length[i]] = torch.zeros(length[i])

    return mask

def pad(input_ele, mel_max_length = None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0)] for i in range(len(input_ele)))

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(batch, (0, max_len-batch.size(0)),"constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(batch,(0,0,0,max_len-batch.size(0)),"constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded