import torch
from torch import nn
import numpy as np

from utils.utils import get_mask

class ScaledDotAttention(nn.Module):
    def __init__(self, temperature = 1.0):
        super(ScaledDotAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, q, k, v, mask):
        """
        :param q: b, l, c
        :param k: b, l, c
        :param v: b, l, c
        :param mask: b, l, l
        :return:
            output : b, l, c
            atten, b, l, l
        """
        atten = torch.bmm(q, k.transpose(-1,-2))
        atten = atten/self.temperature
        atten = atten.masked_fill(mask, -np.inf)
        atten = self.softmax(atten)
        output = torch.bmm(atten, v)

        return output, atten


class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, n_head, hidden_dim, dropout = 0.4, temperature = 1.0):
        super(MultiHeadAttention, self).__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.q_linear = nn.Linear(q_dim, hidden_dim*n_head)
        self.k_linear = nn.Linear(k_dim, hidden_dim*n_head)
        self.v_linear = nn.Linear(v_dim, hidden_dim*n_head)

        self.attention = ScaledDotAttention(temperature)

        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(hidden_dim*n_head, hidden_dim)

    def forward(self, q, k, v, mask):
        """
        :param q: b, l, c
        :param k: b, l, c
        :param v: b, l, c
        :param mask: b, l, l
        :return:
            output : b, l, c
            atten : b, l, l
        """

        b, ql, qc = q.size()
        _, kl, kc = k.size()
        _, vl, vc = v.size()

        residual = q

        q = self.q_linear(q).view(b, ql,  self.n_head, self.hidden_dim)
        k = self.k_linear(k).view(b, kl,  self.n_head, self.hidden_dim)
        v = self.v_linear(v).view(b, vl,  self.n_head, self.hidden_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, ql, self.hidden_dim)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, kl, self.hidden_dim)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, vl, self.hidden_dim)

        mask = mask.repeat(self.n_head, 1, 1)
        output, atten = self.attention(q, k, v, mask)

        output = self.dropout(self.out_linear(output))
        output = self.layer_norm(output+residual)

        return output,atten

class BidirectionalAttention(nn.Module):
    def __init__(self, key1_dim, key2_dim, value1_dim, value2_dim, hidden_dim, temperature = 1.0):
        super(BidirectionalAttention, self).__init__()
        self.k1_dim = key1_dim
        self.k2_dim = key2_dim
        self.v1_dim = value1_dim
        self.v2_dim = value2_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.k1_linear = nn.Linear(self.k1_dim, hidden_dim)
        self.k2_linear = nn.Linear(self.k2_dim, hidden_dim)
        self.v1_linear = nn.Linear(self.v1_dim, hidden_dim)
        self.v2_linear = nn.Linear(self.v2_dim, hidden_dim)


    def forward(self, k1, k2, v1, v2, k1_length, k2_length, v1_length, v2_length):
        """
        :param k1: b, l, c
        :param k2: b, l, c
        :param v1: b, l, c
        :param v2: b, l, c
        :param k1_length: b
        :param k2_length: b
        :param v1_length: b
        :param v2_length: b
        :return:
        """

        mask_k1 = get_mask(k1_length).unsqueeze(-1).to(k1.device) # b, l, 1
        mask_k2 = get_mask(k2_length).unsqueeze(-1).to(k1.device) # b, l, 1
        mask_v1 = get_mask(v1_length).unsqueeze(-1).to(k1.device) # b, l, 1
        mask_v2 = get_mask(v2_length).unsqueeze(-1).to(k1.device) # b, l, 1

        k1 = self.k1_linear(k1).masked_fill(mask_k1, 0)
        k2 = self.k2_linear(k2).masked_fill(mask_k2, 0)
        v1 = self.v1_linear(v1).masked_fill(mask_v1, 0)
        v2 = self.v2_linear(v2).masked_fill(mask_v2, 0)




