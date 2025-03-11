import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from utils.utils import get_mask

PAD = 0
UNK = 1
BOS = 2
EOS = 3
PAD_WORD = "<black>"
UNK_WORD = "<unk>"
BOS_WORD = "<s>"
EOS_WORD = "</s>"

def get_sinusoid_encoder_table(n_position, d_hid, padding_idx = None):
    def cal_angle(position, hid_idx):
        return position/np.power(10000, 2*(hid_idx//2)/d_hid)
    def get_posi_angle_vec(position):
        return [cal_angle(position,hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:,0::2] = np.sin(sinusoid_table[:,0::2])
    sinusoid_table[:,1::2] = np.cos(sinusoid_table[:,1::2])

    if padding_idx is not None:
        sinusoid_table = [padding_idx] = 0.0
    return torch.FloatTensor(sinusoid_table)

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


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in,
                             d_hid,
                             kernel_size=kernel_size[0],
                             padding=(kernel_size[0] - 1) // 2,
                             )
        self.w_2 = nn.Conv1d(d_hid,
                             d_in,
                             kernel_size=kernel_size[1],
                             padding=(kernel_size[1] - 1) // 2,
                             )
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output

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
    def __init__(self, key1_dim, key2_dim, hidden_dim, temperature = 1.0):
        super(BidirectionalAttention, self).__init__()
        self.k1_dim = key1_dim
        self.k2_dim = key2_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.k1_linear = nn.Linear(self.k1_dim, hidden_dim)
        self.k2_linear = nn.Linear(self.k2_dim, hidden_dim)
        self.v1_linear = nn.Linear(self.k2_dim, hidden_dim)
        self.v2_linear = nn.Linear(self.k1_dim, hidden_dim)


    def forward(self, k1, k2, mask_k1, mask_k2):
        """
        :param k1: b, tl, c
        :param k2: b, al, c
        :param v1: b, tl, c
        :param v2: b, al, c
        :param mask_k1: b, l
        :param mask_k2: b, l
        :return:
            output1: b, tl, c
            output2 : b, al, c
        """

        k1 = self.k1_linear(k1).masked_fill(mask_k1, 0)
        k2 = self.k2_linear(k2).masked_fill(mask_k2, 0)
        v1 = self.v1_linear(k2).masked_fill(mask_k2, 0)
        v2 = self.v2_linear(k1).masked_fill(mask_k1, 0)

        atten = torch.bmm(k1, k2.transpose(-1, -2)) # b, tl, al

        atten_mask1 = mask_k1 * mask_k2.transpose(-1,-2)
        atten_mask2 = mask_k2 * mask_k1.transpose(-1,-2)

        atten1 = atten.masked_fill(atten_mask1, -np.inf)
        atten2 = atten.transpose(-1,-2).masked_fill(atten_mask2, -np.inf)

        atten1 = nn.functional.softmax(atten1, dim = -1) # b, tl, al
        atten2 = nn.functional.softmax(atten2, dim = -1) # b, al, tl

        output1 = torch.bmm(v2, atten1.transpose(-1,-2)) # b, al, hidden
        output2 = torch.bmm(v1, atten2.transpose(-1,-2)) # b, tl, hidden

        return output1, output2









