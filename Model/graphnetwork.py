from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Parameter as Param
from torch_scatter import scatter
from torch_sparse import SparseTensor, masked_select_nnz, matmul

from torch_geometric.typing import Adj, OptTensor

from torch_geometric.nn.conv.rgcn_conv import RGCNConv, masked_edge_index
from torch_geometric.nn import GraphConv

from .dialogue_modules import Audio_encoder,Auxilary_Classifier, Global_Feature_Extractor, Local_Feature_Extractor
from .attention import MultiHeadAttention, BidirectionalAttention

#code from https://github.com/thuhcsi/mm2022-conversational-tts
def masked_edge_weight(edge_weight, edge_mask):
    return edge_weight[edge_mask]
class RGCNConv(RGCNConv):
    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]], edge_index: Adj, edge_type: OptTensor = None, edge_weight = None):
        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        for i in range(self.num_relations):
            tmp = masked_edge_index(edge_index, edge_type == i)
            tmp2 = masked_edge_weight(edge_weight, edge_type == i)
            if tmp2.shape[0] == 0:
                continue

            if x_l.dtype == torch.long:
                out += self.propagate(tmp, x=weight[i, x_l], size=size, edge_weight=tmp2)
            else:
                h = self.propagate(tmp, x=x_l, size=size, edge_weight=tmp2)
                out = out + (h @ weight[i])

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

class DialogueGCN(nn.Module):
    def __init__(self, model_config):
        super(DialogueGCN, self).__init__()
        hidden_dim = model_config["GCN"]["hidden_dim"]
        speakers = model_config["speakers"]

        self.global_attention = MultiHeadAttention(hidden_dim, hidden_dim, hidden_dim, 1, hidden_dim)
        self.rgcn = RGCNConv(hidden_dim, hidden_dim, 2 * speakers ** 2)
        self.gcn = GraphConv(hidden_dim, hidden_dim)

        self.edges = [(i, j) for i in range(speakers) for j in range(speakers)]
        edge_types = [[f'{i}{j}0', f'{i}{j}1'] for i in range(speakers) for j in range(speakers)]
        edge_types = [j for i in edge_types for j in i]
        self.edge_type_to_id = {}
        for i, edge_type in enumerate(edge_types):
            self.edge_type_to_id[edge_type] = i

    def forward(self, global_features, speaker):
        edges = torch.tensor(self.edges).T.to(global_features.device)
        edge_type = []
        for i in range(len(speaker)):
            for j in range(len(speaker)):
                direction = 0 if i < j else 1
                edge_type.append(self.edge_type_to_id[f'{speaker[i]}{speaker[j]}{direction}'])
        edge_type = torch.tensor(edge_type).to(global_features.device)

        global_attention_keys = torch.stack([global_features for i in range(len(speaker))])
        _, global_attention_weights = self.global_attention(global_features, global_attention_keys, global_attention_keys)
        edge_weight = torch.flatten(global_attention_weights)

        x = self.rgcn(global_features, edges, edge_type, edge_weight=edge_weight)
        x = self.gcn(x, edges)
        return torch.cat([x, global_features], dim=-1)

#proposed methods
class WiSeGTN(nn.Module):
    #Global, local
    #masking 관련하여 수정이 반드시 필요 masking 빠져 있음
    def __init__(self, model_config):
        super(WiSeGTN, self).__init__()

        hidden_dim = model_config["WiSeGTN"]["TA_Encoder"]["hidden"]
        n_heads = model_config["WiSeGTN"]["n_heads"]

        self.n_top = model_config["WiSeGTN"]["window_size"]
        self.slf_attn = MultiHeadAttention(hidden_dim, hidden_dim, hidden_dim, n_heads, hidden_dim)
        self.past_attn = MultiHeadAttention(hidden_dim, hidden_dim, hidden_dim, n_heads, hidden_dim)
        self.future_attn =MultiHeadAttention(hidden_dim, hidden_dim, hidden_dim, n_heads, hidden_dim)
        self.ReLU1 = nn.ReLU()
        self.ReLU2 = nn.ReLU()

    def top_N(self, global_attn_map):
        b, l, _ = global_attn_map.size()
        [value, indx] = torch.topk(global_attn_map, k = self.n_top, dim = -1, sorted=False)
        new_attention_map = F.softmax(value, dim = -1)
        return new_attention_map, indx

    def wise(self, context, mask, attn_map , indx):
        b, h, l, c = context.size()

        output = context.new_zeros((b, h, l, c))
        for i in range(b): # batch
            for j in range(h): #currnet
                c_context = context[i, j]
                c_attn_map = attn_map[i, j]
                c_indx = indx[i, j]
                for k in range(h): # select
                    if j == c_indx[k]:
                        atten, _ = self.slf_attn(c_context, c_context, c_context)
                    elif j < c_indx[k]:
                        atten, _ = self.future_attn(c_context, context[i, j, c_indx[k]])
                    elif j > c_indx[k]:
                        atten, _ = self.past_attn(c_context, context[i, j, c_indx[k]])

                    output[i, j] = output[i,j] + c_attn_map[k] * atten
        return output

    def forward(self, global_attn_map, history_context, current_context, h_context_mask, c_context_mask, history_mask):
        """
        :param global_attn_map: b, h, h
        :param history_context: b, h, l, c
        :param current_context: b, l, c
        :param h_context_mask: b, h, l
        :param c_context_mask: b, l
        :param history_mask: b, h
        :return:
        """
        dialogue_context = torch.cat([history_context, current_context.unsqueeze(1)], dim = 1)
        dialogue_mask = torch.cat([h_context_mask, c_context_mask.unsqueeze(1)], dim = 1)
        n_attn_map, indx = self.top_N(global_attn_map)
        h_context = self.ReLU1(self.wise(dialogue_context, dialogue_mask, n_attn_map, indx)) + dialogue_context
        h_context = self.ReLU2(self.wise(h_context, dialogue_mask, n_attn_map, indx)) + h_context

        return h_context