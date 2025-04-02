import os
import json
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


from text.symbols import symbols
from .attention import get_sinusoid_encoder_table, MultiHeadAttention, PositionwiseFeedForward
from .attention import PAD,UNK, BOS, EOS, PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD
from utils.utils import get_mask, pad


#FastSpeech 2 Modules
class Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 1,
                 stride = 1,
                 padding= 0,
                 dilation = 1,
                 bias = True,
                 w_init = 'linear'):
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1,2)
        x = self.conv(x).contiguous().transpose(1,2)
        return x

class ConvNorm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 1,
                 stride = 1,
                 padding = None,
                 dilation = 1,
                 bias = True,
                 w_init_gain = "linear",):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size %2 == 1
            padding = int(dilation * (kernel_size - 1)/2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            bias= bias,
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class FFTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout = 0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(d_k, d_k, d_v, n_head, d_model, dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout
        )
    def forward(self, enc_input, mask=None, slf_attn_mask = None):
        enc_output, enc_slf_atten = self.slf_attn(
            enc_input, enc_input, enc_input, slf_attn_mask
        )
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        enc_output = self.pos_ffn(enc_output).masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_atten

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        n_position = config["max_seq_len"]+1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["Encoder"]["hidden"]
        n_head = config["Encoder"]["n_heads"]
        n_layers = config["Encoder"]["layers"]
        d_k = d_v = config["Encoder"]["hidden"]
        d_model = config["Encoder"]["hidden"]
        d_inner = config["Encoder"]["conv_filter"]
        kernel_size = config["Encoder"]["conv_kernel"]
        dropout = config["Encoder"]["dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=PAD)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoder_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad = False
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout
                )
                for _ in range(n_layers)
            ]
        )
    def forward(self, src_seq, mask, return_attns = False):
        """
        :param src_seq: b, l, c
        :param mask: b, l
        :param return_attns: False
        :return:
            enc_output : b, l, c
        """
        enc_slf_attn_list=[]
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        slf_attn_mask = mask.unsqueeze(1).expand(-1,max_len, -1) # b, l, l

        enc_output = self.src_word_emb(src_seq) + self.position_enc[:,:max_len,:].expand(batch_size, -1,-1)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask, slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list

        else:
            return enc_output


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        n_position = config["max_seq_len"] +1
        d_word_vec = config["Decoder"]["hidden"]
        d_model = config["Decoder"]["hidden"]
        n_layers = config["Decoder"]["layers"]
        n_heads = config["Decoder"]["n_heads"]
        d_k = d_v = d_model
        d_inner = config["Decoder"]["conv_filter"]
        kernel_size = config["Decoder"]["conv_kernel"]
        dropout = config["Decoder"]["dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoder_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_heads, d_k, d_v, d_inner, kernel_size, dropout
                )
                for _ in range(n_layers)
            ]
        )
    def forward(self, src_seq, mask, return_attn = False):
        dec_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) #b, l, l
        dec_output = src_seq + self.position_enc[:,:max_len,:].expand(batch_size, -1, -1)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(dec_output, mask, slf_attn_mask)
            if return_attn:
                dec_slf_attn_list += [dec_slf_attn]

        if return_attn:
            return dec_output, dec_slf_attn_list
        else:
            return dec_output


class VariancePredictor(nn.Module):
    def __init__(self, config):
        super(VariancePredictor, self).__init__()
        self.input_size = config["Encoder"]["hidden"]
        self.filter_size = config["variance_predictor"]["filter_size"]
        self.kernel_size = config["variance_predictor"]["kernel_size"]
        self.conv_output_size = config["variance_predictor"]["filter_size"]
        self.dropout = config["variance_predictor"]["dropout"]

        self.conv_layers = nn.Sequential(
            OrderedDict(
                [
                    ("conv1d_1", Conv(self.input_size, self.filter_size, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)),
                    ("relu_1", nn.ReLU()),
                    ("layernorm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    ("conv1d_2", Conv(self.filter_size, self.filter_size, kernel_size=self.kernel_size,
                                      padding=1)),
                    ("relu_2", nn.ReLU()),
                    ("layernorm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout))
                ]
            )
        )

        self.out_linear = nn.Linear(self.conv_output_size,1)
    def forward(self, enc_output, mask = None):
        out = self.conv_layers(enc_output)
        out = self.out_linear(out).squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out

class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self,  x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)
        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size),0),-1))
        out = torch.cat(out,0)
        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len

class VarianceAdaptor(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        assert self.pitch_feature_level in  ["phoneme_level","frame_level"]
        assert self.energy_feature_level in ["phoneme_level","frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad = False
            )

        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad = False
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["Encoder"]["hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["Encoder"]["hidden"]
        )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding
    def forward(self,
                x,
                src_mask,
                mel_mask=None,
                max_len=None,
                pitch_target =None,
                energy_target =None,
                duration_target = None,
                p_control = 1.0,
                e_control = 1.0,
                d_control = 1.0):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, e_control
            )
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction)-1)*d_control),
                min = 0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask(mel_len).to(x.device)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, e_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask
        )


class PostNet(nn.Module):
    def __init__(self,
                 n_mel_channels = 80,
                 postnet_embedding_dim = 512,
                 postnet_kernel_size = 5,
                 postnet_n_convolutions = 5):
        super(PostNet,self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride = 1,
                    padding = int((postnet_kernel_size - 1)/2),
                    dilation = 1,
                    w_init_gain = "tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride = 1,
                    padding = int((postnet_kernel_size - 1)/2),
                    dilation = 1,
                    w_init_gain = "tanh",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )
    def forward(self, x):
        x = x.contiguous().transpose(1,2)

        for i in range(len(self.convolutions)-1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)),0.5, self.training)
        x = F.dropout(torch.tanh(self.convolutions[-1](x)),0.5, self.training)
        x = x.contiguous().transpose(1,2)
        return x
