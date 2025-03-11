import torch
from torch import nn

from attention import MultiHeadAttention, BidirectionalAttention, PositionwiseFeedForward

class Auxilary_Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout = 0.3, level = "local"):
        super(Auxilary_Classifier, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.level = level
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x, mask = None):
        if self.level == "local" :
            x = x.masked_fill(mask.unsqueeze(-1).expand(-1, -1, self.input_dim), 0)
            x = x.sum(dim = -1) / (mask.sum(dim=-1) + 1e-8)

        x = self.classifier(x)
        output = self.linear(x)
        return output


class Audio_encoder(nn.Module):
    def __init__(self,n_mels, hidden_dim, conv_kernel, n_layers, dropout ,level = "local"):
        super(Audio_encoder, self).__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.conv_kernel = conv_kernel
        self.n_layers = n_layers
        self.dropout = dropout
        self.level = level

        self.prenet = nn.Sequential(
            nn.Conv2d(n_mels,
                      hidden_dim,
                      kernel_size = self.conv_kernel,
                      stride = (self.conv_kernel - 1) // 2,
                      padding = 0),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.conv_list = nn.ModuleList()
        for _ in range(self.n_layers - 1):
            self.conv_list.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim,
                              hidden_dim,
                              kernel_size = self.conv_kernel,
                              stride = (self.conv_kernel - 1)//2,
                              padding = 0 ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(hidden_dim,
                              hidden_dim,
                              kernel_size=self.conv_kernel,
                              stride=(self.conv_kernel - 1) // 2,
                              padding=0),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask):
        """
        :param x: b, l, c
        :param mask: b, l
        :return:
            global : b, c
            local  : b, l, c
        """

        mel_mask = mask.unsqueeze(1).expand(-1,self.n_mels, -1)
        output = self.prenet(x.transpose(-1,-2)).masked_fill(mel_mask, 0)

        residual = output
        hidden_mask = mask.unsqueeze(1).expand(-1,self.hidden_dim,-1)
        for i in range(self.n_layers):
            output = self.conv_list(output).masked_fill(hidden_mask,  0) + residual
            residual = output

        if self.level == "global":
            output = output.sum(dim = -1) / (mask.sum(dim=-1) + 1e-8)
            output = self.out_linear(output)
        elif self.level == "local":
            output = output.transpose(1, 2)
            output = self.out_linear(output).masked_fill(mask.unsqueeze(-1).expand(-1, -1, self.hidden_dim), 0)

        return output


class Global_Feature_Extractor(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(Global_Feature_Extractor, self).__init__()
        self.n_mels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.text_dim = model_config["WiSeGCN"]["TA_Encoder"]["text_dim"]
        self.hidden_dim = model_config["WiSeGCN"]["TA_Encoder"]["hidden"]
        self.n_layers = model_config["WiSeGCN"]["TA_Encoder"]["n_layers"]
        self.A_conv_kernel = model_config["WiSeGCN"]["TA_Encoder"]["Audio_Encoder"]["conv_kernel"]
        self.A_n_layers = model_config["WiSeGCN"]["TA_Encoder"]["Audio_Encoder"]["n_layers"]
        self.A_dropout = model_config["WiSeGCN"]["TA_Encoder"]["Audio_Encoder"]["dropout"]
        self.n_speaker = model_config["WiSeGCN"]["TA_Encoder"]["Auxilary_Task"]["speakers"]
        self.n_emotion = model_config["WiSeGCN"]["TA_Encoder"]["Auxilary_Task"]["emotion"]
        self.n_act = model_config["WiSeGCN"]["TA_Encoder"]["Auxilary_Task"]["act"]


        self.audio_encoder = Audio_encoder(self.n_mels, self.hidden_dim, self.A_conv_kernel, self.A_n_layers, self.A_dropout, level = "global")
        self.audio_aux_speaker = Auxilary_Classifier(self.hidden_dim, self.n_speaker, level="global")
        self.audio_aux_emotion = Auxilary_Classifier(self.hidden_dim, self.n_emotion, level="global")
        self.audio_aux_act = Auxilary_Classifier(self.hidden_dim, self.n_act, level="global")

        self.text_linear = nn.Linear(self.text_dim, self.hidden_dim)

        self.alpha_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        self.attention = MultiHeadAttention(self.hidden_dim, self.hidden_dim, self.hidden_dim, 1, self.hidden_dim)

    def forward(self, sbert, mel, history_mask, audio_mask):
        """
        :param sbert: b, h, c
        :param mel: b, h, l, c
        :param history_mask: b, h
        :param audio_mask: b, h, l
        :return:
        """
        b, h,_ = sbert.size()
        _,_,l,c =mel.size()
        sbert = sbert.contiguous().view(b * h, -1)
        mel = mel.contiguous().view(b*h, l, c)
        history_mask = history_mask.unsqueeze(-1).contiguous().view(b*h,1).expand(-1,self.hidden_dim)
        audio_mask = audio_mask.contiguous().view(b*h, l)

        text = self.text_linear(sbert).masked_fill(history_mask, 0)
        audio = self.audio_encoder(mel, audio_mask)

        emotion_class = self.audio_aux_emotion(audio, history_mask)
        act_class = self.audio_aux_act(audio, history_mask)
        speaker_class = self.audio_aux_speaker(audio, history_mask)

        fusion_weight = self.alpha_mlp(torch.cat(text, audio), dim = -1)
        fusion_vector = fusion_weight * text + (1-fusion_weight) * audio
        fusion_vector = fusion_vector.contiguous().view(b, h, self.hidden) # b, h, c

        slf_attn_mask = history_mask.unsqueeze(1).expand(-1, h, -1)
        output, attn = self.attention(fusion_vector, fusion_vector, fusion_vector, slf_attn_mask)
        return output, attn , emotion_class, act_class, speaker_class

class Local_Feature_Extractor(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(Local_Feature_Extractor,self).__init__()
        self.text_dim = model_config["WiSeGCN"]["TA_Encoder"]["text_dim"]
        self.n_mels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.hidden_dim = model_config["WiSeGCN"]["TA_Encoder"]["hidden"]
        self.n_layers = model_config["WiSeGCN"]["TA_Encoder"]["n_layers"]
        self.A_conv_kernel = model_config["WiSeGCN"]["TA_Encoder"]["Audio_Encoder"]["conv_kernel"]
        self.A_n_heads = model_config["WiSeGCN"]["TA_Encoder"]["Audio_Encoder"]["n_heads"]
        self.A_n_layers = model_config["WiSeGCN"]["TA_Encoder"]["Audio_Encoder"]["n_layers"]
        self.A_dropout = model_config["WiSeGCN"]["TA_Encoder"]["Audio_Encoder"]["dropout"]


        self.text_linear = nn.Linear(self.text_dim, self.hidden_dim)
        self.audio_encoder = Audio_encoder(self.n_mels, self.hidden_dim, self.A_conv_kernel, self.A_n_layers, self.A_dropout, level="local")

        self.bi_attention = nn.ModuleList()
        self.T_PWB = nn.ModuleList()
        self.A_PWB = nn.ModuleList()

        for _ in range(self.n_layers):
            self.bi_attention.append(BidirectionalAttention(self.hidden_dim, self.hidden_dim, self.hidden_dim))
            self.T_PWB.append(PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, kernel_size=[9,1], dropout = self.A_dropout))
            self.A_PWB.append(PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, kernel_size=[9,1], dropout = self.A_dropout))

        self.alpha_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def t_a_align(self, audios, alignments):
        """
        :param audio: b*h, l, hidden
        :param alignment: b, h, l
        :return:
        """
        b, h, l = alignments.size()
        alignments = alignments.contiguous().view(b*h, -1)

        output = torch.zeros((b*h, l, self.hidden_dim)).to(audios.device)
        for i in range(b*h):
            audio = audios[i]
            alignment = alignments[i]
            start = 0
            for j in range(l):
                output[i, j] = torch.mean(audio[start:start+alignment[j]])
                start = start + alignment[j]
        return output


    def forward(self, text, audio, history_mask, text_mask, audio_mask, text_audio_align):
        """
        :param text: b, h, l, c
        :param audio: b, h, l, n_mel
        :param history_mask: b, h
        :param text_mask: b, h, l
        :param audio_mask: b, h, l
        :param text_audio_align : b, h, l
        :return:
        """
        b, h, tl, c = text.size()
        _, _, al, n_mels = audio.sze()
        text = text.contiguous().view(b*h, tl, c)
        text_mask = text_mask.contiguous.view(b*h, -1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        audio = audio.contiguous().view(b*h, al, n_mels)
        audio_mask = audio_mask.contiguous.view(b * h, -1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        history_mask = history_mask.unsqueeze(-1).expand(-1, -1, self.hidden_dim)

        text = self.text_linear(text).masked_fill(text_mask, 0)
        audio = self.audio_encoder(audio, audio_mask)

        audio = self.t_a_align(audio, text_audio_align)

        for i in range(self.n_layers):
            text_out, audio_out = self.bi_attention[i](text, audio, text_mask, text_mask)
            audio = self.A_PWB[i](audio_out) + audio
            text = self.T_PWB[i](text_out) + text

        fusion_weight = self.alpha_mlp(torch.cat[text, audio], dim = -1)
        ones = torch.ones_like(fusion_weight)
        fusion_vector = fusion_weight * text + (ones - fusion_weight) * audio

        fusion_vector = fusion_vector.contiguous().view(b, h, -1, -1)
        return fusion_vector


class VoiceAgent(nn.Module):
    def __init__(self):
        super(VoiceAgent, self).__init__()

    def forward(self):

class GCN_based(nn.Module):
    def __init__(self):
        super(GCN_based, self).__init__()
    def forward(self):

class GAT_based(nn.Module):
    def __init__(self):
        super(GAT_based, self)
    def forward(self):

class MRS_GCN(nn.Module):
    def __init__(self):
        super(MRS_GCN, self).__init__()

    def forward(self):

class WiSeGCN(nn.Module):
    def __init__(self):
        super(WiSeGCN, self).__init__()

    def forward(self):