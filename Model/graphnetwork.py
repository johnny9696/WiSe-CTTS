import torch
from torch import nn


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
    def __init__(self,n_mels, hidden_dim, conv_kernel, conv_filter, n_layers, dropout ,level = "local"):
        super(Audio_encoder, self).__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.conv_kernel = conv_kernel
        self.conv_filter = conv_filter
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

class TA_Encoder(nn.Module):
    def __init__(self, n_mels, text_dim, hidden_dim, n_heads, n_layers, level = "local"):
        super(TA_Encoder, self).__init__()
        self.n_mels = n_mels
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.level = level



    def forward(self):
