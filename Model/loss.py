import torch
from torch import nn

class FastSpeech2Loss(nn.Module):
    def __init__(self, preprocess_config, model_config):
