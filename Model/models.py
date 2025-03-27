# all the model in this code
import os
import json
import torch
from torch import nn
from torch.nn import functional as F

from basemodule import Encoder, Decoder, VarianceAdaptor, PostNet
from dialogue_modules import VoiceAgent
from graphnetwork import DialogueGCN, WiSeGTN
from glow import WaveGlow
from utils.utils import get_mask


class FastSpeech2(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        self.preprocess_config = preprocess_config

        self.encoder = Encoder(model_config)
        self.varianceadaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["Decoder"]["hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocess_path"],"speaker.json"
                ),
                "r"
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(n_speaker, model_config["Encoder"]["hidden"])


    def forward(self,
                texts,
                src_lens,
                max_src_lens,
                speakers = None,
                mels=None,
                mel_lens = None,
                max_mel_len = None,
                p_targets = None,
                e_targets = None,
                d_targets = None,
                p_control = 1.0,
                e_control = 1.0,
                d_control = 1.0
                ):
        src_masks = get_mask(src_lens, max_src_lens)
        mel_masks = get_mask(mel_lens, max_mel_len) if mel_lens is not None else None

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_lens, -1)


        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks
        )= self.varianceadaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )

class FastSpeech2_Glow(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2_Glow, self).__init__()
        self.model_config = model_config
        self.preprocess_config = preprocess_config

        self.encoder = Encoder(model_config)
        self.varianceadaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["Decoder"]["hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        )
        self.postnet = WaveGlow(preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                                model_config["Glow"]["n_flows"],
                                model_config["Glow"]["n_group"],
                                model_config["Glow"]["n_early_every"],
                                model_config["Glow"]["n_early_size"],
                                model_config["Glow"]["WN_config"]["m_layers"],
                                model_config["Glow"]["WN_config"]["n_channels"],
                                model_config["Glow"]["WN_config"]["kernel_size"])

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocess_path"],"speaker.json"
                ),
                "r"
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(n_speaker, model_config["Encoder"]["hidden"])


    def forward(self,
                texts,
                src_lens,
                max_src_lens,
                speakers = None,
                mels=None,
                mel_lens = None,
                max_mel_len = None,
                p_targets = None,
                e_targets = None,
                d_targets = None,
                p_control = 1.0,
                e_control = 1.0,
                d_control = 1.0,
                training = True
                ):
        src_masks = get_mask(src_lens, max_src_lens)
        mel_masks = get_mask(mel_lens, max_mel_len) if mel_lens is not None else None

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_lens, -1)


        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks
        )= self.varianceadaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        if training:
            postnet_output, log_s_list, log_det_w_list = self.postnet(output)
        else:
            postnet_output = self.postnet.infer(output)
            log_s_list = None
            log_det_w_list = None

        return (
            output,
            postnet_output,
            log_s_list,
            log_det_w_list,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )


#history context 입력 필요
class VoiceAgent_Glow(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(VoiceAgent_Glow, self).__init__()
        self.model_config = model_config
        self.preprocess_config = preprocess_config

        self.encoder = Encoder(model_config)
        self.voiceagent = VoiceAgent(model_config)
        self.varianceadaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["Decoder"]["hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        )
        self.postnet = WaveGlow(preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                                model_config["Glow"]["n_flows"],
                                model_config["Glow"]["n_group"],
                                model_config["Glow"]["n_early_every"],
                                model_config["Glow"]["n_early_size"],
                                model_config["Glow"]["WN_config"]["m_layers"],
                                model_config["Glow"]["WN_config"]["n_channels"],
                                model_config["Glow"]["WN_config"]["kernel_size"])

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocess_path"],"speaker.json"
                ),
                "r"
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(n_speaker, model_config["Encoder"]["hidden"])


    def forward(self,
                texts,
                src_lens,
                max_src_lens,
                speakers = None,
                current_text = None,
                history_text = None,
                history_speaker = None,
                history_len = None,
                mels=None,
                mel_lens = None,
                max_mel_len = None,
                p_targets = None,
                e_targets = None,
                d_targets = None,
                p_control = 1.0,
                e_control = 1.0,
                d_control = 1.0,
                training = True
                ):
        src_masks = get_mask(src_lens, max_src_lens)
        mel_masks = get_mask(mel_lens, max_mel_len) if mel_lens is not None else None

        output = self.encoder(texts, src_masks)

        dialogue_condition = self.voiceagent(current_text, speakers, history_text, history_speaker, history_len)
        output = output + dialogue_condition.unsqueeze(1).expand(-1, max_src_lens, -1)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_lens, -1)


        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks
        )= self.varianceadaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        if training:
            postnet_output, log_s_list, log_det_w_list = self.postnet(output)
        else:
            postnet_output = self.postnet.infer(output)
            log_s_list = None
            log_det_w_list = None

        return (
            output,
            postnet_output,
            log_s_list,
            log_det_w_list,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )

