# all the model in this code
import os
import json
import torch
from torch import nn
from torch.nn import functional as F

from .attention import MultiHeadAttention
from .basemodule import Encoder, Decoder, VarianceAdaptor, PostNet
from .dialogue_modules import VoiceAgent, Audio_encoder
from .graphnetwork import DialogueGCN, WiSeGTN
from .glow import FlowPostNet
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
                    preprocess_config["path"]["preprocessed_path"],"speakers.json"
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
        src_masks = get_mask(src_lens, max_src_lens).to(texts.device)
        mel_masks = get_mask(mel_lens, max_mel_len).to(texts.device) if mel_lens is not None else None



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

        output = self.decoder(output, mel_masks)
        output = self.mel_linear(output).masked_fill(mel_masks.unsqueeze(-1).expand(-1, -1, 80), 0)

        postnet_output = self.postnet(output) + output
        postnet_output = postnet_output.masked_fill(mel_masks.unsqueeze(-1).expand(-1, -1, 80), 0)
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
        self.postnet = FlowPostNet(preprocess_config, model_config)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"],"speakers.json"
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
                ):
        src_masks = get_mask(src_lens, max_src_lens).to(texts.device)
        mel_masks = get_mask(mel_lens, max_mel_len).to(texts.device) if mel_lens is not None else None



        output = self.encoder(texts, src_masks)
        text_residual = output
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


        variation_residual = output
        output = self.decoder(output, mel_masks)
        output = self.mel_linear(output).masked_fill(mel_masks.unsqueeze(-1).expand(-1, -1, 80), 0)

        if mels is not None:
            postnet_output, log_det = self.postnet(
                output.transpose(1,2),
                mel_masks.unsqueeze(1),
                g = (text_residual + variation_residual).transpose(1,2)
            )
            postnet_output = postnet_output.masked_fill(mel_masks.unsqueeze(-1).expand(-1, -1, 80), 0)
        else:
            postnet_output = self.postnet.inference(mel_masks.unsqueeze(1), g = (text_residual + variation_residual))
            log_det = None
            postnet_output = postnet_output.masked_fill(mel_masks.unsqueeze(-1).expand(-1, -1, 80), 0)
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
            log_det,
        )

class VoiceAgent_Glow(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(VoiceAgent_Glow, self).__init__()
        self.model_config = model_config
        self.preprocess_config = preprocess_config

        self.encoder = Encoder(model_config)
        self.varianceadaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["Decoder"]["hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        )
        self.postnet = FlowPostNet(preprocess_config, model_config)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"],"speakers.json"
                ),
                "r"
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(n_speaker, model_config["Encoder"]["hidden"], padding_idx=n_speaker)

        self.voice_agent = VoiceAgent(model_config)


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
                c_emb=None,
                h_emb=None,
                h_speaker=None,
                h_lens=None,
                p_control = 1.0,
                e_control = 1.0,
                d_control = 1.0,
                ):
        src_masks = get_mask(src_lens, max_src_lens).to(texts.device)
        mel_masks = get_mask(mel_lens, max_mel_len).to(texts.device) if mel_lens is not None else None



        output = self.encoder(texts, src_masks)
        text_residual = output
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_lens, -1)

        dialogue_emb = self.voice_agent(c_emb, speakers, h_emb, h_speaker, h_lens)
        output = output + dialogue_emb.unsqueeze(1).expand(-1, max_src_lens, -1)


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


        variation_residual = output
        output = self.decoder(output, mel_masks)
        output = self.mel_linear(output).masked_fill(mel_masks.unsqueeze(-1).expand(-1, -1, 80), 0)

        if mels is not None:
            postnet_output, log_det = self.postnet(
                output.transpose(1,2),
                mel_masks.unsqueeze(1),
                g = (text_residual + variation_residual).transpose(1,2)
            )
            postnet_output = postnet_output.masked_fill(mel_masks.unsqueeze(-1).expand(-1, -1, 80), 0)
        else:
            postnet_output = self.postnet.inference(mel_masks.unsqueeze(1), g = (text_residual + variation_residual))
            log_det = None
            postnet_output = postnet_output.masked_fill(mel_masks.unsqueeze(-1).expand(-1, -1, 80), 0)

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
            log_det,
        )


class Dialogue_GCN_Glow(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(VoiceAgent_Glow, self).__init__()
        self.model_config = model_config
        self.preprocess_config = preprocess_config

        self.encoder = Encoder(model_config)
        self.varianceadaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["Decoder"]["hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        )
        self.postnet = FlowPostNet(preprocess_config, model_config)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"],"speakers.json"
                ),
                "r"
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(n_speaker, model_config["Encoder"]["hidden"], padding_idx=n_speaker)

        #dialogue Encoder part
        self.audio_encoder = Audio_encoder( preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                                            model_config["WiSeGTN"]["TA_Encoder"]["hidden"],
                                            model_config["WiSeGTN"]["TA_Encoder"]["Audio_Encoder"]["conv_kernel"],
                                            model_config["WiSeGTN"]["TA_Encoder"]["Audio_Encoder"]["n_layers"],
                                            model_config["WiSeGTN"]["TA_Encoder"]["Audio_Encoder"]["dropout"],
                                            "global"
                                            )
        self.text_linear = nn.Linear(model_config["WiSeGTN"]["TA_Encoder"]["text_dim"],
                                     model_config["WiSeGTN"]["TA_Encoder"]["hidden"])
        self.Dialogue_GCN = DialogueGCN(model_config)
        self.D_attention = MultiHeadAttention()


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
                c_text_emb=None,
                h_text_emb=None,
                h_audio_emb=None,
                h_speaker=None,
                h_lens=None,
                p_control = 1.0,
                e_control = 1.0,
                d_control = 1.0,
                ):
        src_masks = get_mask(src_lens, max_src_lens).to(texts.device)
        mel_masks = get_mask(mel_lens, max_mel_len).to(texts.device) if mel_lens is not None else None

        output = self.encoder(texts, src_masks)
        text_residual = output
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_lens, -1)


        #dialogue context maker

        h_dialogue_context = self.Dialogue_GCN()
        output, _ = self.D_attention(output, h_dialogue_context, h_dialogue_context)



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


        variation_residual = output
        output = self.decoder(output, mel_masks)
        output = self.mel_linear(output).masked_fill(mel_masks.unsqueeze(-1).expand(-1, -1, 80), 0)

        if mels is not None:
            postnet_output, log_det = self.postnet(
                output.transpose(1,2),
                mel_masks.unsqueeze(1),
                g = (text_residual + variation_residual).transpose(1,2)
            )
            postnet_output = postnet_output.masked_fill(mel_masks.unsqueeze(-1).expand(-1, -1, 80), 0)
        else:
            postnet_output = self.postnet.inference(mel_masks.unsqueeze(1), g = (text_residual + variation_residual))
            log_det = None
            postnet_output = postnet_output.masked_fill(mel_masks.unsqueeze(-1).expand(-1, -1, 80), 0)

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
            log_det,
        )
