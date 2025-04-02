import os
import json
import yaml


import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

def to_device(model_name, data, device):
    if model_name == "FastSpeech2" or model_name == "FastSpeech2_Glow":
        (basename, text, t_length, max_text_length,
         speaker, mel, mel_length, max_mel_length,
         pitch, energy, duration) = data

        text = text.to(device)
        speaker = speaker.to(device)
        mel = mel.to(device)
        pitch = pitch.to(device)
        energy = energy.to(device)
        duration = duration.to(device)

        return (basename, text, t_length, max_text_length,
                speaker, mel, mel_length, max_mel_length,
                pitch, energy, duration)

    elif model_name == "VoiceAgent_Glow":
        (basename, text, text_length, max_text_length,
        speaker, mel, mel_length, max_mel_length, pitch,
        energy, duration, c_emb, h_emb, h_speaker, h_length) = data

        text = text.to(device)
        speaker = speaker.to(device)
        mel = mel.to(device)
        pitch = pitch.to(device)
        energy = energy.to(device)
        duration = duration.to(device)
        c_emb = c_emb.to(device)
        h_emb = h_emb.to(device)
        h_speaker = h_speaker.to(device)

        return (basename, text, text_length, max_text_length,
                speaker, mel, mel_length, max_mel_length,
                pitch, energy, duration, c_emb, h_emb, h_speaker, h_length)

    else:
        raise("There is no named {}".format(model_name))

def to_device_eval(model_name, data, device):
    if model_name == "FastSpeech2" or model_name == "FastSpeech2_Glow":
        (basename, text, t_length, max_text_length,
         speaker, mel, mel_length, max_mel_length,
         pitch, energy, duration) = data

        text = text.to(device)
        t_length = t_length
        speaker = speaker.to(device)
        mel = None
        mel_length = None
        max_mel_length = None
        pitch = None
        energy = None
        duration = None

        return (basename, text, t_length, max_text_length,
                speaker, mel, mel_length, max_mel_length,
                pitch, energy, duration)

    elif model_name == "VoiceAgent_Glow":
        (basename, text, text_length, max_text_length,
        speaker, mel, mel_length, max_mel_length, pitch,
        energy, duration, c_emb, h_emb, h_speaker, h_length) = data

        text = text.to(device)
        speaker = speaker.to(device)
        h_speaker = h_speaker.to(device)
        c_emb = c_emb.to(device)
        h_emb = h_emb.to(device)
        mel = None
        mel_length = None
        max_mel_length = None
        pitch = None
        energy = None
        duration = None


        return (basename, text, text_length, max_text_length,
                speaker, mel, mel_length, max_mel_length,
                pitch, energy, duration, c_emb, h_emb, h_speaker, h_length)
    else:
        raise("There is no named {}".format(model_name))


def get_mask(length, max_length = None):
    if max_length == None:
        max_length = max(length)

    batch = len(length)
    mask = torch.ones((batch, max_length), dtype = torch.bool)
    for i in range(batch):
        mask[i][:length[i]] = torch.zeros(length[i])

    return mask

def pad(input_ele, mel_max_length = None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(batch, (0, max_len-batch.size(0)),"constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(batch,(0, 0, 0,max_len-batch.size(0)),"constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded



def load_loss(model_name, train_config, preprocess_config, model_config):
    if model_name == "FastSpeech2":
        from Model.loss import FastSpeech2Loss
        loss_func = FastSpeech2Loss(preprocess_config, model_config)
    elif model_name == "FastSpeech2_Glow":
        from Model.loss import FastSpeech2_GlowLoss
        loss_func = FastSpeech2_GlowLoss(preprocess_config, model_config)
    elif model_name == "VoiceAgent_Glow" :
        from Model.loss import FastSpeech2_GlowLoss
        loss_func = FastSpeech2_GlowLoss(preprocess_config, model_config)
    elif model_name == "DialogueGCN":
        from Model.loss import DialogueGCNLoss
        loss_func = DialogueGCNLoss(preprocess_config, model_config)
    else:
        raise("There is no named model {}".format(model_name))

    return loss_func


def log_writer(writer, type, tag, data, steps):
    if type == "scalar":
        writer.add_scalar(tag + "/total_loss", data[0], steps)
        writer.add_scalar(tag + "/mel_loss", data[1], steps)
        writer.add_scalar(tag + "/mel_postnet_loss", data[2], steps)
        writer.add_scalar(tag + "/pitch_loss", data[3], steps)
        writer.add_scalar(tag + "/energy_loss", data[4], steps)
        writer.add_scalar(tag + "/duration_loss", data[5], steps)
    elif type == "image":
        writer.add_figure(tag,
                          data,
                          global_step=steps)
    elif type == "audio":
        writer.add_audio(tag,
                         data/max(np.abs(data[0])),
                         sample_rate = 22050,
                         global_step=steps)
    else:
        raise("wrong type. There is not type named {}".format(type))


