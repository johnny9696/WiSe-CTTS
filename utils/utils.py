import os
import json
import yaml
from matplotlib import pyplot as plt

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Model.models import FastSpeech2,FastSpeech2_Glow, VoiceAgent_Glow
from model import get_vocoder, vocoder_infer

def to_device(model_name, data, device):
    if model_name == "FastSpeech2" or model_name == "FastSpeech2_Glow":
        (basename, text, t_length, max_text_length,
         speaker, mel, mel_length, max_mel_length,
         pitch, energy, duration) = data

        text = text.to(device)
        t_length = t_length.to(device)
        max_text_length = max_text_length.to(device)
        speaker = speaker.to(device)
        mel = mel.to(device)
        mel_length = mel_length.to(device)
        max_mel_length = max_mel_length.to(device)
        pitch = pitch.to(device)
        energy = energy.to(device)
        duration = duration.to(device)

        return (basename, text, t_length, max_text_length,
                speaker, mel, mel_length, max_mel_length,
                pitch, energy, duration)
    else:
        raise("There is no named {}".format(model_name))

def to_device_eval(model_name, data, device):
    if model_name == "FastSpeech2" or model_name == "FastSpeech2_Glow":
        (basename, text, t_length, max_text_length,
         speaker, mel, mel_length, max_mel_length,
         pitch, energy, duration) = data

        text = text.to(device)
        t_length = t_length.to(device)
        max_text_length = max_text_length.to(device)
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
    else:
        raise("There is no named {}".format(model_name))


def get_mask(length, max_length = None):
    if max_length == None:
        max_length = max(length)

    batch = len(length)
    mask = torch.ones((batch, max_length))
    for i in range(batch):
        mask[i][:length[i]] = torch.zeros(length[i])

    return mask

def pad(input_ele, mel_max_length = None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0)] for i in range(len(input_ele)))

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(batch, (0, max_len-batch.size(0)),"constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(batch,(0,0,0,max_len-batch.size(0)),"constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

def load_models(model_name, preprocess_config, model_config, restore_step = None):

    if model_name == "FastSpeech2":
        model = FastSpeech2(preprocess_config, model_config)
    elif model_name == "FastSpeech2_Glow":
        model = FastSpeech2_Glow(preprocess_config, model_config)
    elif model_name == "VoiceAgent_Glow":
        model = VoiceAgent_Glow(preprocess_config, model_config)

    if restore_step is not None:
        ckpt_path = os.path.join(preprocess_config["path"]["ckpt_path"], "{}.pth.tar".format(restore_step))
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    return model

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

def load_dataset(model_name, text_path, train_config, model_config, preprocess_config, shuffle = False):
    if model_name == "FastSpeech2" or model_name == "FastSpeech2_Glow":
        from datasets import FastSpeech2_dataset
        dataset_loader = FastSpeech2_dataset(text_path, model_config, preprocess_config)
    elif model_name == "VoiceAgent_Glow":
        from datasets import VoiceAgent_dataset
        dataset_loader = VoiceAgent_dataset(text_path, model_config, preprocess_config)
    elif model_name == "DialogueGCN":
        from datasets import DialogueGCN_dataset
        dataset_loader = DialogueGCN_dataset(text_path, model_config, preprocess_config)

    else:
        raise("There is no named {}".format(model_name))
    datasets = DataLoader(
        dataset_loader,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=shuffle,
        collate_fn=dataset_loader.collate_fn,
    )
    return datasets


def log_writer(writer, type, tag, data, steps):
    if type == "scalar":
        writer.add_scalar(tag + "/Total", data[0], steps)
        writer.add_scalar(tag + "/Mel_loss", data[1], steps)
        writer.add_scalar(tag + "/Postnet_loss", data[2], steps)
        writer.add_scalar(tag + "/Pitch_loss", data[3], steps)
        writer.add_scalar(tag + "/Energy_loss", data[4], steps)
        writer.add_scalar(tag + "/Duration_loss", data[5], steps)
    elif type == "image":
        writer.add_figure(tag,
                          data,
                          global_step=steps)
    elif type == "audio":
        writer.add_audio(tag,
                         data/max(abs(data)),
                         sample_rate = 22060,
                         global_step=steps)
    else:
        raise("wrong type. There is not type named {}".format(type))


def mel_plot(data, title):
    fig, axes  = plt.subplots(1, 1, squeeze = False)

    axes[0][0].imshow(data, origin = "lower")
    axes[0][0].set_ylim(0, data.shape[0])
    axes[0][0].set_title(title, fontsize="medium")

    return fig

def synth_one_sample(mel, model_config, preprocess_config, device, length = None):
    vocoder = get_vocoder(model_config, device)
    wavs = vocoder_infer(mel, vocoder, model_config, preprocess_config, length)
    return wavs
