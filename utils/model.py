import os
import yaml
import json

import torch
import numpy as np
from matplotlib import pyplot as plt

import hifigan
from torch.utils.data import DataLoader
from Model.models import FastSpeech2,FastSpeech2_Glow, VoiceAgent_Glow

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

def load_dataset(model_name, text_path, train_config, model_config, preprocess_config, shuffle = False):
    if model_name == "FastSpeech2" or model_name == "FastSpeech2_Glow":
        from dataset_loader import FastSpeech2_dataset
        dataset_loader = FastSpeech2_dataset(text_path, model_config, preprocess_config)
    elif model_name == "VoiceAgent_Glow":
        from dataset_loader import VoiceAgent_dataset
        dataset_loader = VoiceAgent_dataset(text_path, model_config, preprocess_config)
    elif model_name == "DialogueGCN":
        from dataset_loader import DialogueGCN_dataset
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

def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]
    for i in range(len(wavs)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs


def mel_plot(data, title, length = None):
    if length is not None:
        data = data[:,:length]

    fig, axes  = plt.subplots(1, 1, squeeze = False)

    axes[0][0].imshow(data, origin = "lower")
    axes[0][0].set_ylim(0, data.shape[0])
    axes[0][0].set_title(title, fontsize="medium")

    return fig

def synth_one_sample(mel, model_config, preprocess_config, device, length = None):
    vocoder = get_vocoder(model_config, device)
    wavs = vocoder_infer(mel.transpose(-1,-2), vocoder, model_config, preprocess_config, [length* preprocess_config["preprocessing"]["stft"]["hop_length"]])
    return wavs

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


