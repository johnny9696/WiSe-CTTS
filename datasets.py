import json
import os
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from text import text_to_sequence

class FastSpeech2_dataset(Dataset):
    def __init__(self, text_path, model_config, preprocess_config):
        self.processed_path = preprocess_config["path"]["preprocessed_path"]
        self.n_speakers = model_config["speakers"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaner"]
        self.mel_channels = preprocess_config["mel"]["n_mel_channels"]
        self.data_path = self.load(os.path.join(preprocess_config["path"]["preprocessd_path"], text_path))

    def load(self, path):
        with open(path, 'r', encoding='UTF-8') as f:
                data = f.read().split("\n")
        data.pop()
        return data

    def get_text(self, text):
        phone = np.array(text_to_sequence(text, self.cleaners))
        text_length = np.shape(phone)[0]
        return phone, text_length

    def get_mel(self, basename):
        mel_path = os.path.join(self.processed_path,"mel/mel-{}.npy".format(basename))
        pitch_path = os.path.join(self.processed_path,"pitch/pitch-{}.npy".format(basename))
        energy_path = os.path.join(self.processed_path, "pitch/pitch-{}.npy".format(basename))
        duration_path = os.path.join(self.processed_path, "duration/duration-{}.npy".format(basename))

        mel = np.load(mel_path)
        pitch = np.load(pitch_path)
        energy = np.load(energy_path)
        duration_path = np.load(duration_path)
        mel_length = np.shape(mel)[0]
        return mel, pitch, energy, duration_path, mel_length


    def get_all(self, data):
        basename, speaker, phone, raw_text, token_dur = data.split("|")
        phone, text_length = self.get_text(phone)
        mel, pitch, energy, duration, mel_length = self.get_mel(basename)
        return basename, phone, text_length, speaker, mel, mel_length, pitch, energy, duration,

    def __len__(self):
        return len(self.data_path)
    def __getitem__(self, indx):
        return self.get_all(self.data_path[indx])

    def collate_fn(self, batch):
        """
        :param batch: [basename, phone, text_length, speaker, mel, mel_length, pitch, energy, duration]
        :return:
        """
        batch_size = len(batch)
        basename = [batch[i][0] for i in range(batch_size)]
        text_length = [batch[i][2] for i in range(batch_size)]
        max_text_length = max(text_length)
        mel_length = [batch[i][4] for i in range(batch_size)]
        max_mel_length = max(mel_length)
        speaker = torch.tensor([batch[i][3] for i in range(batch_size)], dtype = torch.int)

        text_pad = torch.zeros((batch_size, max_text_length), dtype = torch.float)
        mel_pad = torch.zeros((batch_size, max_mel_length, self.mel_channels), dtype = torch.float)
        pitch_pad = torch.zeros((batch_size, max_text_length), dtype = torch.float)
        energy_pad = torch.zeros((batch_size, max_text_length), dtype = torch.float)
        duration_pad = torch.zeros((batch_size, max_text_length), dtype = torch.int)

        for i in range(batch_size):
            text_pad[i, :text_length[i]] = torch.tensor(batch[i][1], dtype = torch.int)
            mel_pad[i, :mel_length[i]] = torch.tensor(batch[i][4], dtype = torch.float)
            pitch_pad[i, : text_length[i]] = torch.tensor(batch[i][6], dtype = torch.float)
            energy_pad[i, : text_length[i]] = torch.tensor(batch[i][7], dtype = torch.float)
            duration_pad[i,:text_length[i]] = torch.tensor(batch[i][8], dtype = torch.float)

        return (basename, text_pad, text_length, max_text_length, speaker, mel_pad, mel_length, max_mel_length, pitch_pad, energy_pad, duration_pad)


class VoiceAgent_dataset(Dataset):
    def __init__(self, text_path, model_config, preprocess_config):
        self.processed_path = preprocess_config["path"]["preprocessed_path"]
        self.n_speakers = model_config["speakers"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaner"]
        self.mel_channels = preprocess_config["mel"]["n_mel_channels"]
        self.data_path = self.load(os.path.join(preprocess_config["path"]["preprocessed_path"], text_path))
        self.history_length = model_config["VoiceAgent"]["history_length"]

        basename_path = os.path.join(preprocess_config["path"]["preprocessed_path"], "basename.txt")
        with open(basename_path, 'r', encoding='UTF-8') as f:
            self.basename_list = f.read().split("\n")
            self.basename_list.pop()

    def load(self, path):
        with open(path, 'r', encoding='UTF-8') as f:
            data = f.read().split("\n")
        data.pop()
        return data

    def get_text(self, text):
        phone = np.array(text_to_sequence(text, self.cleaners))
        text_length = np.shape(phone)[0]
        return phone, text_length

    def get_mel(self, basename):
        mel_path = os.path.join(self.processed_path, "mel/mel-{}.npy".format(basename))
        pitch_path = os.path.join(self.processed_path, "pitch/pitch-{}.npy".format(basename))
        energy_path = os.path.join(self.processed_path, "pitch/pitch-{}.npy".format(basename))
        duration_path = os.path.join(self.processed_path, "duration/duration-{}.npy".format(basename))
        sbert_path = os.path.join(self.processed_path,"sbert/sbert-{}.npy".format(basename))


        mel = np.load(mel_path)
        pitch = np.load(pitch_path)
        energy = np.load(energy_path)
        duration_path = np.load(duration_path)
        mel_length = np.shape(mel)[0]
        return mel, pitch, energy, duration_path, mel_length, sbert_path

    def load_historical(self, basename):
        """turn, speaker, dialogue"""
        c_turn, c_speaker, d_num = basename.split("_")
        history_basename = []
        if int(c_turn) - self.history_length > 0:
            s_turn = int(c_turn) - self.history_length
        else:
            s_turn = 0
        for i in range(s_turn, int(c_turn)):
            for j in range(self.n_speakers):
                h_basename = "_".join([str(i), str(j), d_num])
                if history_basename in self.basename_list:
                    history_basename.append(h_basename)

        sbert_list = []
        for h_basename in history_basename:
            sbert_path = os.path.join(self.processed_path, "sbert/sbert-{}.npy".format(h_basename))
            t_sbert = np.load(sbert_path)
            sbert_list.append(t_sbert)
        return sbert_list, len(history_basename)

    def get_all(self, data):
        basename, speaker, phone, raw_text, token_dur = data.split("|")
        phone, text_length = self.get_text(phone)
        mel, pitch, energy, duration, mel_length, sbert_path = self.get_mel(basename)
        h_sbert, h_length = self.load_historical(basename)
        return basename, phone, text_length, speaker, mel, mel_length, pitch, energy, duration,

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, indx):
        return self.get_all(self.data_path[indx])

    def collate_fn(self, batch):
        """
        :param batch: [basename, phone, text_length, speaker, mel, mel_length, pitch, energy, duration]
        :return:
        """
        batch_size = len(batch)
        basename = [batch[i][0] for i in range(batch_size)]
        text_length = [batch[i][2] for i in range(batch_size)]
        max_text_length = max(text_length)
        mel_length = [batch[i][4] for i in range(batch_size)]
        max_mel_length = max(mel_length)
        speaker = torch.tensor([batch[i][3] for i in range(batch_size)], dtype=torch.int)

        text_pad = torch.zeros((batch_size, max_text_length), dtype=torch.float)
        mel_pad = torch.zeros((batch_size, max_mel_length, self.mel_channels), dtype=torch.float)
        pitch_pad = torch.zeros((batch_size, max_text_length), dtype=torch.float)
        energy_pad = torch.zeros((batch_size, max_text_length), dtype=torch.float)
        duration_pad = torch.zeros((batch_size, max_text_length), dtype=torch.int)

        for i in range(batch_size):
            text_pad[i, :text_length[i]] = torch.tensor(batch[i][1], dtype=torch.int)
            mel_pad[i, :mel_length[i]] = torch.tensor(batch[i][4], dtype=torch.float)
            pitch_pad[i, : text_length[i]] = torch.tensor(batch[i][6], dtype=torch.float)
            energy_pad[i, : text_length[i]] = torch.tensor(batch[i][7], dtype=torch.float)
            duration_pad[i, :text_length[i]] = torch.tensor(batch[i][8], dtype=torch.float)

        return (
        basename, text_pad, text_length, max_text_length,
        speaker, mel_pad, mel_length, max_mel_length, pitch_pad,
        energy_pad, duration_pad)



#audio loader 추가 필요
class DialogueGCN(Dataset):
    def __init__(self, text_path, model_config, preprocess_config):
        self.processed_path = preprocess_config["path"]["preprocessed_path"]
        self.n_speakers = model_config["speakers"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaner"]
        self.mel_channels = preprocess_config["mel"]["n_mel_channels"]
        self.data_path = self.load(os.path.join(preprocess_config["path"]["preprocessed_path"], text_path))
        self.history_length = model_config["VoiceAgent"]["history_length"]

        basename_path = os.path.join(preprocess_config["path"]["preprocessed_path"], "basename.txt")
        with open(basename_path, 'r', encoding='UTF-8') as f:
            self.basename_list = f.read().split("\n")
            self.basename_list.pop()

    def load(self, path):
        with open(path, 'r', encoding='UTF-8') as f:
            data = f.read().split("\n")
        data.pop()
        return data

    def get_text(self, text):
        phone = np.array(text_to_sequence(text, self.cleaners))
        text_length = np.shape(phone)[0]
        return phone, text_length

    def get_mel(self, basename):
        mel_path = os.path.join(self.processed_path, "mel/mel-{}.npy".format(basename))
        pitch_path = os.path.join(self.processed_path, "pitch/pitch-{}.npy".format(basename))
        energy_path = os.path.join(self.processed_path, "pitch/pitch-{}.npy".format(basename))
        duration_path = os.path.join(self.processed_path, "duration/duration-{}.npy".format(basename))
        sbert_path = os.path.join(self.processed_path,"sbert/sbert-{}.npy".format(basename))


        mel = np.load(mel_path)
        pitch = np.load(pitch_path)
        energy = np.load(energy_path)
        duration_path = np.load(duration_path)
        mel_length = np.shape(mel)[0]
        return mel, pitch, energy, duration_path, mel_length, sbert_path

    def load_historical(self, basename):
        """turn, speaker, dialogue"""
        c_turn, c_speaker, d_num = basename.split("_")
        history_basename = []
        if int(c_turn) - self.history_length > 0:
            s_turn = int(c_turn) - self.history_length
        else:
            s_turn = 0
        for i in range(s_turn, int(c_turn)):
            for j in range(self.n_speakers):
                h_basename = "_".join([str(i), str(j), d_num])
                if history_basename in self.basename_list:
                    history_basename.append(h_basename)

        sbert_list = []
        for h_basename in history_basename:
            sbert_path = os.path.join(self.processed_path, "sbert/sbert-{}.npy".format(h_basename))
            t_sbert = np.load(sbert_path)
            sbert_list.append(t_sbert)
        return sbert_list, len(history_basename)

    def get_all(self, data):
        basename, speaker, phone, raw_text, token_dur = data.split("|")
        phone, text_length = self.get_text(phone)
        mel, pitch, energy, duration, mel_length, sbert_path = self.get_mel(basename)
        h_sbert, h_length = self.load_historical(basename)
        return basename, phone, text_length, speaker, mel, mel_length, pitch, energy, duration,

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, indx):
        return self.get_all(self.data_path[indx])

    def collate_fn(self, batch):
        """
        :param batch: [basename, phone, text_length, speaker, mel, mel_length, pitch, energy, duration]
        :return:
        """
        batch_size = len(batch)
        basename = [batch[i][0] for i in range(batch_size)]
        text_length = [batch[i][2] for i in range(batch_size)]
        max_text_length = max(text_length)
        mel_length = [batch[i][4] for i in range(batch_size)]
        max_mel_length = max(mel_length)
        speaker = torch.tensor([batch[i][3] for i in range(batch_size)], dtype=torch.int)

        text_pad = torch.zeros((batch_size, max_text_length), dtype=torch.float)
        mel_pad = torch.zeros((batch_size, max_mel_length, self.mel_channels), dtype=torch.float)
        pitch_pad = torch.zeros((batch_size, max_text_length), dtype=torch.float)
        energy_pad = torch.zeros((batch_size, max_text_length), dtype=torch.float)
        duration_pad = torch.zeros((batch_size, max_text_length), dtype=torch.int)

        for i in range(batch_size):
            text_pad[i, :text_length[i]] = torch.tensor(batch[i][1], dtype=torch.int)
            mel_pad[i, :mel_length[i]] = torch.tensor(batch[i][4], dtype=torch.float)
            pitch_pad[i, : text_length[i]] = torch.tensor(batch[i][6], dtype=torch.float)
            energy_pad[i, : text_length[i]] = torch.tensor(batch[i][7], dtype=torch.float)
            duration_pad[i, :text_length[i]] = torch.tensor(batch[i][8], dtype=torch.float)

        return (
        basename, text_pad, text_length, max_text_length, speaker, mel_pad, mel_length, max_mel_length, pitch_pad,
        energy_pad, duration_pad)