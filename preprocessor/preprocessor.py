import os
import random
import json
import string
import re

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio

# sentence transformer
from sentence_transformers import SentenceTransformer
# BERT Model
from transformers import AutoTokenizer, BertModel


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.val_size = config["preprocessing"]["val_size"]

        self.metadata = self.load_metadata(config)

        self.sentence_embedder = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
        self.text_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.BERT_model =  BertModel.from_pretrained("google-bert/bert-base-uncased")

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def load_metadata(self, config):
        metadata_path = os.path.join(config["path"]["corpus_path"],"metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "BPalign")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "bert")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "sbert")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        outliear = []
        normal = []

        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            wav_list = os.listdir(os.path.join(self.in_dir, speaker))
            inner_bar = tqdm(total=len(wav_list), desc="processing", position=0)
            inner_bar.update()
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                inner_bar.update(1)
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        splitted_name = basename.split("_")
                        d_num = splitted_name[-1]
                        if d_num not in outliear:
                            outliear.append(d_num)
                        continue
                    else:
                        info, pitch, energy, n = ret
                        splitted_name = basename.split("_")
                        d_num = splitted_name[-1]
                        normal.append(d_num)
                    out.append(info)

                    if len(pitch) > 0:
                        pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    if len(energy) > 0:
                        energy_scaler.partial_fit(energy.reshape((-1, 1)))

                    n_frames += n


        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )


        random.shuffle(normal)

        train_set= []
        validataion_set =[]
        validation_d_num = []
        for data in out:
            splitted_data = data.split("|")
            d_num = splitted_data[0].split("_")
            d_num = d_num[-1]
            if d_num not in outliear:
                if d_num not in validation_d_num and len(validation_d_num) < self.val_size:
                    validation_d_num.append(d_num)
                if len(validation_d_num) <= self.val_size and d_num in validation_d_num:
                    print(d_num)
                    validataion_set.append(data)
                else:
                    train_set.append(data)
            else:
                continue


        random.shuffle(train_set)
        random.shuffle(validataion_set)

        # Write metadata
        with open(os.path.join(self.out_dir, "outliear_dialogue.txt"),"w", encoding = "utf-8") as f:
            for m in outliear:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in train_set:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in validataion_set:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration_phone, start_p, end_p = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        word, duration_word, start_w, end_w = self.get_alignment(
            textgrid.get_tier_by_name("words")
        )

        name_list = basename.split("_")
        turn = name_list[0]
        speaker_id = name_list[1]
        dialogue_num = name_list[2][1:]
        raw_text = self.metadata[dialogue_num][turn]["text"]
        raw_text = raw_text.replace("’", " ")
        raw_text = raw_text.replace("…", " ")
        raw_text = raw_text.replace("—", " ")
        raw_text = re.sub(f"[{re.escape(string.punctuation)}]", " ", raw_text)

        token = self.text_tokenizer(raw_text, return_tensors="pt")
        token_T = self.text_tokenizer.tokenize(raw_text)
        outputs = self.BERT_model(**token)
        text_embedding = outputs.last_hidden_state
        text_emb = text_embedding.squeeze(0).detach().numpy()
        text_emb = text_emb[1:-1, :]

        if len(text_emb) != len(word):
            print(basename)
            print(token_T)
            print(word)
            return None

        # SBERT
        s_emb = self.sentence_embedder.encode([raw_text])
        s_emb = np.array(s_emb.squeeze(0))


        if sum(duration_phone) != sum(duration_word):
            return None
        else:
            bert_duration = self.align_BandP(duration_word, duration_phone)
            if len(duration_word) != len(bert_duration):
                return None
            if len(duration_phone) != sum(bert_duration):
                return None
        text = "{" + " ".join(phone) + "}"
        if start_p >= end_p:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start_p) : int(self.sampling_rate * end_p)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration_phone)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration_phone)]
        energy = energy[: sum(duration_phone)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration_phone):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration_phone)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration_phone):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration_phone)]


        splitted_name = basename.split("_")
        speaker = splitted_name[1]
        utter_id = basename[0]
        d_num = splitted_name[-1][1:]

        emotion = self.metadata[d_num][utter_id]["emotion"]
        act = self.metadata[d_num][utter_id]["act"]

        # Save files
        dur_filename = "duration-{}.npy".format( basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration_phone)

        bert_filename = "bert-{}.npy".format( basename)
        np.save(os.path.join(self.out_dir, "bert", bert_filename), text_emb)

        sbert_filename = "bert-{}.npy".format(basename)
        np.save(os.path.join(self.out_dir, "sbert", sbert_filename), s_emb)

        bert_p_align_filename = "BPalign-{}.npy".format(basename)
        np.save(os.path.join(self.out_dir, "BPalign", bert_p_align_filename), bert_duration)

        pitch_filename = "pitch-{}.npy".format(basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "energy-{}.npy".format(basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "mel-{}.npy".format(basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )
        return (
            "|".join([basename, speaker, text, raw_text, emotion, act]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def align_BandP(self, word_duration, phone_duration):
        BERT_duration = []
        start = 0
        for i in word_duration:
            s_length = 0
            cnt = 0
            for j in range(start, len(phone_duration)):
                cnt += 1
                if i > s_length + phone_duration[j]:
                    s_length += phone_duration[j]
                elif i == s_length + phone_duration[j]:
                    start = j + 1
                    BERT_duration.append(cnt)
                    break
                else:
                    if i - (s_length + phone_duration[j]) < 5:
                        start = j + 1
                        BERT_duration.append(cnt)
                    else:
                        start = j
                        BERT_duration.append(cnt - 1)
                    break
        return np.array(BERT_duration)

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
