import os
import torch
from torch.utils.data import Dataset
import glob
import torchaudio
from pathlib import Path
import numpy as np
import pandas as pd
torchaudio.set_audio_backend("soundfile")


class WavDataset(Dataset):
    def __init__(self, root_folder, segment=None, training=False,
                 samplerate=8000, oversample=1):

        super(WavDataset, self).__init__()

        wav_files = glob.glob(os.path.join(root_folder, "**/*mix.wav"), recursive=True)

        self.segment = segment
        self.training = training
        self.samplerate = samplerate

        if not self.training:
            assert self.segment is None

        if self.segment:
            examples = []
            for w in wav_files:
                infos = torchaudio.info(w)
                length = infos.num_frames / infos.sample_rate
                if length >= self.segment:
                    examples.append(w)
            self.examples = examples
        else:
            self.examples = wav_files
        self.examples = self.examples * oversample

    def __len__(self):
        return len(self.examples)

    def _read(self, c_mix):

        offset = 0
        ex_len = torchaudio.info(c_mix).num_frames
        length = ex_len
        if self.segment is not None:
            tgt_len = int(self.segment * self.samplerate)
            if ex_len > tgt_len:
                offset = np.random.randint(0, ex_len - tgt_len)
                length = tgt_len
            elif ex_len == tgt_len:
                pass
            else:
                raise NotImplementedError

        s1, fs = torchaudio.load(os.path.join(Path(c_mix).parent, "source_1.wav"), frame_offset=offset,
                                        num_frames=length)
        s2, fs = torchaudio.load(os.path.join(Path(c_mix).parent, "source_2.wav"), frame_offset=offset,
                                 num_frames=length)

        sources = torch.cat((s1, s2), 0)
        mixture = sources.sum(0)

        return mixture, sources

    def __getitem__(self, item):

        c_mix = self.examples[item]

        mixture, sources = self._read(c_mix)
        return {"mixture": mixture,
                "sources": sources}


class AnnotatedDataset(Dataset):
    def __init__(self, root_folder, parsed_folder, encoder, segment=None, training=False,
                 samplerate=8000, oversample=1, has_oracle=True):
        super(AnnotatedDataset, self).__init__()

        wav_files = glob.glob(os.path.join(root_folder, "wav", "*.wav"))

        self.encoder = encoder
        self.parsed_folder = parsed_folder
        self.segment = segment
        self.training = training
        self.samplerate = samplerate
        self.has_oracle = has_oracle

        if not self.training:
            assert self.segment is None

        if self.segment:
            examples = []
            for ex in wav_files:
                infos = torchaudio.info(ex)
                length = infos.num_frames / infos.sample_rate
                if length >= self.segment:
                    examples.append(ex)
            self.examples = examples
        else:
            self.examples = wav_files
        
        self.examples = self.examples * oversample

    def __len__(self):
        return len(self.examples)

    def _read(self, c_mix):

        offset = 0
        ex_len = torchaudio.info(c_mix).num_frames
        length = ex_len
        if self.segment is not None:
            tgt_len = int(self.segment * self.samplerate)
            if ex_len > tgt_len:
                offset = np.random.randint(0, ex_len - tgt_len)
                length = tgt_len
            elif ex_len == tgt_len:
                pass
            else:
                raise NotImplementedError

        audio, fs = torchaudio.load(c_mix, frame_offset=offset,
                                        num_frames=length)
        if self.has_oracle:
            sources = audio
            mixture = audio.sum(0)
        else:
            sources = torch.zeros_like(audio).repeat(2, 1)
            mixture = audio[0]

        filename = Path(c_mix).stem
        offset = int(offset / self.encoder.frame_hop)
        length = int(length / self.encoder.frame_hop)

        vad_labels, fs = torchaudio.load(os.path.join(self.parsed_folder, filename + ".wav"), frame_offset=offset,
                                         num_frames=length)

        vad_labels = vad_labels[:2]

        return mixture, sources, vad_labels

    def __getitem__(self, item):

        c_mix = self.examples[item]
        filename = Path(c_mix).stem

        rttm_file = os.path.join(Path(c_mix).parent.parent, "rttm", filename + ".rttm")
        mixture, sources, vad_labels  = self._read(c_mix)
        return {"mixture": mixture,
                "sources": sources,
                "vad_labels": vad_labels,
                "rttm_file": rttm_file}