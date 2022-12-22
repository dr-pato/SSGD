import os
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from sepdiarize.vad.tcn import TCN
from sepdiarize.vad.energy import energy_vad
from sepdiarize.utils.encoder import ManyHotEncoder
from sepdiarize.utils.diarization import delete_shorter
from sepdiarize.utils.dsp import check_clipping
import torchaudio
import scipy


class InferTCNVAD():
    def __init__(self, ckpt_path, median_filter=100, delete_shorter_than=0.25, threshold=0.5, use_gpu=True):
        self.median_filter = median_filter
        self.delete_shorter_than = delete_shorter_than
        self.threshold = threshold
        self.use_gpu = use_gpu
        
        self.cfg = OmegaConf.load(os.path.join(Path(ckpt_path).parent.parent.parent,
                                          ".hydra", "config.yaml"))

        # Load model checkpoint
        model_ckpt = torch.load(ckpt_path)

        # Load VAD
        self.vad = TCN(self.cfg.feats, **self.cfg.vad)
        self.vad.eval()
        if self.use_gpu:
            self.vad = self.vad.cuda()

        new_state = {".".join(k.split(".")[1:]): v for k, v in model_ckpt["state_dict"].items() if k.startswith("vad")}
        del new_state["pos_weight"]
        self.vad.load_state_dict(new_state)
    
    def forward(self, file):
        audio, fs = torchaudio.load(file)
        assert audio.shape[0] == 2
        assert fs == self.cfg.data.samplerate
        src, samples = audio.shape
        audio = audio.unsqueeze(0)
        
        if self.use_gpu:
            audio = audio.cuda()

        encoder = ManyHotEncoder(["0", "1"], samples / fs, self.cfg.feats.win_length,
                                 self.cfg.feats.hop_length, net_pooling=1, fs=fs)

        with torch.inference_mode():
            preds = self.vad(audio)[0]
        vad_prob = torch.sigmoid(preds)
        sad = vad_prob.cpu().numpy() > self.threshold
        sad = scipy.ndimage.filters.median_filter(sad, (1, self.median_filter))
        decoded = encoder.decode_strong(sad.T)
        decoded = delete_shorter(decoded, self.delete_shorter_than)
        decoded.sort(key=lambda x: x[1])
        
        return decoded


class InferEnergyVAD():
    def __init__(self, median_filter=50, delete_shorter_than=0, threshold=0.9):
        self.median_filter = median_filter
        self.delete_shorter_than = delete_shorter_than
        self.threshold = threshold

    def forward(self, file):
        fs, audio = scipy.io.wavfile.read(file)
        audio = audio.T
        assert audio.shape[0] == 2
        src, samples = audio.shape
        
        # Run energy-based VAD
        decoded_1 = energy_vad(audio[0], self.threshold, self.median_filter, fs)
        decoded_2 = energy_vad(audio[1], self.threshold, self.median_filter, fs)
        decoded_1 = [['0', x[0], x[1]] for x in decoded_1]
        decoded_2 = [['1', x[0], x[1]] for x in decoded_2]
        decoded = decoded_1 + decoded_2
        decoded = delete_shorter(decoded, self.delete_shorter_than)
        decoded.sort(key=lambda x: x[1])
        
        return decoded