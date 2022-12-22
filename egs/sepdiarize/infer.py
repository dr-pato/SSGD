import sys
import os
from pathlib import Path
import torch
import tqdm
import torchaudio
import numpy as np
import scipy
from omegaconf import DictConfig, OmegaConf
from asteroid.models import BaseModel, DPRNNTasNet, ConvTasNet
from asteroid.dsp.overlap_add import LambdaOverlapAdd
from sepdiarize.vad.tcn import TCN
from sepdiarize.vad.energy import energy_vad
from sepdiarize.utils.encoder import ManyHotEncoder
from sepdiarize.utils.diarization import delete_shorter
from sepdiarize.utils.dsp import check_clipping, remove_leakage
from sepdiarize.utils.css_with_lookahead import css_lookahead


class InferSepTCNVADiarize():
    def __init__(self, sep_ckpt_path, vad_ckpt_path, css_conf, median_filter=100, delete_shorter_than=0.25,
                 threshold=0.5, rl_seg_length=None, rl_threshold=None, use_gpu=True):
        self.median_filter = median_filter
        self.delete_shorter_than = delete_shorter_than
        self.threshold = threshold
        self.rl_seg_length = rl_seg_length
        self.rl_threshold = rl_threshold
        self.use_gpu = use_gpu
        

        # Separator conf
        self.sep_conf = OmegaConf.load(os.path.join(Path(sep_ckpt_path).parent.parent.parent,
                                                  '.hydra', 'config.yaml'))
        # VAD conf
        self.vad_conf = OmegaConf.load(os.path.join(Path(vad_ckpt_path).parent.parent.parent,
                                                   '.hydra', 'config.yaml'))
        
        self.sep_conf.css = css_conf
        
        # Load model checkpoints
        sep_ckpt = torch.load(sep_ckpt_path)
        vad_ckpt = torch.load(vad_ckpt_path)
        
        # Load separator
        if self.sep_conf.training.net == 'convtasnet':
            if self.sep_conf.training.causal:
                self.separator = ConvTasNet(2, norm_type='cLN', causal=True)
            else:
                self.separator = ConvTasNet(2)
        elif self.sep_conf.training.net == 'dprnn':
            if self.sep_conf.training.causal:
                self.separator = DPRNNTasNet(2, norm_type='cLN', bidirectional=False)
            else:
                self.separator = DPRNNTasNet(2)
        else:
            raise NotImplementedError
        
        self.separator.eval()
        if self.use_gpu:
            self.separator = self.separator.cuda()
        new_state = {".".join(k.split(".")[1:]): v for k, v in sep_ckpt["state_dict"].items() if k.startswith("separator")}
        self.separator.load_state_dict(new_state)

        # Load VAD
        self.vad = TCN(self.vad_conf.feats, **self.vad_conf.vad)
        self.vad.eval()
        if self.use_gpu:
            self.vad = self.vad.cuda()
        new_state = {".".join(k.split(".")[1:]): v for k, v in vad_ckpt["state_dict"].items() if k.startswith("vad")}
        del new_state["pos_weight"]
        self.vad.load_state_dict(new_state)
    
    def forward(self, file):
        mixture, fs = torchaudio.load(file)
        assert mixture.shape[0] == 1
        assert fs == self.sep_conf.data.samplerate
        src, samples = mixture.shape
        mixture = mixture.unsqueeze(0)
        
        if self.use_gpu:
            mixture = mixture.cuda()
        
        # Separation
        with torch.inference_mode():
            if self.sep_conf.css.window_size is None:
                separated = self.separator(mixture)
            else:
                separated = css_lookahead(mixture, self.separator, 2, self.sep_conf.css.window_size,
                                          self.sep_conf.css.stride, fs=fs, window_type=self.sep_conf.css.window_type,
                                          lookbehind=self.sep_conf.css.lookbehind)
        # Adjust out of range values
        separated, _ = check_clipping(separated)

        bsz, src, _ = separated.shape
        assert separated.shape[0] == 1

        # Leakage removal
        if self.rl_seg_length:
            if self.use_gpu:
                mixture, separated = mixture.cpu(), separated.cpu()
            mixture, separated = mixture.numpy()[0], separated.numpy()[0]
            seg_length = int(self.rl_seg_length * self.sep_conf.data.samplerate)
            separated, _ = remove_leakage(separated, mixture, seg_length, self.rl_threshold)
            separated = torch.Tensor(separated).unsqueeze(0)
            if self.use_gpu:
               separated = separated.cuda()

        # VAD
        encoder = ManyHotEncoder(["0", "1"], samples / fs, self.vad_conf.feats.win_length,
                                 self.vad_conf.feats.hop_length, net_pooling=1, fs=fs)
        
        with torch.inference_mode():
            preds = self.vad(separated)[0]
        vad_prob = torch.sigmoid(preds)
        sad = vad_prob.cpu().numpy() > self.threshold
        sad = scipy.ndimage.filters.median_filter(sad, (1, self.median_filter))
        decoded = encoder.decode_strong(sad.T)
        decoded = delete_shorter(decoded, self.delete_shorter_than)
        decoded.sort(key=lambda x: x[1])
        
        if self.use_gpu:
            separated = separated.cpu()
        separated = separated[0].numpy()
        
        return separated, decoded


class InferSepEnergyVADiarize():
    def __init__(self, sep_ckpt_path, css_conf, median_filter=100, delete_shorter_than=0.25, threshold=0.5, rl_seg_length=None, rl_threshold=None, use_gpu=True):
        self.median_filter = median_filter
        self.delete_shorter_than = delete_shorter_than
        self.threshold = threshold
        self.rl_seg_length = rl_seg_length
        self.rl_threshold = rl_threshold
        self.use_gpu = use_gpu
        
        self.sep_conf = OmegaConf.load(os.path.join(Path(sep_ckpt_path).parent.parent.parent,
                                          ".hydra", "config.yaml"))
        self.css = css_conf

        # Load SS model checkpoint
        sep_ckpt = torch.load(sep_ckpt_path)
        
        # Load separator
        if self.sep_conf.training.net == 'convtasnet':
            if self.sep_conf.training.causal:
                self.separator = ConvTasNet(2, norm_type='cLN', causal=True)
            else:
                self.separator = ConvTasNet(2)
        elif self.sep_conf.training.net == 'dprnn':
            if self.sep_conf.training.causal:
                self.separator = DPRNNTasNet(2, norm_type='cLN', bidirectional=False)
            else:
                self.separator = DPRNNTasNet(2)
        else:
            raise NotImplementedError

        self.separator.eval()
        if self.use_gpu:
            self.separator = self.separator.cuda()
        new_state = {".".join(k.split(".")[1:]): v for k, v in sep_ckpt["state_dict"].items() if k.startswith("separator")}
        self.separator.load_state_dict(new_state)
        
    def forward(self, file):
        mixture, fs = torchaudio.load(file)
        assert mixture.shape[0] == 1
        assert fs == self.sep_conf.data.samplerate
        src, samples = mixture.shape
        mixture = mixture.unsqueeze(0)
        
        if self.use_gpu:
            mixture = mixture.cuda()
        
        # Separation
        with torch.inference_mode():
            if self.css.window_size is None:
                separated = self.separator(mixture)
            else:
                separated = css_lookahead(mixture, self.separator, 2, self.sep_conf.css.window_size,
                                          self.sep_conf.css.stride, fs=fs, window_type=self.sep_conf.css.window_type,
                                          lookbehind=self.sep_conf.css.lookbehind)
        # Adjust out of range values
        separated, _ = check_clipping(separated)

        bsz, src, _ = separated.shape
        assert separated.shape[0] == 1
        if self.use_gpu:
            mixture, separated = mixture.cpu(), separated.cpu()
        mixture, separated = mixture.numpy()[0], separated.numpy()[0]

        # Leakage removal
        if self.rl_seg_length:
            seg_length = int(self.rl_seg_length * self.sep_conf.data.samplerate)
            separated, _ = remove_leakage(separated, mixture, seg_length, self.rl_threshold)
            
        # Run energy-based VAD
        decoded_1 = energy_vad(separated[0], self.threshold, self.median_filter, fs)
        decoded_2 = energy_vad(separated[1], self.threshold, self.median_filter, fs)
        decoded_1 = [['0', x[0], x[1]] for x in decoded_1]
        decoded_2 = [['1', x[0], x[1]] for x in decoded_2]
        decoded = decoded_1 + decoded_2
        decoded = delete_shorter(decoded, self.delete_shorter_than)
        
        return separated, decoded