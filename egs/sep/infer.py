import sys
import os
import torch
import torchaudio
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from asteroid.models import BaseModel, DPRNNTasNet, ConvTasNet
from sepdiarize.utils.css_with_lookahead import css_lookahead
from sepdiarize.utils.dsp import check_clipping

class InferSep():
    def __init__(self, ckpt_path, css_conf, use_gpu=True):
        self.use_gpu = use_gpu
        
        self.cfg = OmegaConf.load(os.path.join(Path(ckpt_path).parent.parent.parent,
                                          ".hydra", "config.yaml"))
        self.cfg.css = css_conf

        # Load model checkpoint
        model_ckpt = torch.load(ckpt_path)
        
        # Load separator
        if self.cfg.training.net == 'convtasnet':
            if self.cfg.training.causal:
                self.separator = ConvTasNet(2, norm_type='cLN', causal=True)
            else:
                self.separator = ConvTasNet(2)
        elif self.cfg.training.net == 'dprnn':
            if self.cfg.training.causal:
                self.separator = DPRNNTasNet(2, norm_type='cLN', bidirectional=False)
            else:
                self.separator = DPRNNTasNet(2)
        else:
            raise NotImplementedError

        self.separator.eval()
        if self.use_gpu:
            self.separator = self.separator.cuda()
        new_state = {".".join(k.split(".")[1:]): v for k, v in model_ckpt["state_dict"].items() if k.startswith("separator")}
        self.separator.load_state_dict(new_state)

    
    def forward(self, file):
        mixture, fs = torchaudio.load(file)
        assert mixture.shape[0] == 1
        assert fs == self.cfg.data.samplerate
        src, samples = mixture.shape
        mixture = mixture.unsqueeze(0)
        
        if self.use_gpu:
            mixture = mixture.cuda()
        
        # Separation
        with torch.inference_mode():
            if self.cfg.css.window_size is None:
                separated = self.separator(mixture)
            else:
                separated = css_lookahead(mixture, self.separator, 2, self.cfg.css.window_size,
                                          self.cfg.css.stride, fs=fs, window_type=self.cfg.css.window_type,
                                          lookbehind=self.cfg.css.lookbehind)
        # Adjust out of range values
        separated, _ = check_clipping(separated)

        
        if self.use_gpu:
            separated = separated.cpu()
        separated = separated[0].numpy()
        
        return separated