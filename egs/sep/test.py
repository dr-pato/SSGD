import sys
import os
import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from egs.sep.infer import InferSep
from glob import glob
import numpy as np
import hydra
import soundfile as sf
from sepdiarize.utils.dsp import compute_sisdr


@hydra.main(config_path="confs", config_name="test")
def test(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    audio_dir = os.path.join(cfg.data.root_test, cfg.data.audio_dir)
    
    # Model loading
    sep = InferSep(cfg.ckpt, cfg.css, use_gpu=cfg.opt.use_gpu)
    
    # Create output dir if needed
    if cfg.data.out_dir:
        os.makedirs(os.path.join(cfg.data.out_dir, 'sep'), exist_ok=True)

    print('Inference:')
    ss_metrics_list = []
    wav_list = glob(os.path.join(audio_dir, '*.wav'))
    for wav_fname in tqdm(wav_list):
        id = os.path.basename(wav_fname).replace('.wav', '')
        
        separated = sep.forward(wav_fname)

        mix, _ = sf.read(wav_fname, dtype='float32')
        src_fname = os.path.join(cfg.data.root_test, cfg.data.audio_oracle_dir, id + '.wav')
        src, _ = sf.read(src_fname, dtype='float32')
        src = np.transpose(src)
        ss_metrics_dict = compute_sisdr(mix, src, separated, compute_permutation=True)
        ss_metrics_list.append(ss_metrics_dict['si_sdr'] - ss_metrics_dict['input_si_sdr'])

        if cfg.data.out_dir:
            sep_out_path = os.path.join(cfg.data.out_dir, 'sep', id + '.wav')
            sf.write(sep_out_path, np.transpose(separated), sep.cfg.data.samplerate)
        
    print('SI-SDRi: {:.2f}'.format(np.mean(ss_metrics_list)))


if __name__ == '__main__':
    test()
