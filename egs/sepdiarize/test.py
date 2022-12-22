import sys
import os
from pathlib import Path
from glob import glob
import shutil
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import torch
import torchaudio
import numpy as np
import scipy
import hydra
import soundfile as sf
from sepdiarize.vad.tcn import TCN
from sepdiarize.utils.encoder import ManyHotEncoder
from sepdiarize.utils.diarization import write_rttm, delete_shorter
from sepdiarize.utils.der_eval import der_eval, der_eval_overall
from sepdiarize.utils.dsp import compute_sisdr
from infer import InferSepTCNVADiarize, InferSepEnergyVADiarize


@hydra.main(config_path="confs", config_name="test")
def test(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    audio_dir = os.path.join(cfg.data.root_test, cfg.data.audio_dir)
    ref_rttm_dir = os.path.join(cfg.data.root_test, cfg.data.ref_rttm_dir)
    
    # Model loading
    if cfg.vad.type == 'tcn':
        sepdiarize = InferSepTCNVADiarize(cfg.sep_ckpt, cfg.vad_ckpt, cfg.css, median_filter=cfg.vad.median_filter, delete_shorter_than=cfg.vad.del_factor,
                                          threshold=cfg.vad.threshold, rl_seg_length=cfg.leakage_removal.seg_length, rl_threshold=cfg.leakage_removal.threshold,
                                          use_gpu=cfg.opt.use_gpu)
    elif cfg.vad.type == 'energy':
        sepdiarize = InferSepEnergyVADiarize(cfg.sep_ckpt, cfg.css, median_filter=cfg.vad.median_filter, delete_shorter_than=cfg.vad.del_factor,
                                             threshold=cfg.vad.threshold, rl_seg_length=cfg.leakage_removal.seg_length, rl_threshold=cfg.leakage_removal.threshold,
                                             use_gpu=cfg.opt.use_gpu)
    else:
        raise NotImplementedError
    
    # Create temp files for evaluation
    tmp_dir = os.path.join('/tmp', str(os.getpid()) + '_test')
    os.makedirs(tmp_dir)
    
    # Create output dir if needed
    if cfg.data.out_dir:
        os.makedirs(os.path.join(cfg.data.out_dir, 'sep'), exist_ok=True)
        os.makedirs(os.path.join(cfg.data.out_dir, 'diar'), exist_ok=True)

    print('Inference:')
    rttm_ref_list = []
    rttm_hyp_list = []
    ss_metrics_list = []
    wav_list = glob(os.path.join(audio_dir, '*.wav'))
    for wav_fname in tqdm(wav_list):
        id = os.path.basename(wav_fname).replace('.wav', '')
        
        separated, diar_out = sepdiarize.forward(wav_fname)
        
        ref_path = os.path.join(ref_rttm_dir, id + '.rttm')
        hyp_path = os.path.join(tmp_dir, id + '_hyp.rttm')
        write_rttm(diar_out, id, hyp_path)
        rttm_ref_list.append(ref_path)
        rttm_hyp_list.append(hyp_path)

        if cfg.opt.eval_ss:
            mix, sr = sf.read(wav_fname, dtype='float32')
            src_fname = os.path.join(cfg.data.root_test, cfg.data.audio_oracle_dir, id + '.wav')
            src, _ = sf.read(src_fname, dtype='float32')
            src = np.transpose(src)
            ss_metrics_dict = compute_sisdr(mix, src, separated, compute_permutation=True)
            ss_metrics_list.append(ss_metrics_dict['si_sdr'] - ss_metrics_dict['input_si_sdr'])

        if cfg.data.out_dir:
            sep_out_path = os.path.join(cfg.data.out_dir, 'sep', id + '.wav')
            sf.write(sep_out_path, np.transpose(separated), sepdiarize.sep_conf.data.samplerate)
            diar_out_path = os.path.join(cfg.data.out_dir, 'diar', id + '.rttm')
            write_rttm(diar_out, id, diar_out_path)
        
    der_ignovr_collar250, miss_ignovr_collar250, fa_ignovr_collar250, spkerr_ignovr_collar250 = der_eval_overall(rttm_ref_list, rttm_hyp_list, collar=0.25, ignovr=True)
    der_collar250, miss_collar250, fa_collar250, spkerr_collar250 = der_eval_overall(rttm_ref_list, rttm_hyp_list, collar=0.25)
    der_collar0, miss_collar0, fa_collar0, spkerr_collar0 = der_eval_overall(rttm_ref_list, rttm_hyp_list, collar=0)
    
    # Remove temp files
    shutil.rmtree(tmp_dir)
    
    # Print results
    print('Evaluation:')
    print('Forgiving | DER: {:.2f} - MISS: {:.2f} - FA: {:.2f} - SPEAKER ERR: {:.2f}'\
          .format(der_ignovr_collar250, miss_ignovr_collar250, fa_ignovr_collar250, spkerr_ignovr_collar250))
    
    print('Fair      | DER: {:.2f} - MISS: {:.2f} - FA: {:.2f} - SPEAKER ERR: {:.2f}'\
          .format(der_collar250, miss_collar250, fa_collar250, spkerr_collar250))

    print('Full      | DER: {:.2f} - MISS: {:.2f} - FA: {:.2f} - SPEAKER ERR: {:.2f}'\
          .format(der_collar0, miss_collar0, fa_collar0, spkerr_collar0))

    if cfg.opt.eval_ss:
        print('SI-SDRi: {:.2f}'.format(np.mean(ss_metrics_list)))


if __name__ == '__main__':
    test()
