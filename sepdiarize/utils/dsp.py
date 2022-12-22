import torch
import numpy as np
from pb_bss_eval.evaluation import si_sdr


def compute_sisdr(mix, sources, estimation, compute_permutation=True):
    output_dict = {}
    output_dict['input_si_sdr'] = si_sdr(sources, mix).mean()
    output_dict['si_sdr'] = si_sdr(sources, estimation).mean()
    output_dict['perm'] = [0, 1]
    if compute_permutation:
        si_sdr_perm = si_sdr(sources, estimation[[1, 0]]).mean()
        if si_sdr_perm > output_dict['si_sdr']:
            output_dict['si_sdr'] = si_sdr_perm
            output_dict['perm'] = [1, 0]
            
    return output_dict


def check_clipping(src, max_amp=0.9):
    max_val = src.abs().max()
    if max_val >= 1:
        weight = max_amp / max_val
        src = src * weight
    else:
        weight = 1

    return src, weight


def remove_leakage(sources, mix, seglen=8000, threshold=5):
    sources, mix = np.broadcast_arrays(sources, mix)
    start_ids = np.arange(0, sources.shape[-1], seglen)
    sources_mod = sources.copy()
    count = 0
    for idx in start_ids:
        seg_sources = sources[:, idx: idx + seglen]
        seg_mix = mix[:, idx: idx + seglen]
        seg_sisdr = si_sdr(seg_sources, seg_mix)
        if np.all(seg_sisdr > threshold): 
            sources_mod[seg_sisdr.argmin(), idx: idx + seglen] = 0
            count += 1
            
    return sources_mod, count
