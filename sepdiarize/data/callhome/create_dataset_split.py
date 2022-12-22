import sys
import os
import shutil
import numpy as np
import pandas as pd
from sepdiarize.utils.diarization import collapse_segments


def create_dataset(wav_dir, rttm_file, vad_file, out_dir):
    full_rttm = pd.read_csv(rttm_file, names=['type', 'id', 'ch', 'start', 'len', 'ot1', 'ot2', 'spkid', 'ot3'], delim_whitespace=True)
    full_rttm = full_rttm.sort_values(['id', 'start'])
    id_list = full_rttm['id'].unique().tolist()

    full_segs = pd.read_csv(vad_file, names=['segid', 'id', 'start', 'end'], delim_whitespace=True)

    # Create output directories
    out_wav = os.path.join(out_dir, 'wav')
    out_rttm = os.path.join(out_dir, 'rttm')
    out_vad = os.path.join(out_dir, 'oraclevad')
    
    os.makedirs(out_wav, exist_ok=True)
    os.makedirs(out_rttm, exist_ok=True)
    os.makedirs(out_vad, exist_ok=True)
    
    # Create id list file
    list_fname = os.path.join(out_dir, 'list.txt')
    list_f = open(list_fname, 'w')

    for id in id_list:
        # Copy wav
        shutil.copy(os.path.join(wav_dir, id + '.wav'), os.path.join(out_wav, id + '.wav'))
        
        # Generate RTTM file
        rttm = full_rttm[full_rttm['id'] == id]
        rttm.to_csv(os.path.join(out_rttm, id + '.rttm'), sep=' ', na_rep='<NA>', header=False, index=False)

        # Generate oracle VAD label file
        segs = full_segs[full_segs['id'] == id]
        vadlab = segs.filter(['start', 'end'], axis=1).sort_values(['start'])
        vadlab = pd.DataFrame(collapse_segments(vadlab.to_numpy()))
        vadlab['type'] = 'sp'
        vadlab.to_csv(os.path.join(out_vad, id + '.lab'), sep=' ', header=False, index=False)

        # Add row to list file
        list_f.writelines(id + '\n')

    list_f.close()


if __name__ == '__main__':
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    # Callhome 1 - 2spk: adaptation set
    callhome1_2spk_org = os.path.join(data_dir, 'callhome1_2spk')
    callhome1_2spk_out = os.path.join(out_dir, 'callhome1_2spk')
    create_dataset(callhome1_2spk_org)

    # Callhome 2 - 2spk: test set
    callhome2_2spk_org = os.path.join(data_dir, 'callhome2_2spk')
    callhome2_2spk_out = os.path.join(out_dir, 'callhome2_2spk')
    create_dataset(callhome2_2spk_org)