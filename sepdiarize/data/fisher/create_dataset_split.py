import sys
import os
from glob import glob
import random
import numpy as np
from tqdm import tqdm
import soundfile as sf
from sepdiarize.utils.diarization import collapse_segments
from sepdiarize.utils.dsp import check_clipping

random.seed(42)

# Replace with filenames inside the repository
TRAIN_IDS_FILENAME = '/raid/users/gmorrone/data/AGEVOLA/Fisher_wavs/clean_rttm/train_id_list.txt'
VAL_IDS_FILENAME = '/raid/users/gmorrone/data/AGEVOLA/Fisher_wavs/clean_rttm/val_id_list.txt'
TEST_IDS_FILENAME = '/raid/users/gmorrone/data/AGEVOLA/Fisher_wavs/clean_rttm/test_id_list.txt'


def read_rttm(rttm_file):
    with open(rttm_file, 'r') as f:
        rttm_lines= f.readlines()
    lines = [l.split() for l in rttm_lines]
    lines = [[l[1], float(l[3]), float(l[4]), l[7]] for l in lines]
        
    return lines


def write_rttm(rttm_lines, out_file):
    out_list = ['SPEAKER {:s} 1 {:.2f} {:.2f} <NA> <NA> {:s} <NA> <NA>\n'.format(l[0], l[1], l[2], l[3]) for l in rttm_lines]
        
    with open(out_file, 'w') as f:
        f.writelines(out_list)


def write_vadlab(vad_lines, out_file):
    out_list = ['{:.2f} {:.2f} sp\n'.format(l[0], l[1]) for l in vad_lines]

    with open(out_file, 'w') as f:
        f.writelines(out_list)


def read_list_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f]

    return lines


def process_2ch_data(data_dir):
    wav_dir = os.path.join(data_dir, 'wav')
    mixwav_dir = os.path.join(data_dir, 'mix')
    
    # Create output dir if not exist
    os.makedirs(mixwav_dir, exist_ok=True)
    
    wav_list = glob(os.path.join(wav_dir, '*.wav'))
    for wav_file in tqdm(wav_list):
        id = os.path.basename(wav_file).replace('.wav', '')
        wav, sr = sf.read(wav_file, dtype='float32')

        mix = np.sum(wav, axis=1)
        mix, weight = check_clipping(mix)

        sf.write(mixwav_file, mix, sr)


def create_dataset(wav_dir, metadata_dir, file_ids, output_dir):
    out_wav_dir = os.path.join(output_dir, 'wav')
    out_rttm_dir = os.path.join(output_dir, 'rttm')
    out_vad_dir = os.path.join(output_dir, 'oraclevad')
    os.makedirs(out_wav_dir, exist_ok=True)
    os.makedirs(out_rttm_dir, exist_ok=True)
    os.makedirs(out_vad_dir, exist_ok=True)
    
    total_len = 0
    for file_id in tqdm(file_ids):
        sel_file_ids.append(file_id)
        wav_fname = os.path.join(wav_dir, file_id + '.wav')
        rttm_fname = os.path.join(metadata_dir, file_id, 'rttm', 'mix.rttm')
        out_wav_fname = os.path.join(out_wav_dir, file_id + '.wav')
        out_rttm_fname = os.path.join(out_rttm_dir, file_id + '.rttm')
        out_vad_fname = os.path.join(out_vad_dir, file_id + '.lab')
        
        # RTTM
        rttm_lines = read_rttm(rttm_fname)
        start = rttm_lines[0][1]
        end = rttm_lines[-1][1] + rttm_lines[-1][2]
        rttm_lines_adj = [[l[0], l[1] - start, l[2], l[3]] for l in rttm_lines]
        write_rttm(rttm_lines_adj, out_rttm_fname)

        # VAD labels
        vad_lines = [[l[1], l[1] + l[2]] for l in rttm_lines_adj]
        vad_lines_collapsed = collapse_segments(vad_lines)
        write_vadlab(vad_lines_collapsed, out_vad_fname)
        
        wav, sr = sf.read(wav_fname, dtype='float32')
        wav = wav[int(start * sr): int(end * sr), :]
        sf.write(out_wav_fname, wav, sr)
        total_len += len(wav) / sr
        
    # Write list file
    out_list_file = os.path.join(output_dir, 'list.txt')
    out_list = [l + '\n' for l in sorted(sel_file_ids)]
    with open(out_list_file, 'w') as f:
        f.writelines(out_list)


if __name__ == '__main__':
    wav_dir = sys.argv[1]
    metadata_dir = sys.argv[2]
    output_dir =  sys.argv[3]
    
    # Training data
    print('Creating training data...')
    train_dir = os.path.join(output_dir, 'training_set')
    train_file_ids = read_list_file(TRAIN_IDS_FILENAME)
    create_dataset(wav_dir, metadata_dir, train_file_ids, train_dir)
    process_2ch_data(train_dir)
    
    # Validation data
    print('Creating validation data...')
    val_dir = os.path.join(output_dir, 'validation_set')
    val_file_ids = read_list_file(VAL_IDS_FILENAME)
    sel_val_ids = create_dataset(wav_dir, metadata_dir, val_file_ids, val_dir)
    process_2ch_data(val_dir)
    
    # Test data
    print('Creating test data...')
    test_dir = os.path.join(output_dir, 'test_set')
    test_file_ids = read_list_file(TEST_IDS_FILENAME)
    sel_test_ids = create_dataset(wav_dir, metadata_dir, test_file_ids, test_dir)
    process_2ch_data(test_dir)