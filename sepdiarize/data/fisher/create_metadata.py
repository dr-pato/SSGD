import sys
import os
from glob import glob
from tqdm import tqdm
import pandas as pd
from compute_overlap import read_trn_file


def extract_line(line):
    line = line.split()
    duration_sec = int(float(line[1]) - float(line[0]))
    spk = 0 if line[2][0] == 'A' else 1
    if duration_sec >= 10:
        duration_sec = 9
    return float(line[0]), float(line[1]), spk, ' '.join(line[3:]), duration_sec


def has_noise_only(line, min_len=1):
    trn = line[3]
    dur = line[1] - line[0]
    noise_list = ["[cough]", "[breath]", "[sigh]", "[noise]", "[lipsmack]", "[laugh]", "[laughter]", "[mn]"]
    if any(w in trn for w in noise_list):
        if '((' in trn and dur < min_len:
            return True
        if all(w in noise_list for w in trn.split()):
            return True

    # Return False if none of previous conditions is met
    return False


def read_trn_file(file, noise_opt='all'):
    with open(file, 'r') as f:
        lines = f.readlines()
    
    flines = lines[3:]
    plines = [extract_line(l) for l in flines if l.strip()]
    if noise_opt == 'no':
        plines = [l for l in plines if '[' not in l[3]]
    elif noise_opt == 'partial':
        plines = [l for l in plines if not has_noise_only(l, min_len=1)]


def create_fisher_metadata(trn_files_list, noise_opt='all', out=None):
    df_cols = ['filename', 'start', 'end', 'speaker', 'transcription', 'duration_cat']
    df_list = []
    
    for trn_file in tqdm(trn_files_list):
        trn = read_trn_file(trn_file, noise_opt)
        trn_file_id = os.path.basename(trn_file).replace('.txt', '')
        #trnf = [(int(trn_file_id[-5:]), l[0], l[1], l[2], l[3], l[4]) for l in trn]
        trnf = [(trn_file_id, l[0], l[1], l[2], l[3], l[4]) for l in trn]
        cur_df = pd.DataFrame(data=trnf, columns=df_cols)
        df_list.append(cur_df)

    df = pd.concat(df_list, ignore_index=True)

    if out is not None:
        df.to_csv(out, index=False)

    return df


if __name__ == '__main__':
    # We suppose to have the extracted archive 
    filenames = sorted(glob(os.path.join(sys.argv[1], '**', '*.txt'), recursive=True))
    noise_opt = sys.argv[2]
    output_file = sys.argv[3]
    
    create_fisher_metadata(filenames, noise_opt, output_file)

