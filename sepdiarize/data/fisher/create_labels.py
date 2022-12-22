import sys
import os
import pandas
from tqdm import tqdm
from sepdiarize.utils.diarization import collapse_segments


def rttm2vad(rttm_lines):
    segments = [[rl[0], rl[0] + rl[1]] for rl in rttm_lines]
    vad_lines = collapse_segments(segments)

    return vad_lines


def rttmlines2file(id, rttm_lines):
    out_lines = []
    for l in rttm_lines:
        rttm = 'SPEAKER {:s} 1 {:.6f} {:.6f} <NA> <NA> {:s} <NA> <NA>\n'.format(id, l[0], l[1], str(l[2]))
        out_lines.append(rttm)

    return out_lines


def vadlines2file(segments):
    out_lines = []
    for vl in segments:
        out_lines.append('{:.6f} {:.6f} sp\n'.format(vl[0], vl[1]))

    return out_lines


def rttm_vad_labels_from_md(md):
    mixture_id = md['filename'].iloc[0]
    rttm_lines_s1 = []
    vad_lines_s1 = []
    rttm_lines_s2 = []
    vad_lines_s2 = []

    rttm_lines_mix = []
    vad_lines_mix = []
    for i, row in md.iterrows():
        rttm_line = [row['start'], row['end'] - row['start'], row['speaker']]
        if row['speaker'] == 0:
            rttm_lines_s1.append(rttm_line)
        else:
            rttm_lines_s2.append(rttm_line)
        rttm_lines_mix.append(rttm_line)
    # Create VAD labels
    vad_lines_s1 = rttm2vad(rttm_lines_s1)
    vad_lines_s2 = rttm2vad(rttm_lines_s2)
    vad_lines_mix = rttm2vad(rttm_lines_mix)
    # Create RTTM formatted lines
    rttm_lines_s1 = rttmlines2file(mixture_id, rttm_lines_s1)
    rttm_lines_s2 = rttmlines2file(mixture_id, rttm_lines_s2)
    rttm_lines_mix = rttmlines2file(mixture_id, rttm_lines_mix)
    # Create RTTM formatted lines
    vad_lines_s1 = vadlines2file(vad_lines_s1)
    vad_lines_s2 = vadlines2file(vad_lines_s2)
    vad_lines_mix = vadlines2file(vad_lines_mix)

    return rttm_lines_s1, rttm_lines_s2, rttm_lines_mix, vad_lines_s1, vad_lines_s2, vad_lines_mix


def write_file(out_list, out_file):
    with open(out_file, 'w') as f:
        f.writelines(out_list)


if __name__ == '__main__':
    metadata_file = sys.argv[1]
    out_dir = sys.argv[2]

    total_md = pandas.read_csv(metadata_file)

    id_list = total_md['filename'].unique()
    for id in tqdm(id_list):
        id_rows = total_md[total_md['filename'] == id]

        rttm_dir = os.path.join(out_dir, id, 'rttm')
        vad_dir = os.path.join(out_dir, id, 'vad')
        if not os.path.exists(rttm_dir):
            os.makedirs(rttm_dir, exist_ok=True)
        if not os.path.exists(vad_dir):
            os.makedirs(vad_dir, exist_ok=True)

        # Generating oracle RTTM and VAD labels
        rttm_lines_s1, rttm_lines_s2, rttm_lines_mix, vad_lines_s1, vad_lines_s2, vad_lines_mix = rttm_vad_labels_from_md(id_rows)
        
        # Saving files
        write_file(rttm_lines_s1, os.path.join(rttm_dir, 'source_1.rttm'))
        write_file(rttm_lines_s2, os.path.join(rttm_dir, 'source_2.rttm'))
        write_file(rttm_lines_mix, os.path.join(rttm_dir, 'mix.rttm'))
        
        write_file(vad_lines_s1, os.path.join(vad_dir, 'source_1.lab'))
        write_file(vad_lines_s2, os.path.join(vad_dir, 'source_2.lab'))
        write_file(vad_lines_mix, os.path.join(vad_dir, 'mix.lab'))
