import sys
import os
import glob
from sepdiarize.utils.encoder import ManyHotEncoder
from sepdiarize.utils.diarization import load_rttm
import torchaudio
from pathlib import Path
import soundfile as sf
import tqdm
import numpy as np


def prep4vad(root, parsed_dir, WINSIZE, STRIDE, NET_POOLING, SAMPLERATE):

    os.makedirs(parsed_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(root, "wav", "*.wav"))

    tot_2_speech = 0
    tot_1_speech = 0
    tot_silence = 0
    tot_frames = 0
    for w in tqdm.tqdm(wav_files):
        infos = torchaudio.info(w)
        assert infos.sample_rate == SAMPLERATE
        c_len = infos.num_frames / infos.sample_rate

        encoder = ManyHotEncoder(["0", "1"], c_len, WINSIZE, STRIDE, NET_POOLING, fs=SAMPLERATE)
        filename = Path(w).stem
        sads = load_rttm(os.path.join(Path(w).parent.parent, "rttm", filename + ".rttm"))
        sads = sads[filename]
        names = list(set([x[0] for x in sads]))
        names2indx = {names[indx]: str(indx) for indx in range(len(names))}

        sads = [[names2indx[x], y, z] for (x, y, z) in sads]

        labels = encoder.encode_strong_df(sads).astype("float32")
        
        tot_frames += labels.size
        tot_silence += np.sum((labels == 0).astype("float32"))
        tot_1_speech += np.sum(np.logical_xor(labels[:, 0] == 1, labels[:, 1] == 1).astype("float32"))
        tot_2_speech += np.sum(np.logical_and(labels[:, 0] == 1, labels[:, 1] == 1).astype("float32"))
        sf.write(os.path.join(parsed_dir, filename + ".wav"), labels, 1, subtype="FLOAT")
      
    print(root)
    print("TOT FRAMES {} TOT SILENCE {} TOT 1 SPEECH {} TOT 2 SPEECH {}".format(tot_frames, tot_silence,
                                                                                tot_1_speech, tot_2_speech))
    print("#"*10)


if __name__ == '__main__':
    root_fisher = sys.argv[1]
    root_callhome = sys.argv[2]

    # Fisher
    prep4vad(os.path.join(root_fisher, "training_set"), os.path.join(cfg.data.parsed, "training_set"),
             cfg.feats.win_length, cfg.feats.hop_length, 1, cfg.data.samplerate)
    prep4vad(os.path.join(root_fisher, "validation_set"), os.path.join(cfg.data.parsed, "validation_set"),
             cfg.feats.win_length, cfg.feats.hop_length, 1, cfg.data.samplerate)
    prep4vad(os.path.join(root_fisher, "test_set"), os.path.join(cfg.data.parsed, "test_set"),
             cfg.feats.win_length, cfg.feats.hop_length, 1, cfg.data.samplerate)
    # Callhome
    prep4vad(cfg.data.callhome_adapt, os.path.join(cfg.data.parsed, "callhome_adapt"),
         vad_conf.feats.win_length, vad_conf.feats.hop_length, 1, cfg.data.samplerate)
    prep4vad(cfg.data.callhome_test, os.path.join(cfg.data.parsed, "callhome_test"),
         vad_conf.feats.win_length, vad_conf.feats.hop_length, 1, cfg.data.samplerate)
    