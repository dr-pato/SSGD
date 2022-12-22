import glob
import os
from pathlib import Path
import numpy as np
import jiwer
from collections import OrderedDict
import torchaudio
import torch
import torch
import librosa
from asteroid.losses.sdr import PairwiseNegSDR
from asteroid.losses import PITLossWrapper
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "en"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)


def merge_intervals(intervals, delta=0.0):
    """
    A simple algorithm can be used:
    1. Sort the intervals in increasing order
    2. Push the first interval on the stack
    3. Iterate through intervals and for each one compare current interval
       with the top of the stack and:
       A. If current interval does not overlap, push on to stack
       B. If current interval does overlap, merge both intervals in to one
          and push on to stack
    4. At the end return stack
    """
    if not intervals:
        return intervals
    intervals = sorted(intervals, key=lambda x: x[0])

    merged = [intervals[0]]
    for current in intervals[1:]:
        previous = merged[-1]
        if (current[0] - delta) <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    return merged


def load_transcriptions(transpath, jiwer_pipeline):

    with open(transpath, "r") as f:
        lines = f.readlines()

    diarization = OrderedDict({})
    for l in lines:
        if not (l.startswith("#") or l.startswith("\n")):
            split_space = l.split(" ")
            start = float(split_space[0])
            stop = float(split_space[1])
            speaker = split_space[2]
            words = " ".join(split_space[3:]).strip("\n")
            words = jiwer_pipeline(words)
            if words.strip():
                if speaker not in diarization.keys():
                    diarization[speaker] = [[start, stop, words]]
                else:
                    diarization[speaker].append([start, stop, words])
    return diarization


transcript_root = "/home/gmorrone/data/AGEVOLA/Fisher_wavs/clean_rttm/transcriptions"


def compute_metrics_one_file(model, processor, jiwer_tranformation, hyp_diarization, transcript, estimate_src, oracle_src):
    
    sisdr = PairwiseNegSDR("sisdr")
    pairwise_losses = sisdr(estimate_src.unsqueeze(0), oracle_src.unsqueeze(0))
    _, best_perm = PITLossWrapper.find_best_perm(pairwise_losses)
    estimate_src = PITLossWrapper.reorder_source(estimate_src.unsqueeze(0), best_perm)[0]
    estimate_src = torch.stack([torch.from_numpy(librosa.resample(estimate_src[indx].numpy(), 8000, 16000)) for indx in range(len(estimate_src))])

    spk_1 = [[y, z] for (x, y, z) in hyp_diarization if x == "0"]
    spk_2 = [[y, z] for (x, y, z) in hyp_diarization if x == "1"]

    if best_perm[0][0] == 0:
        hyp_diarization = [spk_1, spk_2]
    else:
        hyp_diarization = [spk_2, spk_1]

    # step1 reorder the estimate
    for spk_indx, spk in enumerate(list(transcript.keys())):
        # get the total transcription for this speaker
        ground_truth = jiwer_pipeline(" ".join([x[-1] for x in list(transcript[spk])]))

        # use diarization to run asr on each segment for this speaker
        hyp_text = []
        for seg in merge_intervals(hyp_diarization[spk_indx], 2):
            start, stop = seg
            c_seg = estimate_src[spk_indx][int(start*16000):int(stop*16000)]
            c_seg = c_seg.numpy()
            if c_seg.shape[-1] <= 16000:
                c_seg = np.pad(c_seg, (0, 16000-c_seg.shape[-1]), mode="constant")
            logits = model(torch.from_numpy(c_seg).unsqueeze(0)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcript_clean = processor.batch_decode(predicted_ids)[0]
            text = jiwer_tranformation(transcript_clean)
            hyp_text.append(text)

        print("a")

    transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveSpecificWords(["[sigh]", "[noise]", "[lipsmack]", "((", "))", "[laughter]", "[mn]"]),
    jiwer.RemoveKaldiNonWords(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    ])
    # we compute the words for every segment
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
    ])

    wer_obj = jiwer.wer(
        ground_truth,
        hypothesis,
        truth_transform=transformation,
        hypothesis_transform=transformation
    )


jiwer_pipeline = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveSpecificWords(["[sigh]", "[noise]", "[lipsmack]", "((", "))", "[laughter]", "[mn]"]),
    jiwer.RemoveKaldiNonWords(),
    jiwer.RemovePunctuation(),
jiwer.RemoveMultipleSpaces(),
])


if __name__ == '__main__':
    # we glob all transcriptions
    transcription = load_transcriptions("/home/samco/dgx2/data/AGEVOLA/Fisher_wavs/clean_rttm/transcriptions/fe_03_05437.txt", jiwer_pipeline)
    oracle_s1, rate = torchaudio.load("/home/samco/dgx2/data/AGEVOLA/Fisher_wavs/clean_rttm/test_set_red/sep_wav/fe_03_05437/oracle/mix_est1.wav")
    oracle_s2, _ = torchaudio.load("/home/samco/dgx2/data/AGEVOLA/Fisher_wavs/clean_rttm/test_set_red/sep_wav/fe_03_05437/oracle/mix_est2.wav")
    oracle = torch.cat((oracle_s1, oracle_s2))
    
    est_s1, _ = torchaudio.load("/home/samco/dgx2/data/AGEVOLA/Fisher_wavs/clean_rttm/test_set_red/sep_wav/fe_03_05437/DPRNNTasNet_FisherMix-Fisher_8knowin/mix_est1.wav")
    est_s2, _ = torchaudio.load("/home/samco/dgx2/data/AGEVOLA/Fisher_wavs/clean_rttm/test_set_red/sep_wav/fe_03_05437/DPRNNTasNet_FisherMix-Fisher_8knowin/mix_est2.wav")
    # oracle segments
    estimate = torch.cat((est_s1, est_s2))
    
    from sepdiarize.utils.diarization import load_rttm
    diarization = load_rttm("/home/samco/dgx2/data/AGEVOLA/Fisher_wavs/clean_rttm/test_set_red/sep_wav/fe_03_05437/DPRNNTasNet_FisherMix-Fisher_8knowin/tcnvad-100-0.00-0.5/mix_est.rttm")
    diarization = diarization[list(diarization.keys())[0]]
    metrics = compute_metrics_one_file(model, processor, jiwer_pipeline, diarization, transcription, estimate, oracle)