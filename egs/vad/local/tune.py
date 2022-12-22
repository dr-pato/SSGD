import os
from pathlib import Path
import shutil
from tqdm import tqdm
import numpy as np
import torch
import scipy
from pyannote.metrics.diarization import DiarizationErrorRate
from sepdiarize.utils.css_with_lookahead import css_lookahead
from sepdiarize.utils.diarization import load_rttm, write_rttm, delete_shorter
from sepdiarize.utils.der_eval import convert2pyannote


def evaluate(vad, dataset, encoder, smoother=50, delete_shorter_than=0.25, threshold=0.5, collar=0.25, use_gpu=True):
    
    vad.eval()

    if use_gpu:
        vad = vad.cuda()

    dermetric = DiarizationErrorRate(collar * 2)

    for ex in tqdm(dataset):
        sources = ex["sources"]
        sources = sources.unsqueeze(0)
        assert sources.shape[0] == 1

        if use_gpu:
            sources = sources.cuda()

        bsz, src, _ = sources.shape

        with torch.inference_mode():
            preds = vad(sources)
        vad_prob = torch.sigmoid(preds)

        rttm_file = ex["rttm_file"]
        filename = Path(rttm_file).stem
        vad_prob = vad_prob.reshape(bsz, src, -1)
        sad = vad_prob[0].cpu().numpy() > threshold
        sad = scipy.ndimage.filters.median_filter(sad, (1, smoother))
        decoded = encoder.decode_strong(sad.T)
        decoded = delete_shorter(decoded, delete_shorter_than)
        reference = load_rttm(rttm_file)
        reference, hypothesis = convert2pyannote(reference[filename], decoded)
        dermetric(reference, hypothesis)

    der = abs(dermetric)
    components_full = dermetric[:]
    miss = components_full["missed detection"] / components_full["total"]
    fa = components_full["false alarm"] / components_full["total"]
    conf = components_full["confusion"] / components_full["total"]

    return {"DER": der, "MISS": miss, "FA": fa, "CONF": conf}


def tuna(vad, dataset, encoder, hyperpars):

    smoother, delete_shorter_than, threshold = hyperpars
    f_score_train = evaluate(vad, dataset, encoder, smoother=smoother, delete_shorter_than=delete_shorter_than,
                             threshold=threshold, collar=0.25, use_gpu=True)

    print("SCORES {}".format(f_score_train))
    print("hyperpars {}".format(hyperpars))

    return f_score_train["DER"]  # maximize


