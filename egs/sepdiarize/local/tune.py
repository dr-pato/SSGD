import os
import tqdm
from pathlib import Path
import numpy as np
import torch
import scipy
from librosa.sequence import viterbi_discriminative, transition_loop
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from pyannote.metrics.segmentation import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from sepdiarize.data.data_loader import AnnotatedDataset
from sepdiarize.utils.css_with_lookahead import css_lookahead
from sepdiarize.utils.diarization import load_rttm, delete_shorter
from sepdiarize.utils.dsp import check_clipping
from sepdiarize.utils.der_eval import convert2pyannote


def evaluate(separator, vad, dataset, encoder, css_conf, hyperpars, collar=0.25, use_gpu=True):

    separator.eval()
    vad.eval()

    if use_gpu:
        separator = separator.cuda()
        vad = vad.cuda()

    dermetric = DiarizationErrorRate(collar * 2)

    for ex in tqdm.tqdm(dataset):
        mixture = ex["mixture"]
        mixture = mixture.unsqueeze(0)
        assert mixture.shape[0] == 1

        if use_gpu:
            mixture = mixture.cuda()

        with torch.inference_mode():
            if css_conf.window_size is None:
                separated = separator(mixture)
            else:
                separated = css_lookahead(mixture.unsqueeze(1), separator, 2, css_conf.window_size,
                                          css_conf.stride, fs=8000,
                                          window_type=css_conf.window_type, lookbehind=css_conf.lookbehind)
        # Adjust out of range values
        separated, _ = check_clipping(separated)

        bsz, src, _ = separated.shape
        assert separated.shape[0] == 1


        with torch.inference_mode():
            preds = vad(separated)[0]
        vad_prob = torch.sigmoid(preds)

        rttm_file = ex["rttm_file"]
        filename = Path(rttm_file).stem
        sad = vad_prob.cpu().numpy() > hyperpars['threshold']
        sad = scipy.ndimage.filters.median_filter(sad, (1, hyperpars['median_len']))
        decoded = encoder.decode_strong(sad.T)
        decoded = delete_shorter(decoded, hyperpars['del_factor'])
        reference = load_rttm(rttm_file)
        reference, hypothesis = convert2pyannote(reference[filename], decoded)
        dermetric(reference, hypothesis)

    der = abs(dermetric)
    components_fair = dermetric[:]
    miss = components_fair["missed detection"] / components_fair["total"]
    fa = components_fair["false alarm"] / components_fair["total"]
    conf = components_fair["confusion"] / components_fair["total"]

    return {"DER": der, "MISS": miss, "FA": fa, "CONF": conf}


def tuna(separator, vad, dataset, encoder, css_conf, hyperpars):

    hyperpars = {'median_len': hyperpars[0], 'del_factor': hyperpars[1], 'threshold': hyperpars[2]}
    f_score_train = evaluate(separator, vad, dataset, encoder, css_conf, hyperpars, collar=0.25, use_gpu=True)

    print("SCORES {}".format(f_score_train))
    print("HYPERPARAMS {}".format(hyperpars))

    return f_score_train["DER"]  # maximize