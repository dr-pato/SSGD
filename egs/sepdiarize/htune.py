import os
from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import random
from asteroid.models import ConvTasNet, DPRNNTasNet
from sepdiarize.data.data_loader import AnnotatedDataset
from sepdiarize.vad.tcn import TCN
from sepdiarize.utils.encoder import ManyHotEncoder
from local.tune import tuna
from skopt.space import Real, Integer
from skopt import forest_minimize


np.random.seed(777)
torch.random.manual_seed(777)
random.seed(777)


@hydra.main(config_path="confs", config_name="htune")
def single_run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Load sep conf
    sep_conf = OmegaConf.load(os.path.join(Path(cfg.sep_ckpt).parent.parent.parent,
                                              '.hydra', 'config.yaml'))
    
    # Load VAD conf
    vad_conf = OmegaConf.load(os.path.join(Path(cfg.vad_ckpt).parent.parent.parent,
                                               ".hydra", "config.yaml"))
    
    encoder = ManyHotEncoder(["0", "1"], 10000, vad_conf.feats.win_length, vad_conf.feats.hop_length,
                             net_pooling=1, fs=cfg.data.samplerate)

    cv_ds = AnnotatedDataset(cfg.data.adapt_set,
                             cfg.data.adapt_parsed,
                             encoder,
                             training=False,
                             segment=None,
                             samplerate=cfg.data.samplerate,
                             has_oracle=cfg.data.oracle_sources,
                             oversample=1)
    
    # Separator
    if sep_conf.training.net == 'convtasnet':
        if sep_conf.training.causal:
            separator = ConvTasNet(2, norm_type='cLN', causal=True)
        else:
            separator = ConvTasNet(2)
    elif sep_conf.training.net == 'dprnn':
        if sep_conf.training.causal:
            separator = DPRNNTasNet(2, norm_type='cLN', bidirectional=False)
        else:
            separator = DPRNNTasNet(2)
    else:
        raise NotImplementedError

    # VAD
    vad = TCN(vad_conf.feats, **vad_conf.vad)
    
    # Load checkpoints
    sep_ckpt = torch.load(cfg.sep_ckpt)
    new_state = {".".join(k.split(".")[1:]): v for k, v in sep_ckpt["state_dict"].items()}
    separator.load_state_dict(new_state)
    
    vad_ckpt = torch.load(cfg.vad_ckpt)
    new_state = {".".join(k.split(".")[1:]): v for k, v in vad_ckpt["state_dict"].items() if k.startswith("vad")}
    del new_state["pos_weight"]
    vad.load_state_dict(new_state)

    # tune ssgd system
    # find best params on dev set which minimize the der
    print("FIND BEST HYPERPARAMS")
    helper = lambda x: tuna(separator, vad, cv_ds, encoder, cfg.css, hyperpars=x)
    median_len = Integer(low=20, high=150, name='median')
    delete_shorter = Real(low=0.0, high=1.0, name='delete_shorter')
    threshold = Real(low=0.3, high=0.8, name='threshold')
    search_result = forest_minimize(helper, [median_len, delete_shorter, threshold], n_calls=cfg.htuning.n_calls)
    print("#" * 100)
    print("#" * 100)
    print("BEST HYPERPARAMS on dev set: {}".format(search_result.x))
    print("DER: ", search_result.fun)


if __name__ == "__main__":
    single_run()