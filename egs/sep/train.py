import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from local.trainer import SepTrainer
import numpy as np
import random
from asteroid.engine.schedulers import DPTNetScheduler
from asteroid.models import ConvTasNet, DPRNNTasNet
from sepdiarize.data.data_loader import WavDataset, AnnotatedDataset
from sepdiarize.utils.encoder import ManyHotEncoder

np.random.seed(777)
torch.random.manual_seed(777)
random.seed(777)


@hydra.main(config_path="confs", config_name="train")
def single_run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    SAMPLERATE = cfg.data.samplerate

    WINSIZE = 200
    STRIDE = 80
    NET_POOLING = 1

    encoder = ManyHotEncoder(["0", "1"], 10000, WINSIZE, STRIDE, net_pooling=NET_POOLING, fs=SAMPLERATE)
        
    # parse dataset and create the dev and test json
    tr_fully_ovl = WavDataset(os.path.join(cfg.data.fully_ovl, "training_set"),
                               training=True,
                               segment=cfg.training.segment,
                               samplerate=SAMPLERATE,
                               oversample=cfg.training.oversample_synt)
    
    tr_original_p1 = AnnotatedDataset(os.path.join(cfg.data.root_fisher, "training_set"),
                            os.path.join(cfg.data.parsed, "training_set_p1"),
                            encoder,
                            training=True,
                            segment=cfg.training.segment,
                            samplerate=SAMPLERATE,
                            oversample=cfg.training.oversample)

    tr_ds = torch.utils.data.ConcatDataset([tr_fully_ovl, tr_original_p1])

    cv_fully_ovl = WavDataset(os.path.join(cfg.data.fully_ovl, "validation_set"),
                       training=False,
                       segment=None,
                       samplerate=SAMPLERATE,
                       oversample=1 if cfg.training.oversample_synt > 0 else 0)

    cv_original = AnnotatedDataset(os.path.join(cfg.data.root_fisher, "validation_set"),
                            os.path.join(cfg.data.parsed, "validation_set"),
                            encoder,
                            training=False,
                            segment=None,
                            samplerate=SAMPLERATE,
                            oversample=1 if cfg.training.oversample > 0 else 0)

    cv_ds = torch.utils.data.ConcatDataset([cv_fully_ovl, cv_original])

    if cfg.training.net == "convtasnet":
        if cfg.training.causal:
            separator = ConvTasNet(2, norm_type="cLN", causal=True)
        else:
            separator = ConvTasNet(2)
    elif cfg.training.net == "dprnn":
        if cfg.training.causal:
            separator = DPRNNTasNet(2, norm_type="cLN", bidirectional=False)
        else:
            separator = DPRNNTasNet(2)
    else:
        raise NotImplementedError

    # Load checkpoint
    if cfg.ckpt:
        sep_ckpt = torch.load(cfg.ckpt)
        new_state = {".".join(k.split(".")[1:]): v for k, v in sep_ckpt["state_dict"].items()}
        separator.load_state_dict(new_state)
        print('Checkpoint loaded: %s' % cfg.ckpt)

    optimizer = torch.optim.Adam(list(separator.parameters()), cfg.opt.lr,
                                 weight_decay=cfg.opt.weight_decay)

    schedulers = [{"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                                     factor=cfg.scheduler.reduce_f,
                                                                     patience=cfg.scheduler.patience,
                                                                     verbose=True
                                                                     ), "monitor": "val/loss"}]
    
    sep_trainer = SepTrainer(cfg, tr_ds, cv_ds, separator, optimizer, schedulers)

    if cfg.debug:
        flush_logs_every_n_steps = 1
        log_every_n_steps = 1
        limit_train_batches = 2
        limit_val_batches = 2
        limit_test_batches = 2
        n_epochs = 3
    else:
        flush_logs_every_n_steps = 1
        log_every_n_steps = 1
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = cfg.training.max_epochs

    logger = TensorBoardLogger(
        os.getcwd())
    print(f"experiment dir: {logger.log_dir}")

    callbacks = [
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.training.early_stop_patience,
            verbose=True,
            mode="min",
        ),
        ModelCheckpoint(
            logger.log_dir,
            monitor="val/loss",
            save_top_k=3,
            mode="min",
            save_last=True,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=callbacks,
        gpus=cfg.gpus,
        accumulate_grad_batches=cfg.training.accumulate_batches,
        logger=logger,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
        gradient_clip_val=cfg.training.gradient_clip,
        check_val_every_n_epoch=cfg.training.validation_interval,
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        flush_logs_every_n_steps=flush_logs_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )

    if cfg.test_from_checkpoint is None:
        trainer.fit(sep_trainer)
    else:
        sep_trainer.load_from_checkpoint(cfg.test_from_checkpoint)

    print("RESULTS IN {}".format(os.getcwd()))


if __name__ == "__main__":
    single_run()