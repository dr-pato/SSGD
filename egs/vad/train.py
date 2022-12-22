import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import numpy as np
import random
from sepdiarize.data.data_loader import AnnotatedDataset
from sepdiarize.data.vad_labels import prep4vad
from sepdiarize.vad.tcn import TCN
from sepdiarize.utils.encoder import ManyHotEncoder
from local.trainer import VADTrainer
from local.tune import tuna
from skopt.space import Real, Integer
from skopt import forest_minimize


np.random.seed(777)
torch.random.manual_seed(777)
random.seed(777)


@hydra.main(config_path="confs", config_name="train")
def single_run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    encoder = ManyHotEncoder(["0", "1"], 10000, cfg.feats.win_length, cfg.feats.hop_length,
                             net_pooling=1, fs=cfg.data.samplerate)

    tr_ds = AnnotatedDataset(os.path.join(cfg.data.root_fisher, "training_set"),
                            os.path.join(cfg.data.parsed, "training_set_p1"),
                            encoder,
                            training=True,
                            segment=cfg.training.segment,
                            samplerate=cfg.data.samplerate)

    
    cv_ds = AnnotatedDataset(os.path.join(cfg.data.root_fisher, "validation_set"),
                            os.path.join(cfg.data.parsed, "validation_set"),
                            encoder,
                            training=False,
                            segment=None,
                            samplerate=cfg.data.samplerate,
                            oversample=1 if cfg.training.oversample > 0 else 0)
    
    vad = TCN(cfg.feats, **cfg.vad)

    # Load checkpoint
    if cfg.ckpt:
        vad_ckpt = torch.load(cfg.ckpt)
        new_state = {".".join(k.split(".")[1:]): v for k, v in vad_ckpt["state_dict"].items()}
        del new_state["pos_weight"]
        vad.load_state_dict(new_state)
        print('Checkpoint loaded: %s' % cfg.ckpt)


    optimizer = torch.optim.Adam(list(vad.parameters()), cfg.opt.lr,
                                 weight_decay=cfg.opt.weight_decay)


    schedulers = [{"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                                     factor=cfg.scheduler.reduce_f,
                                                                     patience=cfg.scheduler.patience,
                                                                     verbose=True
                                                                     ), "monitor": "val/loss"}]

    vad_trainer = VADTrainer(cfg, tr_ds, cv_ds, vad, encoder, optimizer, schedulers)

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
    print(f"Experiment dir: {logger.log_dir}")

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
        trainer.fit(vad_trainer)
    else:
        vad_trainer.load_from_checkpoint(cfg.test_from_checkpoint)
    
    print("RESULTS IN {}".format(os.getcwd()))

    # Load best checkpoint
    best_ckpt = torch.load(trainer.checkpoint_callback.best_model_path)
    
    new_state = {".".join(k.split(".")[1:]): v for k, v in best_ckpt["state_dict"].items() if k.startswith("vad")}
    del new_state["pos_weight"]
    vad.load_state_dict(new_state)

    # Tune vad
    # find best params on fisher dev set which minimize the der
    print("FIND BEST HYPERPARAMS")
    helper = lambda x: tuna(vad, cv_ds, encoder, hyperpars=x)
    median_len = Integer(low=3, high=150, name='median')
    delete_shorter = Real(low=0.1, high=1.0, name='delete_shorter')
    threshold = Real(low=0.1, high=0.9, name='threshold')
    search_result = forest_minimize(helper, [median_len, delete_shorter, threshold], n_calls=cfg.htuning.n_calls)
    print("#" * 100)
    print("#" * 100)
    print("BEST HYPERPARAMS on dev set: {}".format(search_result.x))
    print("DER: ", search_result.fun)


if __name__ == "__main__":
    single_run()