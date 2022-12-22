import os
from pathlib import Path
import torch
import pytorch_lightning as pl
import numpy as np
from speechbrain.dataio.batch import PaddedBatch
from asteroid.losses import PITLossWrapper
from asteroid.losses.sdr import PairwiseNegSDR
from sepdiarize.utils.css_with_lookahead import css_lookahead


class SepTrainer(pl.LightningModule):
    def __init__(
            self,
            cfg,
            tr_ds,
            cv_ds,
            separator,
            opt,
            scheduler
    ):
        super(SepTrainer, self).__init__()
        self.cfg = cfg
        self.tr_ds = tr_ds
        self.cv_ds = cv_ds
        self.separator = separator
        self.opt = opt
        self.scheduler = scheduler

        if self.cfg.debug:
            self.num_workers = 1
        else:
            self.num_workers = self.cfg.training.n_workers


        self.sisdr = PairwiseNegSDR("sisdr")
        self.sep_loss = PITLossWrapper(self.sisdr)

    def configure_optimizers(self):

        return [self.opt], self.scheduler

    def train_dataloader(self):

        self.train_loader = torch.utils.data.DataLoader(
            self.tr_ds,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            worker_init_fn=lambda x: np.random.seed(int.from_bytes(os.urandom(4), "little") + x),
            collate_fn=PaddedBatch,
        )

        return self.train_loader

    def val_dataloader(self):

        self.val_loader = torch.utils.data.DataLoader(
            self.cv_ds,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=PaddedBatch
        )
        return self.val_loader

    def training_step(self, batch, batch_indx):
        
        mixture, _ = batch.mixture
        sources, _ = batch.sources

        separated = self.separator(mixture)
        
        loss = self.sep_loss(separated, sources)

        self.log("train/loss", loss, on_epoch=True)
        self.log("{}".format(os.getcwd()), 1, prog_bar=False)
        self.log("train/lr", self.opt.param_groups[-1]["lr"])

        return loss

    def validation_step(self, batch, batch_indx):

        mixture, _ = batch.mixture
        sources, _ = batch.sources

        with torch.inference_mode():
            if self.cfg.css.window_size is None:
                separated = self.separator(mixture)
            else:
                separated = css_lookahead(mixture.unsqueeze(1), self.separator, 2, self.cfg.css.window_size,
                                          self.cfg.css.stride, fs=self.cfg.data.samplerate,
                                          window_type=self.cfg.css.window_type, lookbehind=self.cfg.css.lookbehind)
        loss = self.sep_loss(separated, sources)

        self.log("val/loss", loss, on_epoch=True)
        self.log("{}".format(os.getcwd()), 1, prog_bar=False)

        return loss