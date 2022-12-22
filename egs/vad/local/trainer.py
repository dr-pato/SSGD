import os
from pathlib import Path
import shutil
from collections import defaultdict
import pytorch_lightning as pl
import torch
import numpy as np
import scipy
from speechbrain.dataio.batch import PaddedBatch
from sepdiarize.utils.diarization import load_rttm, write_rttm
from sepdiarize.utils.der_eval import der_eval_overall, convert2pyannote
from pyannote.metrics.diarization import DiarizationErrorRate
from sepdiarize.utils.diarization import delete_shorter


class VADTrainer(pl.LightningModule):
    def __init__(
            self,
            cfg,
            tr_ds,
            cv_ds,
            vad,
            encoder,
            opt,
            scheduler
    ):
        super(VADTrainer, self).__init__()
        self.cfg = cfg
        self.tr_ds = tr_ds
        self.cv_ds = cv_ds
        self.vad = vad
        self.encoder = encoder
        self.opt = opt
        self.scheduler = scheduler

        if self.cfg.debug:
            self.num_workers = 1
        else:
            self.num_workers = self.cfg.training.n_workers

        self.vad_loss = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([self.cfg.training.vad_loss_pos_weight]))
        self.buffer_ref = defaultdict(list)
        self.buffer_hyp = defaultdict(list)

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
        input_sources, _ = batch.sources
        labels, _ = batch.vad_labels
        bsz, spk, _ = input_sources.shape
        
        
        # vad on oracle sources
        preds = self.vad(input_sources)
        
        labels = labels[..., :preds.shape[-1]]
        loss = self.vad_loss(preds, labels).mean()
            
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/lr", self.opt.param_groups[-1]["lr"])

        return loss

    def validation_step(self, batch, batch_indx):
     
        mixture, _ = batch.mixture
        input_sources, _ = batch.sources
        labels, _ = batch.vad_labels
        bsz, spk, _ = input_sources.shape

        with torch.inference_mode():
            preds = self.vad(input_sources)
        
        labels = labels[..., :preds.shape[-1]]
        
        loss = self.vad_loss(preds, labels).mean()
        vad_prob = torch.sigmoid(preds)
            
        rttm_file = batch.rttm_file[0]
        buffer_ref = load_rttm(rttm_file)
        filename = Path(rttm_file).stem
        self.buffer_ref.update(buffer_ref)
        self.buffer_hyp[filename] = []

        # We use fixed hyperparameters during training
        sad = vad_prob[0].cpu().numpy() > 0.5
        sad = scipy.ndimage.filters.median_filter(sad, (1, 100))
        decoded = self.encoder.decode_strong(sad.T)
        decoded = delete_shorter(decoded, 0.25)
        self.buffer_hyp[filename].extend(decoded)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_epoch_end(self, outputs):
    
        if len(self.buffer_ref) > 0:
            der_full = DiarizationErrorRate()
            der_fair = DiarizationErrorRate(0.5)

            for reco_id in self.buffer_ref.keys():
                if reco_id not in self.buffer_hyp.keys():
                    raise NotImplementedError
                ref, hyp = convert2pyannote(self.buffer_ref[reco_id], self.buffer_hyp[reco_id])
                der_full(ref, hyp)
                der_fair(ref, hyp)

            components_full = der_full[:]
            self.log("val/falarm_full", torch.tensor([components_full["missed detection"] / components_full["total"]]),
                     on_epoch=True, prog_bar=False)
            self.log("val/miss_full", torch.tensor([components_full["false alarm"] / components_full["total"]]),
                     on_epoch=True, prog_bar=False)
            self.log("val/conf_full", torch.tensor([components_full["confusion"] / components_full["total"]]),
                     on_epoch=True, prog_bar=False)
            self.log("val/der_full", torch.tensor([abs(der_full)]), on_epoch=True, prog_bar=True)

            components_fair = der_fair[:]
            self.log("val/falarm_fair", torch.tensor([components_fair["missed detection"] / components_fair["total"]]),
                     on_epoch=True, prog_bar=True)
            self.log("val/miss_fair", torch.tensor([components_fair["false alarm"] / components_fair["total"]]),
                     on_epoch=True, prog_bar=True)
            self.log("val/conf_fair", torch.tensor([components_fair["confusion"] / components_fair["total"]]),
                     on_epoch=True, prog_bar=True)
            self.log("val/der_fair", torch.tensor([abs(der_fair)]), on_epoch=True, prog_bar=True)

            self.buffer_ref = defaultdict(list)
            self.buffer_hyp = defaultdict(list)