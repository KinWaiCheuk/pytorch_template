import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import TriStageLRSchedule
from utils import extract_notes_wo_velocity, transcription_accuracy
from utils.text_processing import GreedyDecoder
import fastwer
import contextlib

# from nnAudio.Spectrogram import MelSpectrogram
import pandas as pd

class SpeechCommand(pl.LightningModule):
    def __init__(self,
                 model,
                 lr):
        super().__init__()
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = model

    def training_step(self, batch, batch_idx):
        x = batch['waveforms']
        output = self.model(x)
        loss = self.loss_fn(output["prediction"],batch['labels'].long())      
        self.log("Train/Loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['waveforms']
        with torch.no_grad():
            output = self.model(x)
            loss = self.loss_fn(output["prediction"],batch['labels'].long())      
            self.log("val_CE_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['waveforms']
        with torch.no_grad():
            output = self.model(x)
            loss = self.loss_fn(output["prediction"],batch['labels'].long())      
            self.log("test_CE_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]