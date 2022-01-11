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




class AMT(pl.LightningModule):
    def __init__(self,
                 model,
                 lr,
                 sr,
                 hop_length,
                 min_midi
                ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.sr = sr
        self.hop_length = hop_length
        self.min_midi = min_midi

    def training_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['frame']    
        output = self.model(x)
        pred = torch.sigmoid(output["prediction"])
        
        # removing extra time step occurs in either label or prediction
        max_timesteps = min(pred.size(1), y.size(1))
        y = y[:, :max_timesteps]
        pred = pred[:, :max_timesteps]
        
        l1_loss = torch.norm(pred, 1, 2).mean()
        bce_loss = F.binary_cross_entropy(pred, y)
        loss = bce_loss#+l1_loss
        self.log("Train/BCE", bce_loss)        
        
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['frame']        

        output = self.model(x)
        pred = torch.sigmoid(output["prediction"])
        spec = output["spectrogram"]

        # removing extra time step occurs in either label or prediction
        max_timesteps = min(pred.size(1), y.size(1))
        y = y[:, :max_timesteps]
        pred = pred[:, :max_timesteps]    
        l1_loss = torch.norm(pred, 1, 2).mean()
        bce_loss = F.binary_cross_entropy(pred, y)
        loss = bce_loss#+l1_loss
        metrics = {"Valid/L1": l1_loss,
                   "Valid/BCE": bce_loss}
#                        "Valid/L1": l1_loss}

        y = y.cpu().detach()
        pred = pred.cpu().detach()
        pred_roll = torch.zeros_like(y)

        pred_dict = {pred}
        for idx, (i, j) in enumerate(zip(pred, y)):
            pred_roll[idx] = transcription_accuracy(i, i,
                                                    j, j,
                                                    metrics, self.hop_length, self.sr, self.min_midi)

        if batch_idx==0:
            self.log_images(pred, f'Valid/posteriorgram')
            self.log_images(pred_roll, f'Valid/pred_roll')                
            if self.current_epoch==0:
                self.log_images(spec, f'Valid/spectrogram')                    
                self.log_images(y, f'Valid/ground_truth_roll')                    

        self.log_dict(metrics)


    def test_step(self, batch, batch_idx):
        x = batch['audio']
        y = batch['frame']
        metrics = {}

        pred = self(x)
        max_timesteps = pred.size(1)
        y = y[:,:max_timesteps]
        loss = F.binary_cross_entropy(pred, y)
        metrics["Test/frame"] = loss.item()

        pred = pred.cpu().detach()[0]
        y = y.cpu().detach()[0]

        self.transcription_accuracy(pred, y, metrics)
        self.log_dict(metrics)            


    def log_images(self, tensor, key, num_display=4):
        for idx, spec in enumerate(tensor.squeeze(1)):
            if num_display < idx:
                break
            
            fig, ax = plt.subplots(1,1)
            ax.imshow(spec.cpu().detach().t(), aspect='auto', origin='lower')    
            self.logger.experiment.add_figure(f"{key}/{idx}", fig, global_step=self.current_epoch)



    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         scheduler = TriStageLRSchedule(optimizer,
#                                        [1e-8, self.lr, 1e-8],
#                                        [0.2,0.6,0.2],
#                                        max_update=len(self.train_dataloader.dataloader)*self.trainer.max_epochs)   
#         scheduler = MultiStepLR(optimizer, [1,3,5,7,9], gamma=0.1, last_epoch=-1, verbose=False)

#         return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        return [optimizer]