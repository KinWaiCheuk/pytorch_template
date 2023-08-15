import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import TriStageLRSchedule
from utils import Evaluator

# from nnAudio.Spectrogram import MelSpectrogram
import pandas as pd




class AMT(pl.LightningModule):
    def __init__(self,
                 lr,
                 sr,
                 hop_length,
                 min_midi
                ):
        super().__init__()
        self.evaluator = Evaluator(
            hop_length,
            sr,
            min_midi,
            onset_threshold=0.5,
            frame_threshold=0.5
            )
        self.lr = lr
        
    def step(self, batch):
        # when self.hparams.onset==False
        # output["onset"] is the same as output["frame"]

        x = batch['audio']
        y_frame = batch['frame']
        y_onset = batch['onset']
        output = self(x)
        pred_frame = torch.sigmoid(output["frame"])

        
        # removing extra time step occurs in either label or prediction
        max_timesteps = min(pred_frame.size(1), y_frame.size(1))
        y_frame = y_frame[:, :max_timesteps]
        pred_frame = pred_frame[:, :max_timesteps]


        # updateing output dictionary
        output["frame"] = pred_frame

        # if onset is used, do the same for onset
        if self.hparams.onset:
            pred_onset = torch.sigmoid(output["onset"])
            pred_onset = pred_onset[:, :max_timesteps]
            y_onset = y_onset[:, :max_timesteps]
            output["onset"] = pred_onset
        else:
            output["onset"] = pred_frame

        return output
    
    def compute_loss(self, output, batch):
        if self.hparams.onset:
            bce_loss_onset = F.binary_cross_entropy(output["onset"], batch['onset'])
        else: 
            bce_loss_onset = 0

        bce_loss_frame = F.binary_cross_entropy(output["frame"], batch['frame'])   

        losses = {
            "bce_loss_frame": bce_loss_frame,
            "bce_loss_onset": bce_loss_onset
        }

        return losses


    def training_step(self, batch, batch_idx):
        output = self.step(batch)
        # the frame and onset outputs are after sigmoid
        # i.e. data range is [0,1]      
        # when self.hparams.onset==False
        # output["onset"] is the same as output["frame"]   

        losses = self.compute_loss(output, batch)
        

        loss = losses['bce_loss_frame'] + losses['bce_loss_onset']

        self.log("Train/BCE_frame", losses['bce_loss_frame'])
        self.log("Train/BCE_onset", losses['bce_loss_onset'])
        self.log("Train/total_loss", loss) 
        
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)

        losses = self.compute_loss(output, batch)

        self.log("Valid/BCE_frame", losses['bce_loss_frame'])
        self.log("Valid/BCE_onset", losses['bce_loss_onset'])


        # looping over samples in batch
        for sample_idx in range(batch["frame"].size(0)):
            transcription_metrics = self.evaluator.evaluate(
                output["frame"][sample_idx],
                output["onset"][sample_idx],
                batch["frame"][sample_idx],
                batch["onset"][sample_idx])
            for key_frame, value in transcription_metrics.items():
                self.log(f"Valid/{key_frame}", value)

        # for idx, (i, j) in enumerate(zip(pred, y_frame)):
        #     pred_roll[idx] = transcription_accuracy_frame(i, i,
        #                                             j, j,
        #                                             metrics, self.hop_length, self.sr, self.min_midi)

        if batch_idx==0:
            # plot only two samples
            samples = 2
            self.log_images(output["frame"][:samples], f'Valid/frame_pred')
            self.log_images(output["onset"][:samples], f'Valid/onset_pred')                
            if self.current_epoch==0:
                self.log_images(output["spec"][:samples], f'Valid/spectrogram')
                self.log_images(batch["frame"][:samples], f'Valid/frame_gt')
                self.log_images(batch["onset"][:samples], f'Valid/onset_gt')           


    def test_step(self, batch, batch_idx):
        x = batch['audio']
        y_frame = batch['frame']
        metrics = {}

        pred = self(x)
        max_timesteps = pred.size(1)
        y_frame = y_frame[:,:max_timesteps]
        loss = F.binary_cross_entropy(pred, y_frame)
        metrics["Test/frame"] = loss.item()

        pred = pred.cpu().detach()[0]
        y_frame = y_frame.cpu().detach()[0]

        self.transcription_accuracy_frame(pred, y_frame, metrics)
        self.log_dict(metrics)            


    def log_images(self, tensor, key_frame, num_display_frame=4):
        for idx, spec in enumerate(tensor.squeeze(1)):
            if num_display_frame < idx:
                break
            
            fig, ax = plt.subplots(1,1)
            ax.imshow(spec.cpu().detach().t(), aspect='auto', origin='lower')    
            self.logger.experiment.add_figure(f"{key_frame}/{idx}", fig, global_step=self.current_epoch)



    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         scheduler = TriStageLRSchedule(optimizer,
#                                        [1e-8, self.lr, 1e-8],
#                                        [0.2,0.6,0.2],
#                                        max_update=len(self.train_dataloader.dataloader)*self.trainer.max_epochs)   
#         scheduler = MultiStepLR(optimizer, [1,3,5,7,9], gamma=0.1, last_epoch=-1, verbose=False)

#         return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        return [optimizer]