import pytorch_lightning as pl
import torch
import torch.nn as nn

class SpeechCommand(pl.LightningModule):
    def __init__(self,
                 lr):
        super().__init__()
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x = batch['waveforms']
        output = self(x)
        loss = self.loss_fn(output["prediction"],batch['labels'].long())      
        self.log("Train/Loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['waveforms']
        with torch.no_grad():
            output = self(x)
            loss = self.loss_fn(output["prediction"],batch['labels'].long())      
            self.log("val_CE_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['waveforms']
        with torch.no_grad():
            output = self(x)
            loss = self.loss_fn(output["prediction"],batch['labels'].long())      
            self.log("test_CE_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]