# Useful github libries
from nnAudio import Spectrogram
from AudioLoader.Music import MAPS

# Libraries related to PyTorch
import torch
from torch.utils.data import DataLoader, random_split

# Libraries related to PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# custom packages
from models.Tasks import AMT
import models.Models as Model
from utils.text_processing import TextTransform, data_processing

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

# For loading the output class ddictionary
import pickle


@hydra.main(config_path="config/AMT", config_name="experiment")
def my_app(cfg):
    # Allow users to specify other config files
    # python train_ASR.py user_config=config/xx.yaml
    if cfg.user_config is not None:
        user_config = OmegaConf.load(to_absolute_path(cfg.user_config))
        cfg = OmegaConf.merge(cfg, user_config)
        
    # Loading dataset
    train_dataset = MAPS(**cfg.dataset.train)
    test_dataset = MAPS(**cfg.dataset.test)
    train_dataset, valid_dataset = random_split(train_dataset, [190, 20], generator=torch.Generator().manual_seed(0))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, **cfg.dataloader.train)
    valid_loader = DataLoader(valid_dataset, **cfg.dataloader.valid)
    test_dataset = DataLoader(test_dataset, **cfg.dataloader.test)    
    SpecLayer = getattr(Spectrogram, cfg.spec_layer.type)
    spec_layer = SpecLayer(**cfg.spec_layer.args)
    # Auto inferring input dimension 
    if cfg.spec_layer.type=='STFT':
        cfg.model.args.input_dim = cfg.spec_layer.args.n_fft//2+1
    elif cfg.spec_layer.type=='MelSpectrogram':
        cfg.model.args.input_dim = cfg.spec_layer.args.n_mels
    model = AMT(getattr(Model, cfg.model.type)(spec_layer, **cfg.model.args),
                **cfg.pl)
    checkpoint_callback = ModelCheckpoint(monitor="Train/BCE",
                                          filename="{epoch:02d}-{Train/BCE:.2f}",
                                          save_top_k=3,
                                          mode="min",
                                          auto_insert_metric_name=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=".", version=1, name=f'AMT-{cfg.spec_layer.type}-{cfg.model.type}')
    trainer = pl.Trainer(gpus=cfg.gpus,
                         max_epochs=cfg.epochs,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=logger,
                         check_val_every_n_epoch=20,
                         default_root_dir="./results")
    trainer.fit(model, train_loader, valid_loader)
    
if __name__ == "__main__":
    my_app()    