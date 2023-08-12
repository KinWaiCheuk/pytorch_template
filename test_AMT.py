# Useful github libries
from nnAudio import Spectrogram
from AudioLoader.music.amt import MAPS

# Libraries related to PyTorch
import torch
from torch.utils.data import DataLoader, random_split

# Libraries related to PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# custom packages
from tasks.amt import AMT
import models.Models as Model
from utils.text_processing import TextTransform, data_processing

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

# For loading the output class ddictionary
import pickle


@hydra.main(config_path="config/amt", config_name="test")
def my_app(cfg):       
    cfg.data_root = to_absolute_path(cfg.data_root)
    # Loading dataset
    test_dataset = MAPS(**cfg.dataset.test)

    # Create dataloaders
    test_loader = DataLoader(test_dataset, **cfg.dataloader.test)    
    SpecLayer = getattr(Spectrogram, cfg.spec_layer.type)
    spec_layer = SpecLayer(**cfg.spec_layer.args)
    # Auto inferring input dimension 
    if cfg.spec_layer.type=='STFT':
        cfg.model.args.input_dim = cfg.spec_layer.args.n_fft//2+1
    elif cfg.spec_layer.type=='MelSpectrogram':
        cfg.model.args.input_dim = cfg.spec_layer.args.n_mels
    model = getattr(
        Model, 
        cfg.model.type
        ).load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path, 
            spec_layer=spec_layer, 
            task_kargs=cfg.pl
            )

    if cfg.mfm_path:
        assert cfg.model.args.mfm != False, "mfm_path is provided but mfm is not used"
        exp_name = f"Test_AMT-token_offset-{cfg.spec_layer.type}-{cfg.model.type}-mfm"
    else:
        exp_name = f"Test_AMT-token_offset-{cfg.spec_layer.type}-{cfg.model.type}"
    logger = TensorBoardLogger(save_dir=".", version=1, name=exp_name)
    trainer = pl.Trainer(gpus=cfg.gpus,
                         logger=logger)
    trainer.test(model, test_loader)
    
if __name__ == "__main__":
    my_app()    
