from torch.utils.data import DataLoader
# from torchaudio.transforms import MelSpectrogram
from nnAudio.Spectrogram import MelSpectrogram
# import sys
# sys.path.insert(0, '../AudioLoader/')
from AudioLoader.Speech import TIMIT
from IPython.display import Audio
import matplotlib.pyplot as plt
import tqdm
from datetime import datetime

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import os

from omegaconf import DictConfig, OmegaConf
import hydra

from models.ASR import SimpleASR
from utils.text_processing import TextTransform, wav2vec_processing
import pickle

from hydra.utils import get_original_cwd, to_absolute_path


@hydra.main(config_path="config", config_name="experiment_asr")
def main(cfg):
    # Allow users to specify other config files
    if cfg.user_config is not None:
        print(f"{to_absolute_path('config')=}")
        user_config = OmegaConf.load(to_absolute_path(cfg.user_config))
        config = OmegaConf.merge(cfg, user_config)    
    
    train_dataset = TIMIT(**cfg.dataset.train)
    valid_dataset = TIMIT(**cfg.dataset.valid)
    
    spec_layer = MelSpectrogram(**cfg.spec_layer)
    
    # Text preprocessing
    with open(to_absolute_path('phonemics_dict'), 'rb') as f:
        phonemics_dict = pickle.load(f)
    text_transform = TextTransform(phonemics_dict)
    data_processing = wav2vec_processing

    train_loader = DataLoader(train_dataset,
                              **cfg.dataloader.train,
                              collate_fn=lambda x: data_processing(x,
                                                                   text_transform,
                                                                   **cfg.data_processing))
    valid_loader = DataLoader(valid_dataset,
                              **cfg.dataloader.valid,
                              collate_fn=lambda x: data_processing(x,
                                                                   text_transform,
                                                                   **cfg.data_processing))                        
    
    model = SimpleASR(spec_layer, text_transform, **cfg.model)
    checkpoint_callback = ModelCheckpoint(monitor="train_ctc_loss",
                                          filename="{epoch:02d}-{val_loss:.2f}",
                                          save_top_k=3,
                                          mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=".", version=1, name='ASR_results_no_eval')
    trainer = pl.Trainer(gpus=cfg.gpus,
                         max_epochs=cfg.epochs,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=logger,
                         check_val_every_n_epoch=5)


    trainer.fit(model, train_loader, valid_loader)
    
if __name__ == "__main__":
    main()    