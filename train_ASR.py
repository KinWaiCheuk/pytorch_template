# Useful github libries
from nnAudio import Spectrogram
from AudioLoader.Speech import TIMIT

# Libraries related to PyTorch
import torch
from torch.utils.data import DataLoader, random_split

# Libraries related to PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# custom packages
from models.Tasks import ASR
import models.Models as Model
from utils.text_processing import TextTransform, data_processing

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

# For loading the output class ddictionary
import pickle

@hydra.main(config_path="config/ASR", config_name="experiment")
def main(cfg):
    # Allow users to specify other config files
    # python train_ASR.py user_config=config/xx.yaml
    if cfg.user_config is not None:
        print(f"{to_absolute_path('config')=}")
        user_config = OmegaConf.load(to_absolute_path(cfg.user_config))
        cfg = OmegaConf.merge(cfg, user_config)    
    
    # Loading dataset
    train_dataset = TIMIT(**cfg.dataset.train)
    test_dataset = TIMIT(**cfg.dataset.test)
    train_dataset, valid_dataset = random_split(train_dataset, [4000, 620], generator=torch.Generator().manual_seed(0))
    # Creating a spectrogram layer for dataloading
    print(OmegaConf.to_yaml(cfg)) # Printing out the config file, for debugging         
    SpecLayer = getattr(Spectrogram, cfg.spec_layer.type)
    spec_layer = SpecLayer(**cfg.spec_layer.args)
    
    # Auto inferring output mode and output dimension
    if cfg.output_mode == 'char':
        dict_file = 'characters_dict'
        cfg.data_processing.label_key = 'words'
    elif cfg.output_mode == 'ph':
        dict_file = 'phonemics_dict'
        cfg.data_processing.label_key = 'phonemics'        
    elif cfg.output_mode == 'word':
        dict_file = 'words_dict'
        cfg.data_processing.label_key = 'words'        
    else:
        raise ValueError(f'cfg.output_mode={cfg.output_mode} is not supported')
        
    with open(to_absolute_path(dict_file), 'rb') as f:
        output_dict = pickle.load(f)
    cfg.model.args.output_dim = len(output_dict) # number of classes equals to number of entries in the dict
           
    # Auto inferring input dimension 
    if cfg.spec_layer.type=='STFT':
        cfg.model.args.input_dim = cfg.spec_layer.args.n_fft//2+1
    elif cfg.spec_layer.type=='MelSpectrogram':
        cfg.model.args.input_dim = cfg.spec_layer.args.n_mels
    


    text_transform = TextTransform(output_dict, cfg.output_mode) # for text to int conversion layer

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
    test_loader = DataLoader(test_dataset,
                             **cfg.dataloader.test,
                             collate_fn=lambda x: data_processing(x,
                                                                  text_transform,
                                                                  **cfg.data_processing))      

    model = ASR(getattr(Model, cfg.model.type)(spec_layer, **cfg.model.args),
                text_transform,
                **cfg.pl)
    checkpoint_callback = ModelCheckpoint(monitor="valid_ctc_loss",
                                          filename="{epoch:02d}-{valid_ctc_loss:.2f}-{PER:.2f}",
                                          save_top_k=3,
                                          mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=".", version=1, name=f'ASR-{cfg.spec_layer.type}-{cfg.model.type}')
    trainer = pl.Trainer(gpus=cfg.gpus,
                         max_epochs=cfg.epochs,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=logger,
                         check_val_every_n_epoch=1)


    trainer.fit(model, train_loader, valid_loader)
#     trainer.test(model, test_loader)
    trainer.test(model, test_loader, ckpt_path="best")    
    
if __name__ == "__main__":
    main()    