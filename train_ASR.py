# Useful github libries
from nnAudio import Spectrogram
from AudioLoader.speech import TIMIT

# Libraries related to PyTorch
import torch
from torch.utils.data import DataLoader, random_split

# Libraries related to PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# custom packages
from tasks.asr import ASR
import models.asr_models as Model
from utils.text_processing import TextTransform, data_processing

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

# For loading the output class ddictionary
import pickle

@hydra.main(config_path="config/asr", config_name="experiment")
def main(cfg):
    # converting paths
    cfg.data_root = to_absolute_path(cfg.data_root)
    if cfg.trainer.resume_from_checkpoint:
        cfg.trainer.resume_from_checkpoint = to_absolute_path(cfg.trainer.resume_from_checkpoint)
    
    
    # Loading dataset
    train_dataset = TIMIT(**cfg.dataset.train)
    test_dataset = TIMIT(**cfg.dataset.test)
    train_dataset, valid_dataset = random_split(train_dataset, [4000, 620], generator=torch.Generator().manual_seed(0))
    # Creating a spectrogram layer for dataloading       
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
    


    text_transform = TextTransform(output_dict, cfg.output_mode) # text to int conversion

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

    model = getattr(Model, cfg.model.type)(spec_layer, 
                                           **cfg.model.args, 
                                           text_transform = text_transform, 
                                           lr=cfg.pl.lr
                                           )    
    checkpoint_callback = ModelCheckpoint(**cfg.model_checkpoint)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=".", version=1, name=f'ASR-{cfg.spec_layer.type}-{cfg.model.type}')
    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=logger,)


    trainer.fit(model, train_loader, valid_loader)
#     trainer.test(model, test_loader)
    trainer.test(model, test_loader, ckpt_path="best")    
    
if __name__ == "__main__":
    main()    
