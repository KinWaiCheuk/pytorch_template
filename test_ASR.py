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
import models.asr_models as Model
from utils.text_processing import TextTransform, data_processing

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

# For loading the output class ddictionary
import pickle

@hydra.main(config_path="config/asr", config_name="test")
def main(cfg):
    cfg.data_root = to_absolute_path(cfg.data_root)
    cfg.checkpoint_path = to_absolute_path(cfg.checkpoint_path)  
    
    # Loading dataset
    test_dataset = TIMIT(**cfg.dataset.test)
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

    test_loader = DataLoader(test_dataset,
                             **cfg.dataloader.test,
                             collate_fn=lambda x: data_processing(x,
                                                                  text_transform,
                                                                  **cfg.data_processing))      

    model = getattr(Model, cfg.model.type).load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path, 
            spec_layer=spec_layer
            )
    

    logger = TensorBoardLogger(save_dir=".", version=1, name=f'Test_ASR-{cfg.spec_layer.type}-{cfg.model.type}')
    trainer = pl.Trainer(**cfg.trainer,
                         logger=logger,)


    trainer.test(model, test_loader)    
    
if __name__ == "__main__":
    main()    
