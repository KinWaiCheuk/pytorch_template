# Useful github libries
from nnAudio import Spectrogram
# from AudioLoader.Speech import TIMIT

# Libraries related to PyTorch
import torch
from torch.utils.data import DataLoader, random_split
import torchaudio
# Libraries related to PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# custom packages
from models.Tasks import Speech_cmd_task
import models.Models as Model
from utils.text_processing import speech_command_processing, Speech_Command_label_Transform

# Libraries related to hydra
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

# For loading the output class ddictionary
import pickle

@hydra.main(config_path="config/speechcommand", config_name="experiment")
def main(cfg):
    # Allow users to specify other config files
    # python train_ASR.py user_config=config/xx.yaml
    # if cfg.user_config is not None:
    #     print(f"{to_absolute_path('config')=}")
    #     user_config = OmegaConf.load(to_absolute_path(cfg.user_config))
    #     cfg = OmegaConf.merge(cfg, user_config)    
    
    # Loading dataset
    # train_dataset = SubsetSpeechCommand(**cfg.dataset.train)
    # test_dataset = SubsetSpeechCommand(**cfg.dataset.test) #change here!
    # valid_dataset = SubsetSpeechCommand(**cfg.dataset.val)
    
    train_dataset = torchaudio.datasets.SPEECHCOMMANDS(**cfg.dataset.train)
    test_dataset = torchaudio.datasets.SPEECHCOMMANDS(**cfg.dataset.test) #change here!
    valid_dataset = torchaudio.datasets.SPEECHCOMMANDS(**cfg.dataset.val)
    
    # Creating a spectrogram layer for dataloading
    print(OmegaConf.to_yaml(cfg)) # Printing out the config file, for debugging         
    SpecLayer = getattr(Spectrogram, cfg.spec_layer.type)
    spec_layer = SpecLayer(**cfg.spec_layer.args)
    
    speech_command_transform = Speech_Command_label_Transform(train_dataset)



    # text_transform = TextTransform(output_dict, cfg.output_mode) # for text to int conversion layer
    train_loader = DataLoader(train_dataset,
                              **cfg.dataloader.train,
                              collate_fn=lambda x: speech_command_processing(x,
                                                                   speech_command_transform ,
                                                                   **cfg.data_processing))
    

    valid_loader = DataLoader(valid_dataset,
                              **cfg.dataloader.valid,
                              collate_fn=lambda x: speech_command_processing(x,
                                                                   speech_command_transform,
                                                                   **cfg.data_processing))
    test_loader = DataLoader(test_dataset,
                             **cfg.dataloader.test,
                             collate_fn=lambda x: speech_command_processing(x,
                                                                  speech_command_transform,
                                                                  **cfg.data_processing)) 

    cfg.model.args.output_dim = len(speech_command_transform.labels) # number of classes equals to number of entries in the dict
           
    # Auto inferring input dimension 
    if cfg.spec_layer.type=='STFT':
        # cfg.model.args.input_dim = cfg.spec_layer.args.n_fft//2+1
        cfg.model.args.input_dim =  [x for x in train_loader][0]["waveforms"].shape//spec_layer.stride *(cfg.spec_layer.args.n_fft//2+1)   #get from args!/ create 
    elif cfg.spec_layer.type=='MelSpectrogram':
        cfg.model.args.input_dim = cfg.spec_layer.args.n_mels *101 #change here the dim
        # cfg.model.args.input_dim =train_loader[0]["waveforms"].shape     

    model = Speech_cmd_task(getattr(Model, cfg.model.type)(spec_layer, **cfg.model.args), 
                **cfg.pl)
    checkpoint_callback = ModelCheckpoint(monitor="val_CE_loss", #change here! change the name
                                          filename="{epoch:02d}-{val_CE_loss:.2f}",  
                                          save_top_k=3,
                                          mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=".", version=1, name=f'SC-{cfg.spec_layer.type}-{cfg.model.type}')
    trainer = pl.Trainer(gpus=cfg.gpus,
                         max_epochs=cfg.epochs,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=logger,
                         check_val_every_n_epoch=1)


    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader, ckpt_path="best")    
    
if __name__ == "__main__":
    main()    