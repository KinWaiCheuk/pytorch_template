import torch
import torch.nn as nn
from models.utils import Normalization
from tasks.amt import AMT
import math


# from nnAudio.Spectrogram import MelSpectrogram
import pandas as pd

class simpleLSTM(nn.Module):
    def __init__(self,
                 spec_layer,
                 input_dim,
                 hidden_dim,
                 num_lstms,
                 output_dim,
                 ):
        super().__init__()
        
        self.spec_layer = spec_layer
        self.embedding = nn.Linear(input_dim,hidden_dim)
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=num_lstms, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        spec = self.spec_layer(x) # (B, F, T)
        spec = torch.log(spec+1e-8)
        spec = spec.transpose(1,2) # (B, T, F)
        x = self.embedding(spec)
        x, _ = self.bilstm(x)
        pred = self.classifier(x)

        output = {"prediction": pred,
                  "spectrogram": spec}
        return output
    
    
class simpleLinear(nn.Module):
    def __init__(self,
                 spec_layer,
                 input_dim,
                 hidden_dim1,
                 hidden_dim2,
                 hidden_dim3,                 
                 output_dim,
                 ):
        super().__init__()
        
        self.spec_layer = spec_layer
        self.linear1 = nn.Linear(input_dim,hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1,hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2,hidden_dim3)        
        self.classifier = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        spec = self.spec_layer(x) # (B, F, T)
        spec = torch.log(spec+1e-8)
        spec = spec.transpose(1,2) # (B, T, F)
        x = torch.relu(self.linear1(spec))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))        
        pred = self.classifier(x)
        
        output = {"prediction": pred,
                  "spectrogram": spec}
        return output
    
class simpleLinear_flatten(nn.Module):
    def __init__(self,
                 spec_layer,
                 input_dim,
                 hidden_dim1,
                 hidden_dim2,
                 hidden_dim3,                 
                 output_dim,
                 ):
        super().__init__()
        
        self.spec_layer = spec_layer
        self.linear1 = nn.Linear(input_dim,hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1,hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2,hidden_dim3)        
        self.classifier = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        spec = self.spec_layer(x) # (B, F, T) # 32, 80, 101
        spec = torch.log(spec+1e-8)
        spec = torch.reshape(spec, (spec.shape[0],-1 ))

        x = torch.relu(self.linear1(spec))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))        
        pred = self.classifier(x)
        
        output = {"prediction": pred,
                  "spectrogram": spec}
        return output

class CNN_LSTM(AMT):
    def __init__(self,
                 spec_layer,
                 norm_mode,
                 input_dim,
                 onset=False,
                 hidden_dim=768,
                 output_dim=88,
                 task_kargs=None):
        
        super().__init__(**task_kargs)
        self.save_hyperparameters(ignore=['spec_layer'])

        self.spec_layer = spec_layer
        self.norm_layer = Normalization(mode=norm_mode)
        self.onset = onset
        
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, hidden_dim // 16, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(hidden_dim // 16, hidden_dim // 16, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(hidden_dim // 16, hidden_dim // 8, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((hidden_dim // 8) * (input_dim // 4), hidden_dim),
            nn.Dropout(0.5)
        )
        
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim//2, batch_first=True, num_layers=1, bidirectional=True)
        
        self.frame_classifier = nn.Linear(hidden_dim, output_dim)
        if self.onset:
            self.onset_classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        spec = self.spec_layer(x) # (B, F, T)
        spec = torch.log(spec+1e-8)
        spec = spec.transpose(1,2) # (B, T, F)
        spec = self.norm_layer(spec)
        spec = spec.unsqueeze(1) # (B, 1, T, F)

        x = self.cnn(spec) # (B, hidden_dim//8, T, F//4)
        x = x.transpose(1,2).flatten(2)
        x = self.fc(x) # (B, T, hidden_dim//8*F//4)
        x, _ = self.bilstm(x)
        
        pred_frame = self.frame_classifier(x)
        if self.onset:
            pred_onset = self.onset_classifier(x)

            output = {"frame": pred_frame,
                      "onset": pred_onset,
                      "spec": spec}
        else:
            output = {"frame": pred_frame,
                      "spec": spec}
            
        return output
        
class Attention_CNN(AMT):
    def __init__(self,
                 spec_layer,
                 norm_mode,
                 input_dim,
                 onset=False,
                 hidden_dim=768,
                 output_dim=88,
                 mfm=False,
                 task_kargs=None):
        super().__init__(**task_kargs)
        self.save_hyperparameters(ignore=['spec_layer', 'task_kargs'])
        
        self.spec_layer = spec_layer
        self.mfm = mfm

        attn_embed_dim = 256
        attn_num_heads = 8

        self.spec_proj = nn.Linear(input_dim, attn_embed_dim)
        self.pos_encoder = PositionalEncoder(attn_embed_dim)


        if self.mfm:
            # the mfm tokens (top layer prior) have 3600 feature dimension
            # self.mfm_proj = nn.Linear(3600, attn_embed_dim)
            mfm_dim = 3600
            self.mfm_pos_encoder = PositionalEncoder(mfm_dim)            
            kdim = mfm_dim
            vdim = mfm_dim
        else:
            kdim = None
            vdim = None

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=attn_embed_dim,
            num_heads=attn_num_heads,
            kdim=kdim,
            vdim=vdim,
            batch_first=True
            )        
        self.norm_layer = Normalization(mode=norm_mode)
        self.onset = onset
        
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, hidden_dim // 16, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(hidden_dim // 16, hidden_dim // 16, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(hidden_dim // 16, hidden_dim // 8, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((hidden_dim // 8) * (attn_embed_dim // 4), hidden_dim),
            nn.Dropout(0.5)
        )
        
        self.frame_classifier = nn.Linear(hidden_dim, output_dim)
        if self.onset:
            self.onset_classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, mfm_tokens=None):
        # if x2 is given, do cross attention
        spec = self.spec_layer(x) # (B, F, T)
        spec = torch.log(spec+1e-8)
        spec = spec.transpose(1,2) # (B, T, F)

        spec = self.norm_layer(spec)

        # self-attention
        spec_proj = self.spec_proj(spec) # project spec into attn_embed_dim
        spec_proj = self.pos_encoder(spec_proj) # add positional encoding

        if mfm_tokens != None:
            # sanity check to avoid bugs
            assert self.mfm, "mfm_tokens is given but mfm is not set to True"

            # mfm_tokens_proj = self.mfm_proj(mfm_tokens)
            #(B, T, F)

            attn_k = mfm_tokens
            attn_v = mfm_tokens
        else:
            attn_k = spec_proj
            attn_v = spec_proj
        attn_output, attn_output_weights = self.multihead_attn(
            spec_proj,
            attn_k,
            attn_v
            )

        attn_output = attn_output.unsqueeze(1) # (B, 1, T, F)???

        x = self.cnn(attn_output) # (B, hidden_dim//8, T, F//4)
        x = x.transpose(1,2).flatten(2)
        x = self.fc(x) # (B, T, hidden_dim//8*F//4)
        
        pred_frame = self.frame_classifier(x)
        if self.onset:
            pred_onset = self.onset_classifier(x)

            output = {"frame": pred_frame,
                      "onset": pred_onset,
                      "spec": spec}
        else:
            output = {"frame": pred_frame,
                      "spec": spec}
            
        return output
    

class AvgPool_CNN_Early(AMT):
    def __init__(self,
                 spec_layer,
                 norm_mode,
                 input_dim,
                 onset=False,
                 hidden_dim=768,
                 output_dim=88,
                 mfm=False,
                 task_kargs=None):
        super().__init__(**task_kargs)
        self.save_hyperparameters(ignore=['spec_layer'])
        print(f"In Model ========= {self.hparams=}") 
        
        # input_dim = 229
        # which is the spectrogram bins
        self.input_dim = input_dim
        self.spec_layer = spec_layer
        self.mfm = mfm

        if self.mfm:
            # 1frame = 8 token
            # self.mfm_proj = nn.Linear(3600, input_dim)
            self.avg_pool = torch.nn.AvgPool1d(8)            

        self.norm_layer = Normalization(mode=norm_mode)
        self.onset = onset

        self.feat_selector = nn.Linear(input_dim+3600, input_dim)
        
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, hidden_dim // 16, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(hidden_dim // 16, hidden_dim // 16, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(hidden_dim // 16, hidden_dim // 8, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((hidden_dim // 8) * (input_dim // 4), hidden_dim),
            nn.Dropout(0.5)
        )
        
        self.frame_classifier = nn.Linear(hidden_dim, output_dim)
        if self.onset:
            self.onset_classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, mfm_tokens=None):
        # if x2 is given, do cross attention
        spec = self.spec_layer(x) # (B, F, T)
        spec = torch.log(spec+1e-8)
        spec = spec.transpose(1,2) # (B, T, F)

        spec = self.norm_layer(spec)


        if mfm_tokens != None:
            # sanity check to avoid bugs
            assert self.mfm, "mfm_tokens is given but mfm is not set to True"
            # mfm_tokens = self.mfm_proj(mfm_tokens)
            # downsample mfm_tokens to be the same size as spec
            mfm_tokens = self.avg_pool(mfm_tokens.transpose(-1,-2)).transpose(-1,-2)
            #(B, T, F)

            # discard the last frame of spec
            # because we don't have enough mfm_tokens for the last frame
        else:
            # discard the last frame of spec
            mfm_tokens = torch.zeros(spec.shape[0], spec.shape[1]-1, 3600).to(spec.device)
        x = torch.cat([spec[:,:-1], mfm_tokens], dim=-1)            
        # combine spec and mfm_tokens

        x = self.feat_selector(x)
        x = self.cnn(x.unsqueeze(1)) # (B, hidden_dim//8, T, F//4)
        x = x.transpose(1,2).flatten(2)
        x = self.fc(x) # (B, T, hidden_dim//8*F//4)
        
        pred_frame = self.frame_classifier(x)
        if self.onset:
            pred_onset = self.onset_classifier(x)

            output = {"frame": pred_frame,
                      "onset": pred_onset,
                      "spec": spec}
        else:
            output = {"frame": pred_frame,
                      "spec": spec}
            
        return output
    

class AvgPool_CNN_Early_mfm_proj(AMT):
    def __init__(self,
                 spec_layer,
                 norm_mode,
                 input_dim,
                 onset=False,
                 hidden_dim=768,
                 output_dim=88,
                 mfm=False,
                 task_kargs=None):
        super().__init__(**task_kargs)
        self.save_hyperparameters(ignore=['spec_layer', 'task_kargs'])
        
        # input_dim = 229
        # which is the spectrogram bins
        self.input_dim = input_dim
        self.spec_layer = spec_layer
        self.mfm = mfm

        if self.mfm:
            # 1frame = 8 token
            self.mfm_proj = nn.Linear(3600, input_dim)
            self.avg_pool = torch.nn.AvgPool1d(8)            

        self.norm_layer = Normalization(mode=norm_mode)
        self.onset = onset

        self.feat_selector = nn.Linear(input_dim * 2, input_dim)
        
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, hidden_dim // 16, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(hidden_dim // 16, hidden_dim // 16, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(hidden_dim // 16, hidden_dim // 8, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((hidden_dim // 8) * (input_dim // 4), hidden_dim),
            nn.Dropout(0.5)
        )
        
        self.frame_classifier = nn.Linear(hidden_dim, output_dim)
        if self.onset:
            self.onset_classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, mfm_tokens=None):
        # if x2 is given, do cross attention
        spec = self.spec_layer(x) # (B, F, T)
        spec = torch.log(spec+1e-8)
        spec = spec.transpose(1,2) # (B, T, F)

        spec = self.norm_layer(spec)


        if mfm_tokens != None:
            # sanity check to avoid bugs
            assert self.mfm, "mfm_tokens is given but mfm is not set to True"
            mfm_tokens = self.mfm_proj(mfm_tokens)
            # downsample mfm_tokens to be the same size as spec
            mfm_tokens = self.avg_pool(mfm_tokens.transpose(-1,-2)).transpose(-1,-2)
            #(B, T, F)

            # discard the last frame of spec
            # because we don't have enough mfm_tokens for the last frame
        else:
            mfm_tokens = torch.zeros_like(spec[:,:-1])
        x = torch.cat([spec[:,:-1], mfm_tokens], dim=-1)         
        x = self.feat_selector(x)
        # combine spec and mfm_tokens

        x = self.cnn(x.unsqueeze(1)) # (B, hidden_dim//8, T, F//4)
        x = x.transpose(1,2).flatten(2)
        x = self.fc(x) # (B, T, hidden_dim//8*F//4)
        
        pred_frame = self.frame_classifier(x)
        if self.onset:
            pred_onset = self.onset_classifier(x)

            output = {"frame": pred_frame,
                      "onset": pred_onset,
                      "spec": spec}
        else:
            output = {"frame": pred_frame,
                      "spec": spec}
            
        return output
    

class AvgPool_CNN_Late(AMT):
    def __init__(self,
                 spec_layer,
                 norm_mode,
                 input_dim,
                 onset=False,
                 hidden_dim=768,
                 output_dim=88,
                 mfm=False,
                 task_kargs=None):
        super().__init__(**task_kargs)
        self.save_hyperparameters(ignore=['spec_layer', 'task_kargs'])
        
        # input_dim = 229
        # which is the spectrogram bins
        self.input_dim = input_dim
        self.spec_layer = spec_layer
        self.mfm = mfm

        if self.mfm:
            # 1frame = 8 token
            self.mfm_proj = nn.Linear(3600, input_dim)
            self.avg_pool = torch.nn.AvgPool1d(8)            

        self.norm_layer = Normalization(mode=norm_mode)
        self.onset = onset

        self.feat_selector = nn.Linear(input_dim*2, input_dim)
        
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, hidden_dim // 16, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(hidden_dim // 16, hidden_dim // 16, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(hidden_dim // 16, hidden_dim // 8, (3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((hidden_dim // 8) * (input_dim // 4), hidden_dim),
            nn.Dropout(0.5)
        )
        
        self.frame_classifier = nn.Linear(hidden_dim, output_dim)
        if self.onset:
            self.onset_classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, mfm_tokens=None):
        # if x2 is given, do cross attention
        spec = self.spec_layer(x) # (B, F, T)
        spec = torch.log(spec+1e-8)
        spec = spec.transpose(1,2) # (B, T, F)

        spec = self.norm_layer(spec)


        if mfm_tokens != None:
            # sanity check to avoid bugs
            assert self.mfm, "mfm_tokens is given but mfm is not set to True"
            mfm_tokens = self.mfm_proj(mfm_tokens)
            # downsample mfm_tokens to be the same size as spec
            mfm_tokens = self.avg_pool(mfm_tokens.transpose(-1,-2)).transpose(-1,-2)
            #(B, T, F)

            # discard the last frame of spec
            # because we don't have enough mfm_tokens for the last frame
        else:
            mfm_tokens = torch.zeros_like(spec[:,:-1])
        x = torch.cat([spec[:,:-1], mfm_tokens], dim=-1)            
        # combine spec and mfm_tokens
        
        x = self.feat_selector(x)
        x = self.cnn(x.unsqueeze(1)) # (B, hidden_dim//8, T, F//4)
        x = x.transpose(1,2).flatten(2)
        x = self.fc(x) # (B, T, hidden_dim//8*F//4)
        
        pred_frame = self.frame_classifier(x)
        if self.onset:
            pred_onset = self.onset_classifier(x)

            output = {"frame": pred_frame,
                      "onset": pred_onset,
                      "spec": spec}
        else:
            output = {"frame": pred_frame,
                      "spec": spec}
            
        return output
    

class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=641):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x