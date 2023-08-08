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
        self.save_hyperparameters(ignore=['spec_layer', 'task_kargs'])
        
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
                 task_kargs=None):
        super().__init__(**task_kargs)
        self.save_hyperparameters(ignore=['spec_layer', 'task_kargs'])
        
        self.spec_layer = spec_layer

        attn_embed_dim = 256
        attn_num_heads = 8

        self.spec_proj = nn.Linear(input_dim, attn_embed_dim)
        self.pos_encoder = PositionalEncoder(attn_embed_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=attn_embed_dim,
            num_heads=attn_num_heads,
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
        
    def forward(self, x):
        spec = self.spec_layer(x) # (B, F, T)
        spec = torch.log(spec+1e-8)
        spec = spec.transpose(1,2) # (B, T, F)

        spec = self.norm_layer(spec)

        # self-attention
        spec = self.spec_proj(spec) # project spec into attn_embed_dim
        spec = self.pos_encoder(spec) # add positional encoding
        attn_output, attn_output_weights = self.multihead_attn(spec, spec, spec)

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