import math
import numpy as np
import random

import torch
import torch.nn as nn
from transformers import BertModel

from model import MidiBert


class MidiBertLM(nn.Module):
    def __init__(self, midi_bert: MidiBert):
        super().__init__()
        
        self.midi_bert = midi_bert
        self.mask_lm = MLM(self.midi_bert.e2w, self.midi_bert.n_tokens, self.midi_bert.hidden_size)

    def forward(self, x, attn):
        x = self.midi_bert(x, attn)
        return self.mask_lm(x)
    

class MLM(nn.Module):
    def __init__(self, e2w, n_tokens, hidden_size):
        super().__init__()
        
        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, etype in enumerate(e2w):
            self.proj.append(nn.Linear(hidden_size, n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)

        self.e2w = e2w
    
    def forward(self, y):
        # feed to bert 
        y = y.hidden_states[-1]
        
        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))           # (batch_size, seq_len, dict_size)
        return ys

