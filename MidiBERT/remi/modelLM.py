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
        self.mask_lm = MLM(self.midi_bert.e2w, self.midi_bert.emb_size, self.midi_bert.hidden_size)

    def forward(self, x, attn):
        x = self.midi_bert(x, attn)
        return self.mask_lm(x)
    

class MLM(nn.Module):
    def __init__(self, e2w, emb_size, hidden_size):
        super().__init__()
        
        # proj: project embeddings to logits for prediction
        self.proj = nn.Linear(hidden_size, len(e2w))

        self.e2w = e2w
    
    def forward(self, y):
        # feed to bert 
        y = y.hidden_states[-1]
        
        # convert embeddings back to logits for prediction
        y = self.proj(y)
        return y
    
