import math
import numpy as np
import random

import torch
import torch.nn as nn
from former import FormerModel

from model import MidiFormer


class MidiFormerLM(nn.Module):
    def __init__(self, midi_former: MidiFormer):
        super().__init__()
        
        self.midi_former = midi_former
        self.mask_lm = MLM(self.midi_former.e2w, self.midi_former.emb_size, self.midi_former.hidden_size)
        self.casual_lm = CLM(self.midi_former.e2w, self.midi_former.emb_size, self.midi_former.hidden_size)

    def forward(self, x, attn, mode="mlm"):
        if mode == "mlm":
            self.midi_former.formerConfig.is_decoder = False
        else:
            self.midi_former.formerConfig.is_decoder = True
        x = self.midi_former(x, attn, mode=mode)
        if mode == "mlm":
            return self.mask_lm(x)
        else:
            return self.casual_lm(x)
    

class MLM(nn.Module):
    def __init__(self, e2w, emb_size, hidden_size):
        super().__init__()
        
        # proj: project embeddings to logits for prediction
        self.proj = nn.Linear(hidden_size, len(e2w))

        self.e2w = e2w
    
    def forward(self, y):
        # feed to former
        y = y.hidden_states[-1]
        
        # convert embeddings back to logits for prediction
        y = self.proj(y[:,1:,:])
        return y
    

class CLM(nn.Module):
    def __init__(self, e2w, emb_size, hidden_size):
        super().__init__()
        
        # proj: project embeddings to logits for prediction
        self.proj = nn.Linear(hidden_size, len(e2w))

        self.e2w = e2w
    
    def forward(self, y):
        # feed to former
        y = y.hidden_states[-1]

        # convert embeddings back to logits for prediction
        y = self.proj(y[:, :-1, :])           
        return y