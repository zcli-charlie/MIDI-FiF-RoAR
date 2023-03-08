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
        self.mask_lm = MLM(self.midi_former.e2w, self.midi_former.n_tokens, self.midi_former.hidden_size)
        self.casual_lm = CLM(self.midi_former.e2w, self.midi_former.n_tokens, self.midi_former.hidden_size)

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
    def __init__(self, e2w, n_tokens, hidden_size):
        super().__init__()
        
        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, etype in enumerate(e2w):
            self.proj.append(nn.Linear(hidden_size, n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)

        self.e2w = e2w
    
    def forward(self, y):
        # feed to former
        y = y.hidden_states[-1]
        
        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y[:,1:,:]))           # (batch_size, seq_len, dict_size)
        return ys


class CLM(nn.Module):
    def __init__(self, e2w, n_tokens, hidden_size):
        super().__init__()

        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, etype in enumerate(e2w):
            self.proj.append(nn.Linear(hidden_size, n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)

        self.e2w = e2w

    def forward(self, y):
        # feed to former
        y = y.hidden_states[-1]

        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y[:, :-1, :]))  # (batch_size, seq_len, dict_size)
        return ys