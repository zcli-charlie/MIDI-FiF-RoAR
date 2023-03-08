import math
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MidiFormer


class TokenClassification(nn.Module):
    def __init__(self, midi_former, class_num, hs, mode="mlm"):
        super().__init__()
        
        self.midi_former = midi_former
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hs, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

        self.mode = mode
    
    def forward(self, y, attn, layer):
        # feed to former
        y = self.midi_former(y, attn, output_hidden_states=True, mode=self.mode)
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)
        y = y.hidden_states[layer]
        y = y[:,1:,:]
        return self.classifier(y)


class SequenceClassification(nn.Module):
    def __init__(self, midi_former, class_num, hs, da=128, r=4, mode="mlm"):
        super(SequenceClassification, self).__init__()
        self.midi_former = midi_former
        self.attention = SelfAttention(hs, da, r)
        self.classifier = nn.Sequential(
            nn.Linear(hs*r, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

        self.mode = mode

    def forward(self, x, attn, layer):             # x: (batch, 512, 4)
        x = self.midi_former(x, attn, output_hidden_states=True)   # (batch, 512, 768)
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)
        x = x.hidden_states[layer]
        x = x[:, 1:, :]
        attn_mat = self.attention(x)        # attn_mat: (batch, r, 512)
        m = torch.bmm(attn_mat, x)          # m: (batch, r, 768)
        flatten = m.view(m.size()[0], -1)   # flatten: (batch, r*768)
        res = self.classifier(flatten)      # res: (batch, class_num)
        return res


class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        '''
        Args:
            input_dim (int): batch, seq, input_dim
            da (int): number of features in hidden layer from self-attn
            r (int): number of aspects of self-attn
        '''
        super(SelfAttention, self).__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0,2,1)
        return attn_mat
