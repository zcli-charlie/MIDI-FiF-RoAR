import math
import numpy as np
import random

import torch
import torch.nn as nn
from former import FormerModel


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# Former model: similar approach to "felix"
class MidiFormer(nn.Module):
    def __init__(self, formerConfig, e2w, w2e):
        super().__init__()

        self.former = FormerModel(formerConfig)
        formerConfig.d_model = formerConfig.hidden_size
        self.hidden_size = formerConfig.hidden_size
        self.formerConfig = formerConfig

        # token types: [Bar, Position, Pitch, Duration]
        self.n_token = len(e2w)
        self.emb_size = 256
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.pad_word = self.e2w['Pad_None']        
        self.mask_word = self.e2w['Mask_None']
        
        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = Embeddings(self.n_token, self.emb_size) 

        # linear layer to merge embeddings from different token types 
        self.in_linear = nn.Linear(self.emb_size, formerConfig.d_model)


    def forward(self, input_id, attn_mask=None, output_hidden_states=True, mode="mlm"):
        bs, slen = input_id.shape
        if mode == "mlm":
            special_mark = torch.zeros((bs, 1)).long().to(input_id.device)
        else:
            special_mark = torch.zeros((bs, 1)).long().to(input_id.device) + 1
        special_emb = self.former.embeddings.word_embeddings(special_mark)

        # convert input_ids into embeddings and merge them through linear layer
        emb = self.word_emb(input_id)
        emb_linear = self.in_linear(emb)
        emb_linear = torch.cat([special_emb, emb_linear], dim=1)
        attn_mask = torch.cat([torch.ones((bs, 1)).to(input_id.device), attn_mask], dim=1)
        
        # feed to former 
        y = self.former(inputs_embeds=emb_linear, attention_mask=attn_mask, 
                             output_hidden_states=output_hidden_states)
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)
        return y
