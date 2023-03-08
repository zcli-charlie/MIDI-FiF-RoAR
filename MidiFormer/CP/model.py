import math
import numpy as np
import random

import torch
import torch.nn as nn
from former import FormerModel
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.bert.configuration_bert import BertConfig

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# Former model: similar approach to "felix"
class MidiFormer(nn.Module):
    def __init__(self, formerConfig, e2w, w2e, use_fif):
        super().__init__()

        self.former = FormerModel(formerConfig)
        formerConfig.d_model = formerConfig.hidden_size
        self.hidden_size = formerConfig.hidden_size
        self.formerConfig = formerConfig

        # token types: [Bar, Position, Pitch, Duration]
        self.n_tokens = []  # [3,18,88,66]
        for key in e2w:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256, 256, 256, 256]
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']
        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.e2w], dtype=np.long)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.e2w], dtype=np.long)

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.e2w):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types 
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), formerConfig.d_model)

        attn_config = BertConfig(hidden_size=256, num_attention_heads=4, intermediate_size=512)
        self.interaction_attn = BertAttention(attn_config)

        self.use_fif = use_fif

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True, mode="mlm"):
        bs, slen, _ = input_ids.shape
        if mode == "mlm":
            special_mark = torch.zeros((bs, 1)).long().to(input_ids.device)
        else:
            special_mark = torch.zeros((bs, 1)).long().to(input_ids.device) + 1
        special_emb = self.former.embeddings.word_embeddings(special_mark)

        if not self.use_fif:
            # convert input_ids into embeddings and merge them through linear layer
            embs = []
            for i, key in enumerate(self.e2w):
                embs.append(self.word_emb[i](input_ids[..., i]))
            embs = torch.cat([*embs], dim=-1)
        else:
            embs = []
            for i, key in enumerate(self.e2w):
                embs.append(self.word_emb[i](input_ids[..., i]).unsqueeze(2)) # B x L x 1 x d
            embs = torch.cat([*embs], dim=-2) # B x L x F x d

            embs_shape = embs.shape
            embs = embs.view(-1, embs_shape[2], embs_shape[3]) # (B x L) x F x d

            self_attention_outputs  = self.interaction_attn(embs)
            embs_interaction = self_attention_outputs[0]

            embs = embs_interaction.view(embs_shape[0], embs_shape[1], embs_shape[2], embs_shape[3]).reshape((embs_shape[0], embs_shape[1], embs_shape[2]*embs_shape[3]))


        emb_linear = self.in_linear(embs)
        emb_linear = torch.cat([special_emb, emb_linear], dim=1)
        attn_mask = torch.cat([torch.ones((bs, 1)).to(input_ids.device), attn_mask], dim=1)

        # feed to former
        y = self.former(inputs_embeds=emb_linear, attention_mask=attn_mask,
                             output_hidden_states=output_hidden_states)
        # y = y.last_hidden_state         # (batch_size, seq_len, 768)
        return y

    def get_rand_tok(self):
        c1, c2, c3, c4 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3]
        return np.array(
            [random.choice(range(c1)), random.choice(range(c2)), random.choice(range(c3)), random.choice(range(c4))])
