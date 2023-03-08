import torch
import torch.nn as nn
from transformers import AdamW
from torch.nn.utils import clip_grad_norm_

import numpy as np
import random
import tqdm
import sys
import shutil
import copy

from model import MidiFormer
from modelLM import MidiFormerLM


class FormerTrainer:
    def __init__(self, midi_former: MidiFormer, train_dataloader, valid_dataloader,
                lr, batch, max_seq_len, mask_percent, 
				cpu, use_mlm, use_clm, cuda_devices=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else 'cpu')
        self.midi_former = midi_former        # save this for ckpt
        self.model = MidiFormerLM(midi_former).to(self.device)
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('# total parameters:', self.total_params)

        if torch.cuda.device_count() > 1 and not cpu:
            print("Use %d GPUS" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        
        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.batch = batch
        self.max_seq_len = max_seq_len
        self.mask_percent = mask_percent
        self.Lseq = [i for i in range(self.max_seq_len)]
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        self.use_mlm = use_mlm
        self.use_clm = use_clm
    
    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def get_mask_ind(self):
        mask_ind = random.sample(self.Lseq, round(self.max_seq_len * self.mask_percent))
        mask80 = random.sample(mask_ind, round(len(mask_ind)*0.8))
        left = list(set(mask_ind)-set(mask80))
        rand10 = random.sample(left, round(len(mask_ind)*0.1))
        cur10 = list(set(left)-set(rand10))
        return mask80, rand10, cur10


    def train(self):
        self.model.train()
        if self.use_mlm:
            if self.use_clm:
                train_loss, train_mlm_acc, train_clm_acc = self.iteration(self.train_data, self.max_seq_len)
                return train_loss, train_mlm_acc, train_clm_acc
            else:
                train_loss, train_mlm_acc = self.iteration(self.train_data, self.max_seq_len)
                return train_loss, train_mlm_acc
        else:
            if self.use_clm:
                train_loss, train_clm_acc = self.iteration(self.train_data, self.max_seq_len)
                return train_loss, train_clm_acc
            else:
                train_loss = self.iteration(self.train_data, self.max_seq_len)
                return train_loss

    def valid(self):
        torch.cuda.empty_cache()
        self.model.eval()
        if self.use_mlm:
            if self.use_clm:
                with torch.no_grad():
                    valid_loss, valid_mlm_acc, valid_clm_acc = self.iteration(self.valid_data, self.max_seq_len, train=False)
                return valid_loss, valid_mlm_acc, valid_clm_acc
            else:
                with torch.no_grad():
                    valid_loss, valid_mlm_acc = self.iteration(self.valid_data, self.max_seq_len, train=False)
                return valid_loss, valid_mlm_acc
        else:
            if self.use_clm:
                with torch.no_grad():
                    valid_loss, valid_clm_acc = self.iteration(self.valid_data, self.max_seq_len, train=False)
                return valid_loss, valid_clm_acc
            else:
                with torch.no_grad():
                    valid_loss = self.iteration(self.valid_data, self.max_seq_len, train=False)
                return valid_loss

    def iteration(self, training_data, max_seq_len, train=True):
        pbar = tqdm.tqdm(training_data, disable=False)

        total_mlm_acc, total_clm_acc, total_losses = 0, 0, 0
        
        for ori_seq_batch in pbar:
            batch = ori_seq_batch.shape[0]
            ori_seq_batch = ori_seq_batch.to(self.device)       

            # MLM training
            # input_ids = copy.deepcopy(ori_seq_batch)
            input_ids = ori_seq_batch.clone()

            # bs, slen, _ = input_ids.shape
            # pred_mask = np.random.rand(bs, slen) <= self.mask_percent
            # pred_mask = torch.from_numpy(pred_mask.astype(np.uint8)).bool().to(self.device)
            #
            # # do not predict padding
            # pred_mask[input_ids[:, :, 0] == self.midi_former.pad_word] = 0
            # 
            # pred_probs = torch.FloatTensor([0.8, 0.1, 0.1]).to(self.device)

            # _x_real = input_ids[pred_mask]
            # _x_rand = _x_real.clone().random_(self.midi_former.n_token)
            # _x_mask = _x_real.clone().fill_(self.midi_former.mask_word)
            # probs = torch.multinomial(pred_probs, len(_x_real), replacement=True)
            # _x = _x_mask * (probs == 0).long() + _x_real * (probs == 1).long() + _x_rand * (probs == 2).long()
            # input_ids = input_ids.masked_scatter(pred_mask, _x)
            
            if self.use_mlm:
                loss_mask = torch.zeros(batch, max_seq_len)

                for b in range(batch):
                    # get index for masking
                    mask80, rand10, cur10 = self.get_mask_ind()
                    # apply mask, random, remain current token
                    for i in mask80:
                        input_ids[b][i] = self.midi_former.mask_word 
                        loss_mask[b][i] = 1 
                    for i in rand10:
                        input_ids[b][i] = random.choice(range(len(self.midi_former.e2w)))
                        loss_mask[b][i] = 1 
                    for i in cur10:
                        loss_mask[b][i] = 1 
                
                loss_mask = loss_mask.to(self.device)      

                # avoid attend to pad word
                attn_mask = (input_ids != self.midi_former.pad_word).float().to(self.device)   # (4,512)
                
                y = self.model.forward(x=input_ids, attn=attn_mask)

                # get the most likely choice with max
                output = np.argmax(y.cpu().detach().numpy(), axis=-1)
                output = output.astype(float)
                output = torch.from_numpy(output).to(self.device) #to(device).long()   # (4,512)

                # accuracy
                mlm_acc = torch.sum((ori_seq_batch == output).float() * loss_mask)
                mlm_acc /= torch.sum(loss_mask)
                total_mlm_acc += mlm_acc.item()

                # reshape (b, s, f) -> (b, f, s)
                y = y[:, ...].permute(0, 2, 1)

                # calculate losses
                mlm_loss = self.compute_loss(y, ori_seq_batch, loss_mask)

            if self.use_clm:
                # CLM training
                input_ids = ori_seq_batch.clone()

                bs, slen = input_ids.shape
                pred_mask = torch.ones((bs, slen)).bool().to(self.device)

                # do not predict padding
                pred_mask[input_ids == self.midi_former.pad_word] = 0

                # avoid attend to pad word
                attn_mask = (input_ids != self.midi_former.pad_word).float().to(self.device)  # (batch, seq_len)

                y = self.model.forward(x=input_ids, attn=attn_mask, mode="clm")

                # get the most likely choice with max
                output = np.argmax(y.cpu().detach().numpy(), axis=-1)
                output = output.astype(float)
                output = torch.from_numpy(output).to(self.device) #to(device).long()   # (4,512)

                # accuracy
                clm_acc = torch.sum((ori_seq_batch == output).float() * pred_mask)
                clm_acc /= torch.sum(pred_mask)
                total_clm_acc += clm_acc.item()

                # reshape (b, s, f) -> (b, f, s)
                y = y[:, ...].permute(0, 2, 1)

                # calculate losses
                clm_loss = self.compute_loss(y, ori_seq_batch, pred_mask)


            if self.use_mlm:
                if self.use_clm:
                    total_loss = mlm_loss + clm_loss
                else:
                    total_loss = mlm_loss
            else:
                if self.use_clm:
                    total_loss = clm_loss
                else:
                    total_loss = 0

            # udpate only in train
            if train:
                self.model.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), 3.0)
                self.optim.step()

            if self.use_mlm:
                if self.use_clm:
                    # acc
                    print('Loss: {:06f} | {:03f} {:03f} | acc: {:03f} {:03f} \r'.format(
                        total_loss, mlm_loss, clm_loss, mlm_acc, clm_acc), flush=True)
                else:
                    # acc
                    print('Loss: {:06f} | {:03f} | acc: {:03f} \r'.format(
                        total_loss, mlm_loss, mlm_acc), flush=True)   
            else:
                if self.use_clm:
                    # acc
                    print('Loss: {:06f} | {:03f} | acc: {:03f} \r'.format(
                        total_loss, clm_loss, clm_acc), flush=True)
                else:
                    # acc
                    print('Loss: {:06f} \r'.format(
                        total_loss), flush=True)   


            total_losses += total_loss.item()
        
        if self.use_mlm:
            if self.use_clm:
                return round(total_losses/len(training_data),3), round(total_mlm_acc/len(training_data),3), round(total_clm_acc/len(training_data),3)
            else:
                return round(total_losses/len(training_data),3), round(total_mlm_acc/len(training_data),3)
        else:
            if self.use_clm:
                return round(total_losses/len(training_data),3), round(total_clm_acc/len(training_data),3)
            else:
                return round(total_losses/len(training_data),3)


    def save_checkpoint(self, epoch, best_acc, valid_acc, 
                        valid_loss, train_loss, is_best, filename):
        state = {
            'epoch': epoch + 1,
            'state_dict': self.midi_former.state_dict(),
            'best_acc': best_acc,
            'valid_acc': valid_acc,
            'valid_loss': valid_loss,
            'train_loss': train_loss,
            'optimizer' : self.optim.state_dict()
        }

        torch.save(state, filename)

        best_mdl = filename.split('.')[0]+'_best.ckpt'
        if is_best:
            shutil.copyfile(filename, best_mdl)

