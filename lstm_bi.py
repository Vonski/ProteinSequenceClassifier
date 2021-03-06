#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:03:25 2019

@author: Chonghua Xue (Kolachalama's Lab, BU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence


class LSTM_Bi(nn.Module):
    def __init__(self, in_dim, embedding_dim, hidden_dim, out_dim, device, fixed_len):
        super(LSTM_Bi, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(in_dim, embedding_dim)
        self.lstm_f = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_b = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.fixed_len = fixed_len
        self.forward = self.forward_flen if fixed_len else self.forward_vlen

    def forward_vlen(self, Xs, _aa2id):
        batch_size = len(Xs)

        # pad <EOS> & <SOS>
        Xs_f = [[_aa2id['<SOS>']] + seq[:-1] for seq in Xs]
        Xs_b = [[_aa2id['<EOS>']] + seq[::-1][:-1] for seq in Xs]

        # get sequence lengths
        Xs_len = [len(seq) for seq in Xs_f]
        lmax = max(Xs_len)

        # list to *.tensor
        Xs_f = [torch.tensor(seq, device='cpu') for seq in Xs_f]
        Xs_b = [torch.tensor(seq, device='cpu') for seq in Xs_b]

        # padding
        Xs_f = pad_sequence(Xs_f, batch_first=True).to(self.device)
        Xs_b = pad_sequence(Xs_b, batch_first=True).to(self.device)

        # embedding
        Xs_f = self.word_embeddings(Xs_f)
        Xs_b = self.word_embeddings(Xs_b)

        # packing the padded sequences
        Xs_f = pack_padded_sequence(Xs_f, Xs_len, batch_first=True, enforce_sorted=False)
        Xs_b = pack_padded_sequence(Xs_b, Xs_len, batch_first=True, enforce_sorted=False)

        # feed the lstm by the packed input
        ini_hc_state_f = (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                          torch.zeros(1, batch_size, self.hidden_dim).to(self.device))
        ini_hc_state_b = (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                          torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

        lstm_out_f, _ = self.lstm_f(Xs_f, ini_hc_state_f)
        lstm_out_b, _ = self.lstm_b(Xs_b, ini_hc_state_b)

        # unpack outputs
        lstm_out_f, _ = pad_packed_sequence(lstm_out_f, batch_first=True)
        lstm_out_b, _ = pad_packed_sequence(lstm_out_b, batch_first=True)

        lstm_out_valid_f = lstm_out_f.reshape(-1, self.hidden_dim)
        lstm_out_valid_b = lstm_out_b.reshape(-1, self.hidden_dim)

        idx_f = []
        [idx_f.extend([i * lmax + l - 1]) for i, l in enumerate(Xs_len)]
        idx_f = torch.tensor(idx_f, device=self.device)

        idx_b = []
        [idx_b.extend([i * lmax + l - 1]) for i, l in enumerate(Xs_len)]
        idx_b = torch.tensor(idx_b, device=self.device)

        my_lstm_out_valid_f = torch.index_select(lstm_out_valid_f, 0, idx_f)
        my_lstm_out_valid_b = torch.index_select(lstm_out_valid_b, 0, idx_b)

        my_lstm_out_valid = torch.cat((my_lstm_out_valid_f, my_lstm_out_valid_b), 1)

        # lstm hidden state to output space
        out = F.relu(self.fc1(my_lstm_out_valid))
        # print(out.shape)
        out = F.relu(self.fc2(out))
        # print(out.shape)
        out = F.sigmoid(self.fc3(out))
        # print(out.shape)

        return out
    
    def set_param(self, param_dict):
        try:
            for pn, _ in self.named_parameters():
                exec('self.%s.data = torch.tensor(param_dict[pn])' % pn)
            self.hidden_dim = param_dict['hidden_dim']
            self.fixed_len = param_dict['fixed_len']
            self.forward = self.forward_flen if self.fixed_len else self.forward_vlen
            self.to(self.device)
        except:
            print('Unmatched parameter names or shapes.')      
    
    def get_param(self):
        param_dict = {}
        for pn, pv in self.named_parameters():
            param_dict[pn] = pv.data.cpu().numpy()
        param_dict['hidden_dim'] = self.hidden_dim
        param_dict['fixed_len'] = self.fixed_len
        return param_dict
        