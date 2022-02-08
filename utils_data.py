#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:05:54 2019

@author: Chonghua Xue (Kolachalama's Lab, BU)
"""

from torch.utils.data import Dataset

# true if gapped else false
vocab_o = { True: ['-'] + ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
           False: ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']}
aa2id_o = { True: dict(zip(vocab_o[True],  list(range(len(vocab_o[True]))))),
           False: dict(zip(vocab_o[False], list(range(len(vocab_o[False])))))}
id2aa_o = { True: dict(zip(list(range(len(vocab_o[True]))),  vocab_o[True])),
           False: dict(zip(list(range(len(vocab_o[False]))), vocab_o[False]))}

vocab_i = { True: vocab_o[True]  + ['<SOS>', '<EOS>'],
           False: vocab_o[False] + ['<SOS>', '<EOS>']}
aa2id_i = { True: dict(zip(vocab_i[True],  list(range(len(vocab_i[True]))))),
           False: dict(zip(vocab_i[False], list(range(len(vocab_i[False])))))}
id2aa_i = { True: dict(zip(list(range(len(vocab_i[True]))),  vocab_i[True])),
           False: dict(zip(list(range(len(vocab_i[False]))), vocab_i[False]))}


class ProteinSeqDataset(Dataset):
    def __init__(self, hfn, mfn, gapped=True):
        # load data
        with open(hfn, 'r') as f:
            humans_data = [l.strip('\n') for l in f]
        with open(mfn, 'r') as f:
            mice_data = [l.strip('\n') for l in f]
        self.data = humans_data + mice_data
        self.labels = [1] * len(humans_data) + [0] * len(mice_data)

        # char to id
        self.data = [[aa2id_i[gapped][c] for c in r] for r in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    X = [x[0] for x in batch]
    y = [[float(x[1])] for x in batch]
    return X, y