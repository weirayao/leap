# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 22:08:42 2021

@author: 
"""
import os
import torch
import numpy as np

DIR = "non_linear_data/"
VALIDATION_RATIO = 0.2

class SimulationDataset(torch.utils.data.Dataset):
    def __init__(self, split: str = "train"):
        super().__init__()
        assert split in ("train", "val")
        with open(os.path.join(DIR, "%s.txt"%split), 'r') as f:
            self.datum_names = [datum_name.rstrip() for datum_name in f.readlines()]
        self.samples_per_datum = 128

    def __len__(self):
        return len(self.datum_names) * self.samples_per_datum

    def __getitem__(self, idx):
        datum_idx  = idx // self.samples_per_datum 
        sample_idx = idx % self.samples_per_datum 
        datum = np.load(self.datum_names[datum_idx])
        sample = {"xt_1": torch.from_numpy(datum["xt_1"][sample_idx]),
                  "xt": torch.from_numpy(datum["xt"][sample_idx])}
        return sample
