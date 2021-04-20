import os
import torch
from torch.utils.data import Dataset
import numpy as np
import glob

DIR = "/home/cmu_wyao/projects/data/"

class SimulationDataset(Dataset):
    
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
        sample = {"x": torch.from_numpy(datum["x"][sample_idx]),
                  "y": torch.from_numpy(datum["y"][sample_idx])}
        return sample