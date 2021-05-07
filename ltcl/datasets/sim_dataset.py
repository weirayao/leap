import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import glob

DIR = "/home/cmu_wyao/projects/data/"
MIN = torch.Tensor([-78.12708  ,  -1.7142837, -19.542505 , -22.891457 ])
MAX = torch.Tensor([591.5796 , 496.26367,  50.26638, 143.61914])

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
        sample_1 = {"x": torch.from_numpy(datum["x"][sample_idx]),
                    "y": torch.from_numpy(datum["y"][sample_idx])
                  }
        random_idx = random.randint(0, self.samples_per_datum-1)
        sample_2 = {"x": torch.from_numpy(datum["x"][random_idx]),
                    "y": torch.from_numpy(datum["y"][random_idx])
                    }
        sample_1["x"] = (sample_1["x"] - MIN) / (MAX - MIN)
        sample_1["y"] = (sample_1["y"] - MIN) / (MAX - MIN)
        sample_2["x"] = (sample_2["x"] - MIN) / (MAX - MIN)
        sample_2["y"] = (sample_2["y"] - MIN) / (MAX - MIN)
        return sample_1, sample_2