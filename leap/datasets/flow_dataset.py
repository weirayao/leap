import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import glob

DIR = "/home/cmu_wyao/projects/data/flow"
MIN = torch.Tensor([-1.9354342, -1.8592738, -1.7779869, -1.6923263, -1.6568404,
       -2.1547406, -1.7960885, -1.8174026, -1.9915712, -1.7210511,
       -2.112092 , -2.5274713, -1.8707783, -1.9960984, -2.3613997,
       -2.3149474, -1.8315119, -1.9161125, -2.0667543, -2.266219 ,
       -2.2407985, -1.824772 , -2.0922415, -1.9095674, -2.0145454,
       -1.9617085, -1.9070922, -1.9322165, -1.8413062, -1.8627869,
       -2.1569269, -1.6828921])

MAX = torch.Tensor([1.7299943, 1.840655 , 1.4923024, 1.431921 , 1.4853816, 2.122885 ,
       1.9869661, 2.0246725, 1.957955 , 1.5493772, 1.8394616, 2.7398658,
       1.922616 , 2.1277268, 2.636941 , 2.3323367, 1.6864558, 2.379957 ,
       1.8595207, 2.2347615, 2.3447344, 1.5971818, 2.07827  , 2.2094557,
       2.1073754, 1.9490763, 1.4255725, 1.9031885, 1.9159393, 1.6150727,
       2.1884263, 1.6171508])

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