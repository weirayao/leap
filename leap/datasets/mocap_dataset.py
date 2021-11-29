import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class MocapTwoSampleNS(Dataset):

    def __init__(self, directory, dataset='mocap_point_cloud'):
        self.n_class = 12
        self.length = 6
        self.path = os.path.join(directory, dataset, "data.npz")
        self.npz = np.load(self.path, allow_pickle=True)
        self.data = { }
        for key in ["xt", "ct"]:
            self.data[key] = self.npz[key]
        self.mean = self.data["xt"].mean(axis=0)
        self.std = self.data["xt"].std(axis=0)
        self.data["xt"] = (self.data["xt"] - self.mean)/self.std
    
    def __len__(self):
        return self.data['xt'].shape[0] - self.length + 1
    
    def retrieve_by_idx(self, idx):
        while self.data["ct"][idx] != self.data["ct"][idx+self.length-1]:
            idx= random.randint(0, self.__len__()-1)
        ct = [ ]
        xt = [ ]
        for t in range(self.length):
            ct.append(torch.from_numpy(self.data["ct"][idx+t].astype('float32')))
            xt.append(torch.from_numpy(self.data["xt"][idx+t].astype('float32')))
        xt = torch.stack(xt)
        return xt, ct[0]        

    def __getitem__(self, idx):
        xt, ct = self.retrieve_by_idx(idx)
        idx_rnd = random.randint(0, self.__len__()-1)
        xtr, ctr = self.retrieve_by_idx(idx_rnd)
        sample = {"s1": {"xt": xt, "ct": ct},
                  "s2": {"xt": xtr, "ct": ctr}
                  }
        return sample

class MocapTwoSample(Dataset):

    def __init__(self, directory, dataset='mocap_point_cloud'):
        self.n_class = 12
        self.length = 6
        self.path = os.path.join(directory, dataset, "data.npz")
        self.npz = np.load(self.path, allow_pickle=True)
        self.data = { }
        for key in ["xt", "ct"]:
            self.data[key] = self.npz[key]
        self.mean = self.data["xt"].mean(axis=0)
        self.std = self.data["xt"].std(axis=0)
        self.data["xt"] = (self.data["xt"] - self.mean)/self.std
    
    def __len__(self):
        return self.data['xt'].shape[0] - self.length + 1
    
    def retrieve_by_idx(self, idx):
        while self.data["ct"][idx] != self.data["ct"][idx+self.length-1]:
            idx= random.randint(0, self.__len__()-1)
        ct = [ ]
        xt = [ ]
        for t in range(self.length):
            ct.append(torch.from_numpy(self.data["ct"][idx+t].astype('float32')))
            xt.append(torch.from_numpy(self.data["xt"][idx+t].astype('float32')))
        xt = torch.stack(xt)
        return xt, ct[0]        

    def __getitem__(self, idx):
        xt, ct = self.retrieve_by_idx(idx)
        idx_rnd = random.randint(0, self.__len__()-1)
        xtr, ctr = self.retrieve_by_idx(idx_rnd)
        sample = {"s1": {"xt": xt, "ct": ct},
                  "s2": {"xt": xtr, "ct": ctr}
                  }
        return sample

class MocapPCL(Dataset):

    def __init__(self, directory, dataset='mocap_point_cloud'):
        self.n_class = 12
        self.length = 6
        self.path = os.path.join(directory, dataset, "data.npz")
        self.npz = np.load(self.path, allow_pickle=True)
        self.data = { }
        for key in ["xt", "ct"]:
            self.data[key] = self.npz[key]
        self.mean = self.data["xt"].mean(axis=0)
        self.std = self.data["xt"].std(axis=0)
        self.data["xt"] = (self.data["xt"] - self.mean)/self.std
        self.L = 2
    
    def __len__(self):
        return self.data['xt'].shape[0] - self.length + 1
    
    def retrieve_by_idx(self, idx):
        while self.data["ct"][idx] != self.data["ct"][idx+self.length-1]:
            idx= random.randint(0, self.__len__()-1)
        ct = [ ]
        xt = [ ]
        for t in range(self.length):
            ct.append(torch.from_numpy(self.data["ct"][idx+t].astype('float32')))
            xt.append(torch.from_numpy(self.data["xt"][idx+t].astype('float32')))
        xt = torch.stack(xt)
        return xt, ct[0]        

    def __getitem__(self, idx):
        xt, ct = self.retrieve_by_idx(idx)
        xt_cur, xt_his = self.seq_to_pairs(xt)
        idx_rnd = random.randint(0, self.__len__()-1)
        xtr, ctr = self.retrieve_by_idx(idx_rnd)
        xtr_cur, xtr_his = self.seq_to_pairs(xtr)
        xt_cat = torch.cat((xt_cur, xt_his), dim=1)
        xtr_cat = torch.cat((xt_cur, xtr_his), dim=1)
        sample = {"s1": {"xt": xt, "ct": ct},
                  "s2": {"xt": xtr, "ct": ctr},
                  "pos": {"x": xt_cat, "y": 1},
                  "neg": {"x": xtr_cat, "y": 0}
                  }
        return sample

    def seq_to_pairs(self, x):
        x = x.unfold(dimension = 0, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 1, 2)
        xx, yy = x[:,-1:], x[:,:-1]
        return xx, yy


