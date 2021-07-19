import os
import glob
import torch
import random
import numpy as np
import ipdb as pdb
from torch.utils.data import Dataset

class SimulationDataset(Dataset):
	
	def __init__(self, directory, transition="linear_nongaussian"):
		super().__init__()
		assert transition in ["linear_nongaussian", "nonlinear_gaussian", 
							  "nonlinear_nongaussian", "post_nonlinear"]
		self.path = os.path.join(directory, transition, "data.npz")
		self.npz = np.load(self.path)
		self.data = { }
		for key in ["yt", "xt", "yt_", "xt_"]:
			self.data[key] = self.npz[key]

	def __len__(self):
		return len(self.data["yt"])

	def __getitem__(self, idx):
		yt = torch.from_numpy(self.data["yt"][idx].astype('float32'))
		xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
		yt_ = torch.from_numpy(self.data["yt_"][idx].astype('float32')).unsqueeze(0)
		xt_ = torch.from_numpy(np.expand_dims(self.data["xt_"][idx], axis=0).astype('float32'))

		sample = {"yt": yt, "yt_": yt_, "xt": xt, "xt_": xt_}
		return sample

class SimulationDataset_ts(Dataset):
	
	def __init__(self, directory, transition="linear_nongaussian_ts"):
		super().__init__()
		assert transition in ["linear_nongaussian_ts", "nonlinear_gaussian_ts", 
							  "nonlinear_nongaussian_ts", "pnl_nongaussian_ts"]
		self.path = os.path.join(directory, transition, "data.npz")
		self.npz = np.load(self.path)
		self.data = { }
		for key in ["yt", "xt"]:
			self.data[key] = self.npz[key]

	def __len__(self):
		return len(self.data["yt"])

	def __getitem__(self, idx):
		yt = torch.from_numpy(self.data["yt"][idx].astype('float32'))
		xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
		sample = {"yt": yt, "xt": xt}
		return sample

