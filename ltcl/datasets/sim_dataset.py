import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset

class SimulationDataset(Dataset):
	
	def __init__(self, directory, transition="linear_nongaussian"):
		super().__init__()
		assert transition in ["linear_nongaussian", "post_nonlinear_gaussian", 
							  "post_nonlinear_nongaussian", "post_nonlinear_nongaussian"]
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

class SimulationDatasetTwoSample(Dataset):
	
	def __init__(self, transition="linear_nongaussian"):
		super().__init__()
		assert transition in ["linear_nongaussian", "post_nonlinear_gaussian", 
							  "post_nonlinear_nongaussian", "post_nonlinear_nongaussian"]
		self.path = os.path.join(DIR, transition, "data.npz")
		self.npz = np.load(self.path)
		self.data = { }
		for key in ["yt", "xt", "yt_", "xt_"]:
			self.data[key] = self.npz[key]
		self.min = np.min(self.data["xt_"], axis=0).reshape(1, -1)
		self.max = np.max(self.data["xt_"], axis=0).reshape(1, -1)

	def __len__(self):
		return len(self.data["yt"])

	def __getitem__(self, idx):
		yt = torch.from_numpy(self.data["yt"][idx])
		xt = torch.from_numpy((self.data["xt"][idx]-self.min / (self.max-self.min)))
		yt_ = torch.from_numpy(self.data["yt_"][idx]).unsqueeze(0)
		xt_ = torch.from_numpy((np.expand_dims(self.data["xt_"][idx], axis=0)-self.min) / (self.max-self.min))

		sample1 = {"yt": yt,
				  "yt_": yt_,
				  "xt": xt,
				  "xt_": xt_}

		idx_rnd = random.randint(0, len(self.data["yt"])-1)
		ytr = torch.from_numpy(self.data["yt"][idx_rnd])
		xtr = torch.from_numpy((self.data["xt"][idx_rnd]-self.min / (self.max-self.min)))
		ytr_ = torch.from_numpy(self.data["yt_"][idx_rnd]).unsqueeze(0)
		xtr_ = torch.from_numpy((np.expand_dims(self.data["xt_"][idx_rnd], axis=0)-self.min) / (self.max-self.min))

		sample2 = {"yt": ytr,
				  "yt_": ytr_,
				  "xt": xtr,
				  "xt_": xtr_}

		return sample1, sample2

class TupleDataset(torch.utils.data.Dataset):
	def __init__(self, split: str = "train"):
		super().__init__()
		assert split in ("train", "val")
		with open(os.path.join(DIR, "%s.txt"%split), 'r') as f:
			self.datum_names = [datum_name.rstrip() for datum_name in f.readlines()]
		self.samples_per_datum = 64
	def __len__(self):
		return len(self.datum_names) * self.samples_per_datum
	
	def __getitem__(self, idx):
		datum_idx  = idx // self.samples_per_datum 
		sample_idx = idx % self.samples_per_datum 
		self.datum_names = [ele.replace('\\', '/') for ele in self.datum_names]
		datum = np.load(self.datum_names[datum_idx])
		# latent factor
		# yt = y_t        (batch_size, length, size)
		# yt_ = y_(t+1)   (batch_size, 1, size)
		# observed variable
		# xt = x_t        (batch_size, length, size)
		# xt_ = x_(t+1)   (batch_size, 1, size)
		sample = {"yt": torch.from_numpy(datum["yt"][sample_idx]),
				  "yt_": torch.from_numpy(datum["yt_"][sample_idx]),
				  "xt": torch.from_numpy(datum["xt"][sample_idx]),
				  "xt_": torch.from_numpy(datum["xt_"][sample_idx])}
		return sample

class SimulationDatasetTS(Dataset):
	
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

class SimulationDatasetTSTwoSample(Dataset):
	
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
		idx_rnd = random.randint(0, len(self.data["yt"])-1)
		ytr = torch.from_numpy(self.data["yt"][idx_rnd].astype('float32'))
		xtr = torch.from_numpy(self.data["xt"][idx_rnd].astype('float32'))
		sample = {"s1": {"yt": yt, "xt": xt},
				  "s2": {"yt": ytr, "xt": xtr}
				  }
		return sample


class SimulationDatasetTSTwoSampleNS(Dataset):
	
	def __init__(self, directory, transition="linear_nongaussian_ts"):
		super().__init__()
		assert transition in ["linear_nongaussian_ts", "nonlinear_gaussian_ts", 
							  "nonlinear_nongaussian_ts", "nonlinear_ns", "nonlinear_gau_ns"]
		self.path = os.path.join(directory, transition, "data.npz")
		self.npz = np.load(self.path)
		self.data = { }
		for key in ["yt", "xt", "ct"]:
			self.data[key] = self.npz[key]

	def __len__(self):
		return len(self.data["yt"])

	def __getitem__(self, idx):
		yt = torch.from_numpy(self.data["yt"][idx].astype('float32'))
		xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
		ct = torch.from_numpy(self.data["ct"][idx].astype('float32'))
		idx_rnd = random.randint(0, len(self.data["yt"])-1)
		ytr = torch.from_numpy(self.data["yt"][idx_rnd].astype('float32'))
		xtr = torch.from_numpy(self.data["xt"][idx_rnd].astype('float32'))
		ctr = torch.from_numpy(self.data["ct"][idx_rnd].astype('float32'))
		sample = {"s1": {"yt": yt, "xt": xt, "ct": ct},
				  "s2": {"yt": ytr, "xt": xtr, "ct": ctr}
				  }
		return sample