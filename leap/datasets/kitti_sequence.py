"""

Taken from:
https://github.com/bethgelab/slow_disentanglement/blob/master/scripts/dataset.py

required args:
- args.delta_t
- args.kitti_len

"""

import os
import torch
import pickle
import random
import numpy as np
from PIL import Image
from urllib import request
from torchvision import transforms
from torch.utils.data import Dataset

import ipdb as pdb

class KittiMasks(Dataset):
	'''
	latents encode:
	0: center of mass vertical position
	1: center of mass horizontal position
	2: area
	'''
	def __init__(self, path='./data/kitti/', transform=None, max_delta_t=5, kitti_len=5):
		self.path = path
		self.data = None
		self.latents = None
		self.lens = None
		self.cumlens = None
		self.delta_t = max_delta_t
		self.kitti_len = kitti_len
		self.threshold = self.delta_t*self.kitti_len
		self.fname = 'kitti_peds_v2.pickle'
		self.url = 'https://zenodo.org/record/3931823/files/kitti_peds_v2.pickle?download=1'

		if transform == 'default':
			self.transform = transforms.Compose(
				[
					transforms.ToPILImage(),
					transforms.RandomAffine(degrees=(2., 2.), translate=(5 / 64., 5 / 64.)),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					lambda x: x.numpy()
				])
		else:
			self.transform = None

		self.load_data()

	def load_data(self):
		# download if not avaiable
		file_path = os.path.join(self.path, self.fname)
		if not os.path.exists(file_path):
			os.makedirs(self.path, exist_ok=True)
			print(f'file not found, downloading from {self.url} ...')
			url = self.url
			request.urlretrieve(url, file_path)

		with open(file_path, 'rb') as data:
			data = pickle.load(data)
		self.data = data['pedestrians']
		self.latent = data['pedestrians_latents']
		# filter short time series
		self.data = [x for x in self.data if x.shape[0] >= self.threshold]
		self.latent = [x for x in self.latent if x.shape[0] >= self.threshold]

		self.lens = [len(seq) - 1 for seq in self.data]
		self.cumlens = np.cumsum(self.lens) 

	def sample_observations(self, num, random_state, return_latents=False):
		"""Sample a batch of observations X. Needed in dis. lib."""
		assert not (num % 2)
		batch_size = int(num / 2)
		indices = random_state.choice(self.__len__(), 2 * batch_size, replace=False)
		batch, latents = [], []
		for ind in indices:
			first_sample, second_sample, l1, l2 = self.__getitem__(ind)
			batch.append(first_sample)
			latents.append(l1)
		batch = np.stack(batch)
		if not return_latents:
			return batch
		else:
			return batch, np.stack(latents)

	def sample(self, num, random_state):
		# Sample a batch of factors Y and observations X
		x, y = self.sample_observations(num, random_state, return_latents=True)
		return y, x

	def __getitem__(self, index):
		# given index, output sequence_ind
		sequence_ind = np.searchsorted(self.cumlens, index, side='right')
		if sequence_ind == 0:
			start_ind = index
		else:
			start_ind = index - self.cumlens[sequence_ind - 1]
		seq_len = len(self.data[sequence_ind])
		# determine start_ind
		sample_list = []  # hight_size, width_size
		latent_list = []  # center of mass vertical, com hor, area
		for i in range(self.kitti_len):
			sample_list.append(self.data[sequence_ind][start_ind].astype(np.uint8) * 255)
			latent_list.append(self.latent[sequence_ind][start_ind])
			t_steps_forward = np.random.randint(1, self.delta_t + 1)
			start_ind = min(start_ind + t_steps_forward, seq_len - 1)

		samples = np.array(sample_list)	
		latents = np.array(latent_list)	

		stack = sample_list[0]
		if self.transform:
			for i in range(self.kitti_len-1):
				stack = np.concatenate([stack[:, :, None],sample_list[i+1][:, :, None]],axis=2)
			stack = np.concatenate([stack, np.ones_like(sample_list[0][:, :, None]) * 255], axis=2)  # add ones to treat like RGB image
			samples = self.transform(stack)  # do same transforms to start and ending
			samples = samples[:,:-1]
			
		if len(samples[0].shape) == 3:  # set channel dim to 1
			for i in range(self.kitti_len):
				samples[i] = samples[i][None]
		
		for i in range(self.kitti_len):
			if np.issubdtype(samples[i].dtype, np.uint8):
				samples[i] = samples[i].astype(np.float32) / 255.
		
		sample = {"yt": latents, 
				  "xt": samples}

		return sample

	def __len__(self):
		return self.cumlens[-1]

class KittiMasksTwoSample(Dataset):
	'''
	latents encode:
	0: center of mass vertical position
	1: center of mass horizontal position
	2: area
	'''
	def __init__(self, path='./data/kitti/', transform=None, max_delta_t=5, kitti_len=6):
		self.path = path
		self.data = None
		self.latents = None
		self.lens = None
		self.cumlens = None
		self.delta_t = max_delta_t
		self.kitti_len = kitti_len
		self.threshold = self.delta_t*self.kitti_len
		self.fname = 'kitti_peds_v2.pickle'
		self.url = 'https://zenodo.org/record/3931823/files/kitti_peds_v2.pickle?download=1'

		if transform == 'default':
			self.transform = transforms.Compose(
				[
					transforms.ToPILImage(),
					transforms.RandomAffine(degrees=(2., 2.), translate=(5 / 64., 5 / 64.)),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					lambda x: x.numpy()
				])
		else:
			self.transform = None

		self.load_data()

	def load_data(self):
		# download if not avaiable
		file_path = os.path.join(self.path, self.fname)
		if not os.path.exists(file_path):
			os.makedirs(self.path, exist_ok=True)
			print(f'file not found, downloading from {self.url} ...')
			url = self.url
			request.urlretrieve(url, file_path)

		with open(file_path, 'rb') as data:
			data = pickle.load(data)
		self.data = data['pedestrians']
		self.latent = data['pedestrians_latents']
		# filter short time series
		self.data = [x for x in self.data if x.shape[0] >= self.threshold]
		self.latent = [x for x in self.latent if x.shape[0] >= self.threshold]

		self.lens = [len(seq) - 1 for seq in self.data]
		self.cumlens = np.cumsum(self.lens) 

	def sample_observations(self, num, random_state, return_latents=False):
		"""Sample a batch of observations X. Needed in dis. lib."""
		assert not (num % 2)
		batch_size = int(num / 2)
		indices = random_state.choice(self.__len__(), 2 * batch_size, replace=False)
		batch, latents = [], []
		for ind in indices:
			first_sample, second_sample, l1, l2 = self.__getitem__(ind)
			batch.append(first_sample)
			latents.append(l1)
		batch = np.stack(batch)
		if not return_latents:
			return batch
		else:
			return batch, np.stack(latents)

	def sample(self, num, random_state):
		# Sample a batch of factors Y and observations X
		x, y = self.sample_observations(num, random_state, return_latents=True)
		return y, x
	
	def retrieve_by_index(self, index):
		# given index, output sequence_ind
		sequence_ind = np.searchsorted(self.cumlens, index, side='right')
		if sequence_ind == 0:
			start_ind = index
		else:
			start_ind = index - self.cumlens[sequence_ind - 1]
		seq_len = len(self.data[sequence_ind])
		sample_list = []  # hight_size, width_size
		latent_list = []  # center of mass vertical, com hor, area
		for i in range(self.kitti_len):
			sample_list.append(self.data[sequence_ind][start_ind].astype(np.uint8) * 255)
			latent_list.append(self.latent[sequence_ind][start_ind])
			t_steps_forward = np.random.randint(1, self.delta_t + 1)
			start_ind = min(start_ind + t_steps_forward, seq_len - 1)

		samples = np.array(sample_list)	
		latents = np.array(latent_list)	

		if self.transform:
			stack = np.transpose(samples, (1, 2, 0)) #[H, W, C=T]
			stack = self.transform(stack) # do same transforms for the sequence

		if len(samples.shape) == 3:  # set channel dim to 1
			samples = samples[:, None]

		if np.issubdtype(samples.dtype, np.uint8):
			samples =  samples.astype(np.float32) / 255.
		
		return samples, latents

	def __getitem__(self, index):
		xt, yt = self.retrieve_by_index(index)
		idx_rnd = random.randint(0, self.cumlens[-1] -1)
		xtr, ytr = self.retrieve_by_index(idx_rnd)

		sample = {"s1": {"yt": yt, 
						 "xt": xt},
				  "s2": {"yt": ytr, 
				  		 "xt": xtr}
				  }

		return sample

	def __len__(self):
		return self.cumlens[-1]