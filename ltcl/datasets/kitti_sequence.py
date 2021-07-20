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
import numpy as np
from PIL import Image
from urllib import request
from torchvision import transforms
from torch.utils.data import Dataset


class KittiMasks(Dataset):
	'''
	latents encode:
	0: center of mass vertical position
	1: center of mass horizontal position
	2: area
	'''
	def __init__(self, path='./data/kitti/', transform=None, args=None):
		self.path = path
		self.data = None
		self.latents = None
		self.lens = None
		self.cumlens = None
		self.delta_t = args.delta_t
		self.kitti_len = args.kitti_len
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
		# determine start_ind
		start_ind = 0
        
		sample_list = []  # hight_size, width_size
		for i in range(self.kitti_len):
			sample_list.append(self.data[sequence_ind][start_ind + i*self.delta_t].astype(np.uint8) * 255)
		samples = np.array(sample_list)	

		latent_list = []  # center of mass vertical, com hor, area
		for i in range(self.kitti_len):
			latent_list.append(self.latent[sequence_ind][start_ind + i*self.delta_t])
		latents = np.array(latent_list)	

		stack = sample_list[0]
		if self.transform:
			for i in range(self.kitti_len-1):
				stack = np.concatenate([stack[:, :, None],sample_list[i+1][:, :, None]],axis=2)
			stack = np.concatenate([stack, np.ones_like(sample_list[0][:, :, None]) * 255], axis=2)  # add ones to treat like RGB image
			samples = self.transform(stack)  # do same transforms to start and ending
			samples = samples[:,:-1]
			
		if len(samples[0].shape) == 2:  # set channel dim to 1
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