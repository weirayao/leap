"""
Taken from:
https://github.com/bethgelab/slow_disentanglement/blob/master/scripts/dataset.py
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
	def __init__(self, path='./data/kitti/', transform=None,
				 max_delta_t=5):
		self.path = path
		self.data = None
		self.latents = None
		self.lens = None
		self.cumlens = None
		self.max_delta_t = max_delta_t
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
			from urllib import request
			url = self.url
			request.urlretrieve(url, file_path)

		with open(file_path, 'rb') as data:
			data = pickle.load(data)
		self.data = data['pedestrians']
		self.latents = data['pedestrians_latents']

		self.lens = [len(seq) - 1 for seq in self.data]  # start image in sequence can never be starting point
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
		sequence_ind = np.searchsorted(self.cumlens, index, side='right')
		if sequence_ind == 0:
			start_ind = index
		else:
			start_ind = index - self.cumlens[sequence_ind - 1]
		seq_len = len(self.data[sequence_ind])
		t_steps_forward = np.random.randint(1, self.max_delta_t + 1)
		end_ind = min(start_ind + t_steps_forward, seq_len - 1)

		first_sample = self.data[sequence_ind][start_ind].astype(np.uint8) * 255
		second_sample = self.data[sequence_ind][end_ind].astype(np.uint8) * 255

		latents1 = self.latents[sequence_ind][start_ind]  # center of mass vertical, com hor, area
		latents2 = self.latents[sequence_ind][end_ind]  # center of mass vertical, com hor, area

		if self.transform:
			stack = np.concatenate([first_sample[:, :, None],
									second_sample[:, :, None],
									np.ones_like(second_sample[:, :, None]) * 255],  # add ones to treat like RGB image
								   axis=2)
			samples = self.transform(stack)  # do same transforms to start and ending
			first_sample, second_sample = samples[0], samples[1]

		if len(first_sample.shape) == 2:  # set channel dim to 1
			first_sample = first_sample[None]
			second_sample = second_sample[None]

		if np.issubdtype(first_sample.dtype, np.uint8) or np.issubdtype(second_sample.dtype, np.uint8):
			first_sample = first_sample.astype(np.float32) / 255.
			second_sample = second_sample.astype(np.float32) / 255.
		
		sample = {"yt": np.expand_dims(latents1, 0), 
				  "yt_": np.expand_dims(latents2, 0), 
				  "xt": np.expand_dims(first_sample, 0), 
				  "xt_": np.expand_dims(second_sample, 0), }

		return sample

	def __len__(self):
		return self.cumlens[-1]