'''
ref:
    https://github.com/yatindandi/Disentangled-Sequential-Autoencoder/blob/master/model.py
    https://github.com/bethgelab/slow_disentanglement/blob/26eef4557ad25f1991b6f5dc774e37e192bdcabf/scripts/model.py
'''
import torch
from torch import nn

from utils_vae import LinearUnit


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TemporalVAE(nn.Module):
	def __init__(self, e_dim=2, input_dim=2, nonlinearity=None):
		super(TemporalVAE, self).__init__()
		self.e_dim = e_dim
		self.input_dim = input_dim 
		'''
		please add your flow module
		self.flow = XX
		'''
		self.enc = nn.Sequential(
								nn.Linear(input_dim*2, 8),
								nn.Tanh(),
								nn.Linear(8, e_dim),
								nn.Tanh()
								)

		self.past_enc = nn.Sequential(
								nn.Linear(input_dim*2, 8),
								nn.Tanh(),
								nn.Linear(8, e_dim),
								nn.Tanh()
								)

		self.dec = nn.Sequential(
								nn.Linear(e_dim, 8),
								nn.Tanh(),
								nn.Linear(8, input_dim),
								nn.Tanh()
								)

		self.y_mean = nn.Linear(self.e_dim, self.e_dim)
		self.y_logvar = nn.Linear(self.e_dim, self.e_dim)
		self.f1 = LinearUnit(self.e_dim, self.e_dim, batchnorm=False)
		self.f2 = LinearUnit(self.e_dim, self.e_dim, batchnorm=False)

	def encode(self, xt, xt_):
		input_x = torch.cat((xt, xt_), dim=0)
		input_x = input_x.view(-1, self.input_dim)
		et = self.enc(input_x)
		return et

	def past_encode(self, xt):
		xt = xt.view(-1, self.input_dim)
		yt = self.past_enc(xt)
		return yt

	def decode(self, yt):
		xt = self.dec(yt) 
		xt = xt.view(-1, self.input_dim)
		return xt

	def reparameterize(self, mean, logvar, random_sampling=True):
		if random_sampling is True:
			eps = torch.randn_like(logvar)
			std = torch.exp(0.5*logvar)
			z = mean + eps*std
			return z
		else:
			return mean

	def encode_y(self, et, yt):
		mean = self.y_mean(et)
		logvar = self.y_logvar(et)
		et = self.reparameterize(mean, logvar, self.training)
		'''
		please add your flow module here
		eps_t = self.flow(et)
		'''
		yt_ = self.f1(yt) + eps_t # addictive noise model
		yt_ = self.f2(yt_) # pnl model
		return mean, logvar, yt_

	def forward(self, xt, xt_):
		# xt = x_t        (batch_size, input_dim)
		# xt_ = x_(t+1)   (batch_size, input_dim)
		yt = self.past_encode(xt) # I have question about this part
		et = self.encode(xt, xt_)
		et_mean, et_logvar, yt_ = self.encode_y(et, yt)
		# et ~ Guassian distribution, while we do not constraint on yt
		recon_xt_ = self.decode(yt_)
		return et_mean, et_logvar, yt_, recon_xt_