"""Post-Nonlinear Causal Model Estimated by Temporal VAE"""
import torch
from torch import nn

class LinearUnit(nn.Module):
	def __init__(self, in_features, out_features, batchnorm=False, nonlinearity=nn.LeakyReLU(0.2)):
		super(LinearUnit, self).__init__()
		if batchnorm is True:
			self.model = nn.Sequential(
					nn.Linear(in_features, out_features),
					nn.BatchNorm1d(out_features), nonlinearity)
		else:
			self.model = nn.Sequential(
					nn.Linear(in_features, out_features), nonlinearity)

	def forward(self, x):
		return self.model(x)

class TemporalVAESynthetic(nn.Module):
	def __init__(
		self, 
		y_dim=2, 
		input_dim=2, 
		hidden_dim=128,
		negative_slope = 0.2,  
		factorised=True):
		"""Synthetic uses 3-layer MLP+leakly RELU as mixing/unmixing function
		Args:
			y_dim: Dimensions of latent causal factors.
			input_dim: Dimensions of observation data.
			hidden_dim: Dimensions of MLP hidden layer.
			negative_slope: LeakyRELU slope.
		"""
		super().__init__()
		self.y_dim = y_dim
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.factorised = factorised

		self.encoder = nn.Sequential(
									nn.Linear(input_dim, hidden_dim),
									nn.LeakyReLU(negative_slope),
									nn.Linear(hidden_dim, hidden_dim),
									nn.LeakyReLU(negative_slope),
									nn.Linear(hidden_dim, hidden_dim)
									)

		self.decoder = nn.Sequential(
									nn.Linear(input_dim, hidden_dim),
									nn.LeakyReLU(1/negative_slope),
									nn.Linear(hidden_dim, hidden_dim),
									nn.LeakyReLU(1/negative_slope),
									nn.Linear(hidden_dim, hidden_dim)
									)

		# Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
		self.z_prior_lstm = nn.LSTMCell(self.y_dim, self.hidden_dim)
		self.z_prior_mean = nn.Linear(self.hidden_dim, self.y_dim)
		self.z_prior_logvar = nn.Linear(self.hidden_dim, self.y_dim)

		if self.factorised is True:
			# Paper says : 1 Hidden Layer MLP. Last layers shouldn't have any nonlinearities
			self.z_inter = LinearUnit(self.hidden_dim, self.hidden_dim, batchnorm=False)
			self.z_mean = nn.Linear(self.hidden_dim, self.y_dim)
			self.z_logvar = nn.Linear(self.hidden_dim, self.y_dim)
		else:
			# TODO: Check if one affine transform is sufficient. Paper says distribution is parameterised by RNN over LSTM. Last layer shouldn't have any nonlinearities
			self.z_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
			self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
			# Each timestep is for each z so no reshaping and feature mixing
			self.z_mean = nn.Linear(self.hidden_dim, self.y_dim)
			self.z_logvar = nn.Linear(self.hidden_dim, self.y_dim)

	def sample_y(self, batch_size, random_sampling=True):
		y_out = None 
		y_means = None
		y_logvars = None

		# All states are initially set to 0, especially z_0 = 0
		y_t = torch.zeros(batch_size, self.y_dim, device=device)
		y_mean_t = torch.zeros(batch_size, self.y_dim, device=device)
		y_logvar_t = torch.zeros(batch_size, self.y_dim, device=device)
		h_t = torch.zeros(batch_size, self.hidden_dim, device=device)
		c_t = torch.zeros(batch_size, self.hidden_dim, device=device)

		h_t, c_t = self.z_prior_lstm(y_t, (h_t, c_t))
		y_mean_t = self.z_prior_mean(h_t)
		y_logvar_t = self.z_prior_logvar(h_t)
		y_t = self.reparameterize(y_mean_t, y_logvar_t, random_sampling)
		if y_out is None:
			# If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, y_dim]
			y_out = y_t.unsqueeze(1)
			y_means = y_mean_t.unsqueeze(1)
			y_logvars = y_logvar_t.unsqueeze(1)
		else:
			# If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
			y_out = torch.cat((y_out, y_t.unsqueeze(1)), dim=1)
			y_means = torch.cat((y_means, y_mean_t.unsqueeze(1)), dim=1)
			y_logvars = torch.cat((y_logvars, y_logvar_t.unsqueeze(1)), dim=1)

		return y_means, y_logvars, y_out

	def encode_frames(self, x):
		x = x.view(-1, self.input_dim)
		yt = self.encoder(x)
		# yt = x.view(-1, self.hidden_dim)
		return yt

	def decode_frames(self, yt):
		x = self.decoder(yt) 
		x = x.view(-1, self.input_dim)
		return x

	def reparameterize(self, mean, logvar, random_sampling=True):
		if random_sampling is True:
			eps = torch.randn_like(logvar)
			std = torch.exp(0.5*logvar)
			z = mean + eps*std
			return z
		else:
			return mean

	def encode_y(self, yt):
		mean = self.z_mean(yt)
		logvar = self.z_logvar(yt)
		yt_ = self.z_inter(yt)
		yt_ = self.z_inter(yt_)
		return mean, logvar, self.reparameterize(mean, logvar, self.training), yt_

	def forward(self, xt, xt_):
		# Past frames/snapshots: xt = x_t           (batch_size, length, size)
		# Current frame/snapshot: xt_ = x_(t+1)     (batch_size, length, size)
		yt = self.encode_frames(xt)
		y_mean_prior, y_logvar_prior, _ = self.sample_y(xt.size(0), random_sampling=self.training)
		yt_mean, yt_logvar, yt, yt_ = self.encode_y(yt)
		# yt ~ Guassian distribution, while we do not constraint on yt_
		recon_xt = self.decode_frames(yt)
		recon_xt_ = self.decode_frames(yt_)
		return yt_mean, yt_logvar, yt, yt_, y_mean_prior, y_logvar_prior, recon_xt, recon_xt_
