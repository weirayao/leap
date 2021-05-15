'''
ref:
    https://github.com/yatindandi/Disentangled-Sequential-Autoencoder/blob/master/model.py
    https://github.com/bethgelab/slow_disentanglement/blob/26eef4557ad25f1991b6f5dc774e37e192bdcabf/scripts/model.py
'''
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# A block consisting of convolution, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
class ConvUnit(nn.Module):
	def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
		super(ConvUnit, self).__init__()
		if batchnorm is True:
			self.model = nn.Sequential(
					nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
					nn.BatchNorm2d(out_channels), nonlinearity)
		else:
			self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride, padding), nonlinearity)

	def forward(self, x):
		return self.model(x)

# A block consisting of a transposed convolution, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
class ConvUnitTranspose(nn.Module):
	def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, out_padding=0, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
		super(ConvUnitTranspose, self).__init__()
		if batchnorm is True:
			self.model = nn.Sequential(
					nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, out_padding),
					nn.BatchNorm2d(out_channels), nonlinearity)
		else:
			self.model = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, out_padding), nonlinearity)

	def forward(self, x):
		return self.model(x)

# A block consisting of an affine layer, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
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

class TemporalVAE(nn.Module):
	def __init__(self, y_dim=128, conv_dim=512, step=128, in_size=64, factorised=True, nonlinearity=None):
		super(TemporalVAE, self).__init__()
		self.step = step
		self.y_dim = y_dim
		self.in_size = in_size
		self.conv_dim = conv_dim
		self.factorised = factorised
		nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity

		self.conv = nn.Sequential(
				ConvUnit(3, step, 5, 1, 2),    # 3*64*64 -> 256*64*64
				ConvUnit(step, step, 5, 2, 2), # 256,64,64 -> 256,32,32
				ConvUnit(step, step, 5, 2, 2), # 256,32,32 -> 256,16,16
				ConvUnit(step, step, 5, 2, 2), # 256,16,16 -> 256,8,8
				)
		self.final_conv_size = in_size // 8
		self.conv_fc = nn.Sequential(LinearUnit(step * (self.final_conv_size ** 2), self.conv_dim * 2),
				LinearUnit(self.conv_dim * 2, self.conv_dim))

		self.deconv_fc = nn.Sequential(LinearUnit(self.y_dim, self.conv_dim * 2, False),
				LinearUnit(self.conv_dim * 2, step * (self.final_conv_size ** 2), False))
		self.deconv = nn.Sequential(
				ConvUnitTranspose(step, step, 5, 2, 2, 1),
				ConvUnitTranspose(step, step, 5, 2, 2, 1),
				ConvUnitTranspose(step, step, 5, 2, 2, 1),
				ConvUnitTranspose(step, 3, 5, 1, 2, 0, nonlinearity=nn.Tanh()))

		# Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
		self.z_prior_lstm = nn.LSTMCell(self.y_dim, self.y_dim)
		self.z_prior_mean = nn.Linear(self.y_dim, self.y_dim)
		self.z_prior_logvar = nn.Linear(self.y_dim, self.y_dim)

		if self.factorised is True:
			# Paper says : 1 Hidden Layer MLP. Last layers shouldn't have any nonlinearities
			self.z_inter = LinearUnit(self.conv_dim, self.y_dim, batchnorm=False)
			self.z_mean = nn.Linear(self.y_dim, self.y_dim)
			self.z_logvar = nn.Linear(self.y_dim, self.y_dim)
			self.z_inter_ = nn.Linear(self.y_dim, self.y_dim)
            
		else:
			# TODO: Check if one affine transform is sufficient. Paper says distribution is parameterised by RNN over LSTM. Last layer shouldn't have any nonlinearities
			self.z_lstm = nn.LSTM(self.conv_dim, self.y_dim, 1, bidirectional=True, batch_first=True)
			self.z_rnn = nn.RNN(self.y_dim * 2, self.y_dim, batch_first=True)
			# Each timestep is for each z so no reshaping and feature mixing
			self.z_mean = nn.Linear(self.y_dim, self.y_dim)
			self.z_logvar = nn.Linear(self.y_dim, self.y_dim)

		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 1)
			elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight)

	def sample_y(self, batch_size, random_sampling=True):
		y_out = None 
		y_means = None
		y_logvars = None

		# All states are initially set to 0, especially z_0 = 0
		y_t = torch.zeros(batch_size, self.y_dim, device=device)
		y_mean_t = torch.zeros(batch_size, self.y_dim, device=device)
		y_logvar_t = torch.zeros(batch_size, self.y_dim, device=device)
		h_t = torch.zeros(batch_size, self.y_dim, device=device)
		c_t = torch.zeros(batch_size, self.y_dim, device=device)

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
		# The frames are unrolled into the batch dimension for batch processing such that x goes from
		# [batch_size, frames, channels, size, size] to [batch_size * frames, channels, size, size]
		x = x.view(-1, 3, self.in_size, self.in_size)
		x = self.conv(x)
		x = x.view(-1, self.step * (self.final_conv_size ** 2))
		yt = self.conv_fc(x)
		# The frame dimension is reintroduced and x shape becomes [batch_size, frames, conv_dim]
		# This technique is repeated at several points in the code
		yt = yt.view(-1, self.conv_dim)
		return yt

	def decode_frames(self, yt):
		x = self.deconv_fc(yt)
		x = x.view(-1, self.step, self.final_conv_size, self.final_conv_size)
		x = self.deconv(x)
		x = x.view(-1, 3, self.in_size, self.in_size)
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
		yt = self.z_inter(yt) # torch.Size([64, 512]) --> torch.Size([64, 128])
		mean = self.z_mean(yt) # torch.Size([64, 128])
		logvar = self.z_logvar(yt) # torch.Size([64, 128])
		yt_ = self.z_inter_(yt) # post nonlinear f1 & torch.Size([64, 128])
		yt_ = self.z_inter_(yt_) # post nonlinear f2 & torch.Size([64, 128])
		return mean, logvar, self.reparameterize(mean, logvar, self.training), yt_

	def forward(self, xt, xt_):
		# xt = x_t        (batch_size, n_channel, in_size, in_size)
		# xt_ = x_(t+1)   (batch_size, n_channel, in_size, in_size)
		yt = self.encode_frames(xt)
		y_mean_prior, y_logvar_prior, _ = self.sample_y(xt.size(0), random_sampling=self.training)
		yt_mean, yt_logvar, yt, yt_ = self.encode_y(yt)
		# yt ~ Guassian distribution, while we do not constraint on yt_
		recon_xt = self.decode_frames(yt)
		recon_xt_ = self.decode_frames(yt_)
		return yt_mean, yt_logvar, yt, yt_, y_mean_prior, y_logvar_prior, recon_xt, recon_xt_
