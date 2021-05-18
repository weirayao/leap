'''
ref:
    https://github.com/yatindandi/Disentangled-Sequential-Autoencoder/blob/master/model.py
    https://github.com/bethgelab/slow_disentanglement/blob/26eef4557ad25f1991b6f5dc774e37e192bdcabf/scripts/model.py
'''
import torch
from torch import nn

from utils_vae import ConvUnit, ConvUnitTranspose, LinearUnit


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TemporalVAE(nn.Module):
	def __init__(self, e_dim=128, y_dim=128, conv_dim=512, step=128, in_size=64, nonlinearity=None):
		super(TemporalVAE, self).__init__()
		self.step = step
		self.e_dim = e_dim
		self.y_dim = y_dim
		self.in_size = in_size
		self.conv_dim = conv_dim
		'''
		please add your flow module
		self.flow = XX
		'''
		self.conv = nn.Sequential(
				ConvUnit(3, step, 5, 1, 2),    # 3*64*64 -> 256*64*64
				ConvUnit(step, step, 5, 2, 2), # 256,64,64 -> 256,32,32
				ConvUnit(step, step, 5, 2, 2), # 256,32,32 -> 256,16,16
				ConvUnit(step, step, 5, 2, 2), # 256,16,16 -> 256,8,8
				)
		self.final_conv_size = in_size // 8
		self.conv_fc = nn.Sequential(LinearUnit(step * (self.final_conv_size ** 2), self.conv_dim * 2),
				LinearUnit(self.conv_dim * 2, self.conv_dim))
		self.lstm = nn.LSTM(self.conv_dim*2, self.e_dim, 1, bidirectional=True, batch_first=True)
		self.rnn = nn.RNN(self.e_dim*2, self.e_dim, batch_first=True)

		self.past_conv = nn.Sequential(
				ConvUnit(3, step, 5, 1, 2),    
				ConvUnit(step, step, 5, 2, 2), 
				ConvUnit(step, step, 5, 2, 2), 
				ConvUnit(step, step, 5, 2, 2), 
				)
		self.past_conv_fc = nn.Sequential(LinearUnit(step * (self.final_conv_size ** 2), self.conv_dim * 2),
				LinearUnit(self.conv_dim * 2, self.conv_dim))

		self.past_lstm = nn.LSTM(self.conv_dim, self.y_dim, 1, bidirectional=True, batch_first=True)
		self.past_rnn = nn.RNN(self.y_dim*2, self.y_dim, batch_first=True)

		self.deconv_fc = nn.Sequential(LinearUnit(self.e_dim, self.conv_dim * 2, False),
				LinearUnit(self.conv_dim * 2, step * (self.final_conv_size ** 2), False))
		self.deconv = nn.Sequential(
				ConvUnitTranspose(step, step, 5, 2, 2, 1),
				ConvUnitTranspose(step, step, 5, 2, 2, 1),
				ConvUnitTranspose(step, step, 5, 2, 2, 1),
				ConvUnitTranspose(step, 3, 5, 1, 2, 0, nonlinearity=nn.Tanh()))

		self.y_mean = nn.Linear(self.e_dim, self.e_dim)
		self.y_logvar = nn.Linear(self.e_dim, self.e_dim)

		self.f1 = nn.Sequential(
				nn.LSTM(self.y_dim, self.y_dim, 1, bidirectional=True, batch_first=True),
				nn.RNN(self.y_dim*2, self.y_dim, batch_first=True)
				)
		self.f2 = LinearUnit(self.y_dim, self.y_dim, batchnorm=False)

		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 1)
			elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight)

	def encode_frames(self, xt):
		xt = xt.view(-1, 3, self.in_size, self.in_size)
		xt = self.conv(xt)
		xt = xt.view(-1, self.step * (self.final_conv_size ** 2))
		features = self.conv_fc(xt)
		features = features.view(-1, self.conv_dim)	
		return features

	def encode(self, xt, xt_):
		features = self.encode_frames(xt)
		features_ = self.encode_frames(xt_)
		lstm_out, _ = self.lstm(torch.cat((features,features_), dim=1))
		et, _ = self.rnn(lstm_out)			
		return et

	def past_encode(self, xt):
		xt = xt.view(-1, 3, self.in_size, self.in_size)
		xt = self.past_conv(xt)
		xt = xt.view(-1, self.step * (self.final_conv_size ** 2))
		features = self.past_conv_fc(xt)
		features = features.view(-1, self.conv_dim)
		lstm_out, _ = self.past_lstm(features)
		yt, _ = self.past_rnn(lstm_out)			
		return yt

	def decode(self, yt):
		xt = self.deconv_fc(yt)
		xt = xt.view(-1, self.step, self.final_conv_size, self.final_conv_size)
		xt = self.deconv(xt)
		xt = xt.view(-1, 3, self.in_size, self.in_size)
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
		# xt = x_t        (batch_size, n_channel, in_size, in_size)
		# xt_ = x_(t+1)   (batch_size, n_channel, in_size, in_size)
		yt = self.past_encode(xt) # have question
		et = self.encode(xt, xt_)
		et_mean, et_logvar, yt_ = self.encode_y(et, yt)
		# et ~ Guassian distribution, while we do not constraint on yt
		recon_xt_ = self.decode(yt_)
		return et_mean, et_logvar, yt_, recon_xt_
