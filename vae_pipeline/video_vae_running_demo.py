import torch
from torch.utils.data import DataLoader, random_split

from video_vae_trainer import Trainer
from video_vae_model import TemporalVAE
from utils_dataset import Cars3D, Shapes3D, KittiMasks, NaturalSprites

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################## loading dataset #####################################
# 1. car3d
# dataset = Cars3D(prior='uniform', rate=1, k=-1)
# train_data, val_data = random_split(dataset, [10000, 7568])
# 2. shapes3d
# dataset = Shapes3D(prior='laplace', rate=1, k=-1)
# train_data, val_data = random_split(dataset, [400000, 80000])
# 3. natural_sprites
dataset = NaturalSprites()
train_data, test_data, val_data = random_split(dataset, [200, 794, 206800])
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False)

#%%
########################## training model #####################################
vae_model = TemporalVAE()
trainer = Trainer(vae_model, train_dataloader, test_dataloader, epochs=1)
torch.cuda.empty_cache()
trainer.load_checkpoint()
trainer.train_model()
