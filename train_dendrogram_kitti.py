import ltcl
from ltcl.modules.trainer import Dendrogram
from ltcl.datasets.kitti import KittiMasks
import torch
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import argparse

def main(args):
    print("L1 penalty: %f"%args.l1_penalty)
    dataset = KittiMasks("/home/cmu_wyao/projects/slow_disentanglement/data/kitti/")
    train_data, val_data = random_split(dataset, [80000, 2506])
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_data, batch_size=16, shuffle=False, pin_memory=True)

    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min')

    model = Dendrogram(input_dims = (1, 64, 64),
                       width = 128,
                       depth = 32,
                       n_levels = 3,
                       lags = 1,
                       lr = 1e-4,
                       l1 = args.l1_penalty) 

    trainer = pl.Trainer(default_root_dir = "/home/cmu_wyao/checkpoints/dendrogram_%f"%args.l1_penalty,
                         gpus=[2], 
                         max_epochs=3,
                         callbacks=[early_stop_callback])

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-l1',
        '--l1_penalty',
        type=float
    )
    args = argparser.parse_args()
    main(args)