import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ltcl.datasets.sim_dataset import SimulationDatasetTwoSample, SimulationDataset
from ltcl.modules.lvae import AfflineVAESynthetic
import argparse

def main(args):
    data = SimulationDataset()
    num_validation_samples = 500
    train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_data, batch_size=1280, shuffle=False, pin_memory=True)

    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_elbo_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min')

    model = AfflineVAESynthetic(4,4,2, beta = args.beta, gamma = args.gamma, lr=args.lr)

    trainer = pl.Trainer(default_root_dir = "/home/cmu_wyao/checkpoints/linear_vae",
                         gpus=[2], 
                         max_epochs=10,
                         callbacks=[early_stop_callback])

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-beta',
        default=1.,
        type=float
    )

    argparser.add_argument(
        '-gamma',
        default=10,
        type=float
    )

    argparser.add_argument(
        '-lr',
        default=1e-3,
        type=float
    )
    args = argparser.parse_args()
    main(args)