import torch
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ltcl.datasets.sim_dataset import SimulationDataset
from ltcl.modules.lvae_nonlinear import AfflineVAESynthetic


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

    model = AfflineVAESynthetic(
        input_dim=args.input_dim, 
        z_dim=args.z_dim, 
        lag=args.lag, 
        beta=args.beta, 
        alpha=args.alpha, 
        gamma=args.gamma, 
        lr=args.lr,
        nonlinear_type=args.nonlinear_type
    )

    trainer = pl.Trainer(default_root_dir="/home/yuewen/checkpoints/nonlinear_vae",
                         gpus=[0], 
                         val_check_interval=0.1,
                         max_epochs=8
                        #  ,
                        #  callbacks=[early_stop_callback]
                         )
    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-lag', default=1, type=int)
    argparser.add_argument('-z_dim', default=4, type=int)
    argparser.add_argument('-lr', default=5e-4, type=float)
    argparser.add_argument('-input_dim', default=4, type=int)
    argparser.add_argument('-beta', default=5e-5, type=float) # kld_normal
    argparser.add_argument('-alpha', default=15, type=float) # coff
    argparser.add_argument('-gamma', default=4, type=float) # recon_zt_
    argparser.add_argument('-nonlinear_type', default='gaussian')

    args = argparser.parse_args()
    main(args)
