import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from mix import (FeatureMLP,
                 ScoringFunc)

class PCL(pl.pl.LightningModule):

    def __init__(self, num_sources, contrast_samples, 
                hidden_dims=64, encoder_layers=3, scoring_layers=3):
        super().__init__()
        self.sources = num_sources
        self.constrast = contrast_samples
        self.encoder = FeatureMLP(input_dims=num_sources, 
                                  num_layers=encoder_layers, 
                                  negative_slope=0.01)

        self.scoring_funcs = nn.ModuleList([
            ScoringFunc(input_dims=contrast_samples, 
                        hidden_dims=hidden_dims, 
                        num_layers=scoring_layers) for _ in range(num_sources)]
            )

    def forward(self, x):
        embeddings = [ ]
        for sample in x:
            embeddings.append(self.encoder(x))
        return embeddings
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # list of contrastive_samples lists [BS X sources]
        embeddings = [ ] 
        for sample in x:
            embeddings.append(self.encoder(x))
        # embeddings of shape BS X sources X contrastive_samples
        embeddings = torch.stack(embeddings, dim=-1)
        scores = 0
        for i in range(self.sources):
            embedding = embeddings[:, i, :]
            score = self.scoring_funcs[i](embedding)
            scores += score
        loss = F.binary_cross_entropy_with_logits(scores, y)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer