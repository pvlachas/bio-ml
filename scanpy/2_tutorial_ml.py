"""
In this tutorial we will build a Conditional Variational Autoencoder (CVAE) for
single-cell transcriptomics data (e.g. gene expression matrices), implemented in
Pytorch with Pyro for probabilistic modeling.

The goal is to model the joint distribution of gene expression counts X and cell
classes C conditioned on batch effects B using a latent variable Z.

The architecture looks like this:
Z, B -> X
Z -> C

- The latent representation Z and batch covariates B explain the observed gene counts X.
- The latent variable Z alone explains cell-type class C.

"""
import gdown
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import numpy as np
import scanpy as sc
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from anndata.experimental.pytorch import AnnLoader


"""
Flexible multi-layer perceptron with:
- Linear → LayerNorm → ReLU → Dropout blocks
- Optionally varying hidden dimensions
- Final linear projection to out_dim
-> used in both encoder and decoder networks
"""

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, out_dim):
        super().__init__()

        modules = []
        for in_size, out_size in zip([input_dim] + hidden_dims, hidden_dims):
            modules.append(nn.Linear(in_size, out_size))
            modules.append(nn.LayerNorm(out_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=0.05))
        modules.append(nn.Linear(hidden_dims[-1], out_dim))
        self.fc = nn.Sequential(*modules)

    def forward(self, *inputs):
        input_cat = torch.cat(inputs, dim=-1)
        return self.fc(input_cat)

"""
Conditional VAE

The CVAE extends a standard VAE by conditioning on batch effects (covariates).

"""

# input_dim: Number of input genes (features)
# n_conds: Number of batch condition variables (one-hot encoded)
# n_classes: Number of cell-type classes
# latent_dim: Dimension of latent variable  Z
# hidden_dims: Hidden layer sizes for encoder/decoder

class CVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        n_conds,
        n_classes,
        hidden_dims,
        latent_dim,
    ):
        super().__init__()
        # encoder maps [X, B] -> [mu_z, log sigma_z^2]
        self.encoder = MLP(
            input_dim + n_conds,
            hidden_dims,
            2 * latent_dim,
        )

        # decoder reconstructs X from [Z, B]
        self.decoder = MLP(latent_dim + n_conds, hidden_dims[::-1], input_dim)

        # theta leans gene-specific dispersion parameters
        self.theta = nn.Linear(n_conds, input_dim, bias=False)

        # predicts class probabilities from latent Z
        self.classifier = nn.Linear(latent_dim, n_classes)

        self.latent_dim = latent_dim

    def model(self, x, batches, classes, size_factors):
        """Probabilistic model: Generative Process
        For each observation (cell):
        1. Prior over latent variable z ~ N(0,1)
        2. Class probabilities p(C|Z) = softmax (W*z) (self.classifier)
        3. Reconstruction of gene expression (X)
            -   The decoder predicts normalized means (dec_mu)
            -   dec_theta models dispersion per gene
            -   Observations X are modeled via a Negative Binomial
                (typical for count data in single-cell RNA-seq)
        """
        pyro.module("cvae", self)

        batch_size = x.shape[0]

        with pyro.plate("data", batch_size):
            z_loc = x.new_zeros((batch_size, self.latent_dim))
            z_scale = x.new_ones((batch_size, self.latent_dim))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            classes_probs = self.classifier(z).softmax(dim=-1)
            pyro.sample("class", dist.Categorical(probs=classes_probs), obs=classes)

            dec_mu = self.decoder(z, batches).softmax(dim=-1) * size_factors[:, None]
            dec_theta = torch.exp(self.theta(batches))

            logits = (dec_mu + 1e-6).log() - (dec_theta + 1e-6).log()

            pyro.sample(
                "obs",
                dist.NegativeBinomial(
                    total_count=dec_theta,
                    logits=logits,
                ).to_event(1),
                obs=x.int(),
            )

    def guide(self, x, batches, classes, size_factors):
        batch_size = x.shape[0]

        with pyro.plate("data", batch_size):
            z_loc_scale = self.encoder(x, batches)

            z_mu = z_loc_scale[:, :self.latent_dim]
            z_var = torch.sqrt(torch.exp(z_loc_scale[:, self.latent_dim:]) + 1e-4)

            pyro.sample("latent", dist.Normal(z_mu, z_var).to_event(1))







"""

"""













