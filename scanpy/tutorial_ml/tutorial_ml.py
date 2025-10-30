
"""
exec(open("tutorial_ml.py").read())

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
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from anndata.experimental.pytorch import AnnLoader
import os

sc.settings.figdir = "./figures"

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



adata = sc.read('./data/pancreas.h5ad')

"""
>>> adata.obs.head()
          batch            study             cell_type  size_factors
index
0-0-0-0-0     0  Pancreas inDrop  Pancreas Endothelial       13073.0
1-0-0-0-0     0  Pancreas inDrop       Pancreas Acinar       17227.0
2-0-0-0-0     0  Pancreas inDrop       Pancreas Acinar        7844.0
3-0-0-0-0     0  Pancreas inDrop       Pancreas Acinar       10966.0
4-0-0-0-0     0  Pancreas inDrop  Pancreas Endothelial        8815.0
>>> adata.var.head()
Empty DataFrame
Columns: []
Index: [COL3A1, CPA2, SPP1, SPARC, PNLIP]
>>> adata.uns.keys()
dict_keys(['pca', 'neighbors', 'umap'])
>>> adata.X.shape
(15681, 1000)
>>> adata.X[:5, :5]
array([[0.       , 9.108002 , 0.       , 0.       , 9.689126 ],
       [0.       , 8.77993  , 0.       , 0.       , 7.9117913],
       [0.       , 8.632374 , 0.       , 0.       , 7.151376 ],
       [0.       , 8.068617 , 0.       , 0.       , 7.152796 ],
       [0.       , 8.738794 , 0.       , 0.       , 8.602685 ]],
      dtype=float32)
>>>
"""

# let's check sparsity
nnz = adata.X > 0
nnz = np.sum(nnz)
N, G = adata.X.shape
nnz_percent = nnz / (N * G)
print(f"Number of nonzero elements: {nnz_percent*100:.2f} %")

# quick quality control and summaries
# compute number of genes per cell
temp = adata.X > 0
temp = temp.sum(axis=1)
adata.obs['n_genes'] = temp

# Count total counts per cell:
temp = adata.X.sum(axis=1)
adata.obs['n_counts'] = temp

# plot summaries
sc.pl.violin(
    adata,
    ["n_genes", "n_counts"],
    jitter=0.4,
    multi_panel=True,
    show=False,
    save='violin_genes_counts.png',
)
plt.close()

# Exploring the categorical variables
adata.obs["batch"].value_counts()
adata.obs["cell_type"].value_counts()

sc.pl.violin(
    adata,
    keys="n_genes",
    groupby="batch",
    show=False,
    save='violin_genes_group_by_batch.png',
)
plt.close()
sc.pl.violin(
    adata,
    keys="n_genes",
    groupby="cell_type",
    show=False,
    save='violin_genes_group_by_cell_type.png',
)
plt.close()

"""
>>> adata.obs["batch"].value_counts()
batch
0    12720
1     2961
Name: count, dtype: int64
>>> adata.obs["cell_type"].value_counts()
cell_type
Pancreas Beta           5085
Pancreas Alpha          4704
Pancreas Ductal         2104
Pancreas Delta          1041
Pancreas Endothelial     836
Pancreas Acinar          713
Pancreas Gamma           637
Pancreas Stellate        561
Name: count, dtype: int64
>>>
"""

# check expression of a few marker genes
marker_genes = ["SST", "PPY", "PRSS1", "CPA1"]
# set use_raw to false to plot normalized and log-transformed expression,
# not raw counts.
sc.pl.violin(
    adata,
    keys=marker_genes,
    groupby="cell_type",
    rotation=45,
    show=False,
    save='violin_marker_genes_group_by_cell_type.png',
    use_raw=False,
)
plt.close()

# keep only "SST" gene, and cell_type "Pancreas Delta"

adata_selection = adata[
    adata.obs["cell_type"] == "Pancreas Delta",
    adata.var_names == "SST",
]
vals = adata_selection.X.flatten()

# adata_selection = adata[
#     adata.obs["cell_type"] == "Pancreas Delta",
#     adata.var_names == "SST",
# ].raw
# vals = adata_selection.X.flatten()

# VALIDATION
fig, ax = plt.subplots(figsize=(4, 5))
parts = ax.violinplot(
    vals,
    showmeans=False,
    showmedians=True,   # Scanpy shows median
    showextrema=True    # whiskers
)
ax.set_title("SST expression in Pancreas Delta cells")
ax.set_ylabel("Expression value")
ax.set_xticks([1])
ax.set_xticklabels(["Pancreas Delta"])
plt.savefig('figures/SST_PancreasDelta.png')
plt.close()


"""
>>> adata_selection
View of AnnData object with n_obs × n_vars = 1041 × 1
    obs: 'batch', 'study', 'cell_type', 'size_factors', 'n_genes', 'n_counts'
    var: 'n_counts'
    uns: 'pca', 'neighbors', 'umap', 'batch_colors', 'cell_type_colors'
    obsm: 'X_pca', 'X_umap'
    varm: 'PCs'
    obsp: 'distances', 'connectivities'
>>>
"""

sc.pl.dotplot(
    adata,
    marker_genes,
    groupby="cell_type",
    show=False,
    save='dotplot_marker_genes_group_by_cell_type.png',
    use_raw=False,
)
plt.close()

# Highly variable genes (HVGs) identification
# using local copy to avoid modifying original adata
# if original data are modified, subsequent UMAP and PCA computation are computed only on HVGs
adata_local = adata.copy()
sc.pp.highly_variable_genes(adata_local, flavor="seurat_v3", n_top_genes=20)
print(f"Number of HVGs: {adata_local.var['highly_variable'].sum()}")
hvgenes = adata_local.var[adata_local.var['highly_variable']].sort_values('highly_variable_rank').head(20)
print("Top 20 HVGs: {}".format(hvgenes.index.tolist()))

"""
These make biological sense for pancreas datasets:
- C1QA / C1QC / VSIG4 / MPEG1 / FPR3 → macrophage / immune-associated genes
- CD2 / CD52 / CD69 / CCR7 → T-cell–related markers
- KRT1 / SOX10 → epithelial or neural crest lineage genes
- HBB → hemoglobin β, can appear due to contaminating red blood cells
- HDC → histidine decarboxylase, histamine-producing endocrine cells
- HPGDS → prostaglandin synthase, can mark immune or stromal cells
This pattern shows that your dataset likely contains diverse cell types
(immune, epithelial, endocrine), and the HVGs are capturing that heterogeneity well.
"""

sc.pl.violin(
    adata,
    keys=["C1QA", "CD2", "KRT1", "HBB"],
    groupby="cell_type",
    rotation=45,
    show=False,
    save='violin_HVG_genes_group_by_cell_type.png',
    use_raw=False,
)


# # PCA exploration
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# sc.pp.scale(adata)
# sc.tl.pca(adata, svd_solver='arpack')
#
# # plot components
# sc.pl.pca(
#     adata,
#     color=["batch", "cell_type"],
#     components=['1, 2'],
#     show=False,
#     save='PCA_batch_cell_type.png',
# )
# plt.close()
#
# sc.pl.pca_loadings(
#     adata,
#     components=[1, 2],
#     show=False,
#     save='PCA_components.png',
# )
# plt.close()



# sc.pp.neighbors(adata, n_pcs=10)
# sc.tl.umap(adata)
#
# sc.pl.umap(
#     adata,
#     color=['study', 'cell_type'],
#     wspace=0.35,
#     show=False,
#     save='umap_by_cell_type.png',
#     use_raw=False,
# )
# plt.close()

adata.X = adata.raw.X # put raw counts to .X

"""For our model we need size factors (library sizes) for each cell for the means of negative binomial reconstruction loss.
We also need to encode batch and cell-type labels as one-hot vectors."""

adata.obs['size_factors'] = adata.X.sum(1)

"""
Here we set up the encoders for labels in our AnnData object.
These encoders will be used by AnnLoader to convert the labels on the fly when they are accessed from the dataloader during the training phase.
"""

print("study categories: ")
print(list(adata.obs['study'].cat.categories))
# ['Pancreas CelSeq', 'Pancreas CelSeq2', 'Pancreas Fluidigm C1', 'Pancreas SS2', 'Pancreas inDrop']

encoder_study = OneHotEncoder(dtype=np.float32)
encoder_study.fit(adata.obs['study'].to_numpy()[:, None])

temp = np.array(
    ['Pancreas inDrop', 'Pancreas CelSeq2', 'Pancreas inDrop', 'Pancreas CelSeq2']
)
temp = temp[:, None]
print("One-hot encoded 'study' labels:")
print(encoder_study.transform(temp).toarray())

"""
>>> print(encoder_study.transform(temp).toarray())
[[0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0.]]
>>>
"""


print("Cell types: ")
print(list(adata.obs['cell_type'].cat.categories))

encoder_celltype = LabelEncoder()
encoder_celltype.fit(adata.obs['cell_type'])

# test the propagation
temp = np.array(
    ['Pancreas Acinar', 'Pancreas Alpha', 'Pancreas Alpha', 'Pancreas Beta']
)
print("One-hot encoded 'study' labels:")
print(encoder_celltype.transform(temp))
"""
>>> print(encoder_celltype.transform(temp))
[0 1 1 2]
>>>


"""

"""
You can create the converter with a function or a Mapping of functions which will be applied to the values of attributes (.obs, .obsm, .layers, .X) or to specific keys of these attributes in the subset object.
Specify an attribute and a key (if needed) as keys of the passed Mapping and a function to be applied as a value.
Here we define a converter which will transform the values of the keys 'study' and 'cell_type' of .obs using the encoders created above.
"""

encoders = {
    'obs': {
        'study': lambda s: encoder_study.transform(s.to_numpy()[:, None]),
        'cell_type': encoder_celltype.transform
    }
}

"""
Here we create an AnnLoader object which will return a PyTorch dataloader properly set for our AnnData object.

The use_cuda parameter indicates that we want to lazily convert all numeric values in the AnnData object.
By lazy conversion we mean that no data is converted until you access it.
The AnnLoader object creates a wrapper object from the provided AnnData object and it takes care about subsetting and conversion. No copying of the full AnnData object happens here.

The encoders passed to convert are applied before sending anything to cuda.
"""

dataloader = AnnLoader(
    adata,
    batch_size=128,
    shuffle=True,
    convert=encoders,
)
"""
>>> dataloader.dataset
AnnCollection object with n_obs × n_vars = 15681 × 1000
  constructed from 1 AnnData objects
    view of obsm: 'X_pca', 'X_umap'
    obs: 'batch', 'study', 'cell_type', 'size_factors', 'n_genes', 'n_counts'
>>>
"""

"""
view of obsm means that the wrapper object doesn’t copy anything from .obsm of the 
underlying AnnData object .obs
insted of view of obs means that the object copied .obs from the AnnData object.
You can configure what is copied, please see the AnnCollection tutorial for deatils.
The wrapper object (AnnCollection) never copies the full .X or .layers from the underlying AnnData object, all conversions happen when the AnnCollection object is subset.
"""

batch = dataloader.dataset[:10]

# print('X:', batch.X.device, batch.X.dtype)
# print('X_pca:', batch.obsm['X_pca'].device, batch.obsm['X_pca'].dtype)
# print('X_umap:', batch.obsm['X_umap'].device, batch.obsm['X_umap'].dtype)
# # and here you can see that the converters are applied to 'study' and 'cell_type'.
# print('study:', batch.obs['study'].device, batch.obs['study'].dtype)
# print('cell_type:', batch.obs['cell_type'].device, batch.obs['cell_type'].dtype)

"""
X: cuda:0 torch.float32
X_pca: cuda:0 torch.float32
X_umap: cuda:0 torch.float32
study: cuda:0 torch.float32
cell_type: cuda:0 torch.int32
"""

"""
You can also use a custom sampler instead of the default one with automatic bacthing in AnnLoader.
Just pass your sampler and batch_size.
"""

# from torch.utils.data import WeightedRandomSampler
# weights = np.ones(adata.n_obs)
# weights[adata.obs['cell_type'] == 'Pancreas Stellate'] = 2.
# sampler = WeightedRandomSampler(weights, adata.n_obs)
# dataloader = AnnLoader(adata, batch_size=128, sampler=sampler, convert=encoders)

# Initialize and train the model

n_conds = len(adata.obs['study'].cat.categories)
n_classes = len(adata.obs['cell_type'].cat.categories)
cvae = CVAE(
    adata.n_vars,
    n_conds=n_conds,
    n_classes=n_classes,
    hidden_dims=[128, 128],
    latent_dim=10,
)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")
cvae.to(device)

optimizer = pyro.optim.Adam({"lr": 1e-3})
svi = pyro.infer.SVI(
    cvae.model,
    cvae.guide,
    optimizer,
    loss=pyro.infer.TraceMeanField_ELBO(),
)


# for batch in dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     output = model(batch)


"""
Here is the code for our training phase.
The AnnLoader object is passed as a dataloader, it iterates
through dataloader.dataset (as in a standard PyTorch dataloader).

Note that now you can simply take a batch from the dataloader,
select a required attribute, do something with it if needed and pass to your loss 
function.
Everything is already converted by the pre-defined converters.
You don’t need to copy your AnnData object, you don’t need a custom dataloader for a 
dictionary of required keys, all observation keys are already in the batches.
"""

from tqdm import tqdm
import numpy as np
import time
import torch

# Detect device automatically (CUDA → MPS → CPU)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"🔥 Using device: {device.upper()}")

def train_epoch(svi, dataloader, device):
    """Train for one epoch and return average loss."""
    epoch_loss = 0.0

    for batch in dataloader:
        # Move tensors to the appropriate device
        x = batch.X.to(device)
        study = batch.obs["study"].to(device)
        cell_type = batch.obs["cell_type"].to(device)
        size_factors = batch.obs["size_factors"].to(device)

        # One optimization step
        loss = svi.step(x, study, cell_type, size_factors)
        epoch_loss += loss

    # Average over number of cells in the dataset
    total_loss = epoch_loss / len(dataloader.dataset)
    return total_loss



from tqdm import tqdm
import numpy as np
import time
import torch

def train_model(svi, dataloader, num_epochs=200, early_stop=False, patience=10):
    """Train CVAE for multiple epochs with tqdm progress bar and optional early stopping."""
    best_loss = np.inf
    patience_counter = 0
    losses = []

    start_time = time.time()
    progress_bar = tqdm(
        range(1, num_epochs + 1),
        desc="🚀 Training CVAE",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar:30}{r_bar}",
    )

    for epoch in progress_bar:
        loss = train_epoch(svi, dataloader, device)
        losses.append(loss)

        # Update tqdm postfix live
        progress_bar.set_postfix({"avg_loss": f"{loss:.4f}"})

        # Periodic log print
        if epoch % 20 == 0 or epoch == num_epochs:
            tqdm.write(f"[Epoch {epoch:03d}] 🧠 Avg training loss: {loss:.4f}")

        # Early stopping
        if early_stop:
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(f"⏹ Early stopping at epoch {epoch}. Best loss: {best_loss:.4f}")
                    break

    duration = time.time() - start_time
    tqdm.write(f"✅ Training finished in {duration/60:.2f} min. Final loss: {losses[-1]:.4f}")
    return losses



losses = train_model(svi, dataloader, num_epochs=20, early_stop=False)
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Average training loss")
plt.title("CVAE Training Loss")
plt.tight_layout()
plt.savefig("figures/training_loss_curve.png")
plt.close()
SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)
# 1️⃣ Save model weights
model_path = os.path.join(SAVE_DIR, "cvae_model.pt")
torch.save(cvae.state_dict(), model_path)
print(f"💾 Model saved to {model_path}")


# load trained model
SAVE_DIR = "./results"
MODEL_PATH = "./results/cvae_model.pt"
cvae.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# build dataloader WITHOUT shuffling
dataloader_eval = AnnLoader(
    adata,
    batch_size=128,
    shuffle=False,
    convert=encoders,
)

# 2️⃣ Encode the dataset to get latent representations (Z)
cvae.eval()
all_z = []
with torch.no_grad():
    for batch in dataloader_eval:
        x = batch.X.to(device)
        batches = batch.obs["study"].to(device)
        z_loc_scale = cvae.encoder(x, batches)
        z_mu = z_loc_scale[:, :cvae.latent_dim]
        all_z.append(z_mu.cpu().numpy())

latent_Z = np.concatenate(all_z, axis=0)

# 3️⃣ Store latent representations in adata.obsm
adata.obsm["X_cvae"] = latent_Z
print("🧠 Latent embeddings X_cvae stored in adata.obsm['X_cvae'].")

# 4️⃣ (Optional) Compute UMAP of the latent space
sc.pp.neighbors(adata, use_rep="X_cvae")
sc.tl.umap(adata)
sc.pl.umap(
    adata,
    color=["cell_type", "study"],
    save='CVAE_latent_u_map.png',
    wspace=0.35,
    show=False,
)
print("🎨 UMAP of latent space saved to ./figures/")

# 5️⃣ Save the updated AnnData object
adata_path = os.path.join(SAVE_DIR, "pancreas_cvae_results.h5ad")
adata.write(adata_path)
print(f"✅ Full AnnData object saved to {adata_path}")

# compute accuracy of cell-type prediction from latent space
cvae_class = cvae.classifier(torch.tensor(latent_Z, device=device))
cvae_class = cvae_class.argmax(dim=-1)
cvae_class = cvae_class.cpu().numpy()

true_class = encoder_celltype.transform(adata.obs["cell_type"])

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

acc = accuracy_score(true_class, cvae_class)
bal_acc = balanced_accuracy_score(true_class, cvae_class)
f1_macro = f1_score(true_class, cvae_class, average="macro")  # equally weights each class
f1_weighted = f1_score(true_class, cvae_class, average="weighted")  # accounts for class freq

print(f"Accuracy: {acc:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Macro F1: {f1_macro:.4f}")
print(f"Weighted F1: {f1_weighted:.4f}")

