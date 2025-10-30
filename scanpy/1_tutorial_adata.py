"""
exec(open("adata.py").read())
"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
print(ad.__version__)


"""
Let’s start by building a basic AnnData object with some sparse count information, perhaps representing gene expression counts.

"""

counts = csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)
adata = ad.AnnData(counts)
# AnnData object with n_obs × n_vars = 100 × 2000
"""
We see that AnnData provides a representation with summary stastics of the data The initial data we passed are accessible as a sparse matrix using adata.X.
"""

adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
print(adata.obs_names[:10])

"""
These index values can be used to subset the AnnData, which provides a view of the AnnData object. We can imagine this to be useful to subset the AnnData to particular cell types or gene modules of interest. The rules for subsetting AnnData are quite similar to that of a Pandas DataFrame. You can use values in the obs/var_names, boolean masks, or cell index integers.
"""

print(adata[["Cell_1", "Cell_10"], ["Gene_5", "Gene_1900"]])

# View of AnnData object with n_obs × n_vars = 2 × 2

"""So we have the core of our object and now we’d like to add metadata at both the observation and variable levels. This is pretty simple with AnnData, both adata.obs and adata.var are Pandas DataFrames.
"""
ct = np.random.choice(["B", "T", "Monocyte"], size=(adata.n_obs,))
adata.obs["cell_type"] = pd.Categorical(ct)  # Categoricals are preferred for efficiency

"""
>>> adata.obs
        cell_type
Cell_0          T
Cell_1          T
Cell_2          B
Cell_3          T
Cell_4   Monocyte
...           ...
Cell_95  Monocyte
Cell_96         B
Cell_97  Monocyte
Cell_98         B
Cell_99  Monocyte

[100 rows x 1 columns]
"""


"""
Subsetting using metadata
We can also subset the AnnData using these randomly generated cell types:
"""

mono_data = adata[adata.obs.cell_type == "Monocyte"]

"""
>>> mono_data
View of AnnData object with n_obs × n_vars = 31 × 2000
    obs: 'cell_type'
>>> adata
AnnData object with n_obs × n_vars = 100 × 2000
    obs: 'cell_type'
"""


"""
Observation/variable-level matrices
We might also have metadata at either level that has many dimensions to it, such as a UMAP embedding of the data.
For this type of metadata, AnnData has the .obsm/.varm attributes.
We use keys to identify the different matrices we insert.
The restriction of .obsm/.varm are that .obsm matrices must length equal to the number of observations as .n_obs and .varm matrices must length equal to .n_vars.
They can each independently have different number of dimensions.

Let’s start with a randomly generated matrix that we can interpret as a UMAP embedding of the data we’d like to store, as well as some random gene-level metadata:
"""

adata.obsm["X_umap"] = np.random.normal(0, 1, size=(adata.n_obs, 2))
adata.varm["gene_feat_matrix"] = np.random.normal(0, 1, size=(adata.n_vars, 5))

"""
>>> adata.obsm
AxisArrays with keys: X_umap
>>> adata
AnnData object with n_obs × n_vars = 100 × 2000
    obs: 'cell_type'
    obsm: 'X_umap'
    varm: 'gene_feat_matrix'

A few more notes about .obsm/.varm

The “array-like” metadata can originate from a Pandas DataFrame, scipy sparse matrix, or numpy dense array.

When using scanpy, their values (columns) are not easily plotted, where instead items from .obs are easily plotted on, e.g., UMAP plots.
"""


"""
Unstructured metadata

adata.uns["random"] = [1, 2, 3]
adata.uns

"""

"""
Layers
Finally, we may have different forms of our original core data, perhaps one that is normalized and one that is not.
These can be stored in different layers in AnnData. For example, let’s log transform the original data and store it in a layer:
"""


adata.layers["log_transformed"]  = np.log1p(adata.X)

"""
>>> adata.layers['log_transformed']
<100x2000 sparse matrix of type '<class 'numpy.float32'>'
	with 126585 stored elements in Compressed Sparse Row format>
>>>
"""

"""
Conversion to DataFrames
We can also ask AnnData to return us dataframes.
We see that the .obs_names/.var_names are used in the creation of this Pandas object.
"""
print(adata.to_df())
print(adata.to_df(layer="log_transformed"))


"""Writing the results to disk
AnnData comes with its own persistent HDF5-based file format: h5ad.
If string columns with small number of categories aren’t yet categoricals, AnnData will auto-transform to categoricals.
"""

adata.write('my_results.h5ad', compression="gzip")

""""
Check results from the terminal using h5ls:

 ~/research/repos/bio-ml/scanpy  main ?2  h5ls my_results.h5ad                                                                    ✔
X                        Group
layers                   Group
obs                      Group
obsm                     Group
obsp                     Group
uns                      Group
var                      Group
varm                     Group
varp                     Group
 ~/research/repos/bio-ml/scanpy  main ?2                                                                                          ✔
"""

"""
Views and copies
For the fun of it, let’s look at another metadata use case.
Imagine that the observations come from instruments characterizing 10 readouts in a multi-year study with samples taken from different subjects at different sites.
We’d typically get that information in some format and then store it in a DataFrame:
"""

obs_meta = pd.DataFrame({
        'time_yr': np.random.choice([0, 2, 4, 8], adata.n_obs),
        'subject_id': np.random.choice(['subject 1', 'subject 2', 'subject 4', 'subject 8'], adata.n_obs),
        'instrument_type': np.random.choice(['type a', 'type b'], adata.n_obs),
        'site': np.random.choice(['site x', 'site y'], adata.n_obs),
        'cell_type': np.random.choice(['T', 'B', 'Monocyte'], adata.n_obs),
    },
    index=adata.obs.index,    # these are the same IDs of observations as above!
)

# let's make a adata object

adata_new = ad.AnnData(
    adata.X,
    obs=obs_meta,
    var=adata.var,
)

"""
>>>
>>> adata_new = ad.AnnData(
...     adata.X,
...     obs=obs_meta,
...     var=adata.var,
... )
>>> print(adata)
AnnData object with n_obs × n_vars = 100 × 2000
    obs: 'cell_type'
    uns: 'random'
    obsm: 'X_umap'
    varm: 'gene_feat_matrix'
    layers: 'log_transformed'
>>> print(adata_new)
AnnData object with n_obs × n_vars = 100 × 2000
    obs: 'time_yr', 'subject_id', 'instrument_type', 'site', 'cell_type'
>>>
"""


"""
Subsetting the joint data matrix can be important to focus on subsets of variables or observations, or to define train-test splits for a machine learning model.

Similar to numpy arrays, AnnData objects can either hold actual data or reference another AnnData object. In the later case, they are referred to as “view”.
Subsetting AnnData objects always returns views, which has two advantages:
no new memory is allocated
it is possible to modify the underlying AnnData object

You can get an actual AnnData object from a view by calling .copy() on the view. Usually, this is not necessary, as any modification of elements of a view (calling .[] on an attribute of the view) internally calls .copy() and makes the view an AnnData object that holds actual data. See the example below.
"""

"""
Get access to the first 5 rows for two variables.
Indexing into AnnData will assume that integer arguments to [] behave like .iloc in pandas, whereas string arguments behave like .loc. AnnData always assumes string indices.
"""


temp = adata[:10, ["Gene_10", "Gene_18", "Gene_100"]] # VIEW
temp = adata[:10, ["Gene_10", "Gene_18", "Gene_100"]].copy()
"""
View of AnnData object with n_obs × n_vars = 10 × 3
    obs: 'cell_type'
    uns: 'random'
    obsm: 'X_umap'
    varm: 'gene_feat_matrix'
    layers: 'log_transformed'
"""


# changing the VIEW -> CHANGES THE ORIGNAL DATA !
print(adata[:3, 'Gene_1'].X.toarray().tolist())
adata[:3, 'Gene_1'].X = [0, 0, 0]
print(adata[:3, 'Gene_1'].X.toarray().tolist())

""""
>> print(adata[:3, 'Gene_1'].X.toarray().tolist())
[[1.0], [0.0], [1.0]]
>>> adata[:3, 'Gene_1'].X = [0, 0, 0]
>>> print(adata[:3, 'Gene_1'].X.toarray().tolist())
[[0.0], [0.0], [0.0]]
"""

# if instead you do:
adata_subset = adata[:3, 'Gene_1']
"""
>>> adata_subset.to_df()
        Gene_1
Cell_0     0.0
Cell_1     0.0
Cell_2     0.0
>>>
"""
adata_subset.X[1] = 2.
adata_subset.obs['foo'] = range(3)
"""
>>> adata_subset.to_df()
        Gene_1
Cell_0     0.0
Cell_1     2.0
Cell_2     0.0
>>> print(adata_subset)
AnnData object with n_obs × n_vars = 3 × 1
    obs: 'cell_type', 'foo'
    uns: 'random'
    obsm: 'X_umap'
    varm: 'gene_feat_matrix'
    layers: 'log_transformed'
>>>
"""


"""
Partial reading of large data
If a single .h5ad is very large, you can partially read it into memory by using backed mode:
"""


# adata = ad.read('my_results.h5ad', backed='r')
adata = ad.read_h5ad('my_results.h5ad', backed='r')

"""
>>> adata = ad.read_h5ad('my_results.h5ad', backed='r')
>>> adata.isbacked
True
>>> adata.filename
PosixPath('my_results.h5ad')
>>> adata.file.close()
>>>
"""

"""
If you do this, you’ll need to remember that the AnnData object has an open connection to the file used for reading:

adata.filename
PosixPath('my_results.h5ad')
As we’re using it in read-only mode, we can’t damage anything. To proceed with this tutorial, we still need to explicitly close it:

adata.file.close()
As usual, you should rather use with statements to avoid dangling open files (up-coming feature).

Manipulating the object on disk is possible, but experimental for sparse data. Hence, we leave it out of this tutorial.
"""















