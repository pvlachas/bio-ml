"""
exec(open("tutorial_1.py").read())
"""

# Core scverse libraries
from __future__ import annotations

import anndata as ad

# Data retrieval
import pooch
import scanpy as sc

sc.settings.set_figure_params(dpi=50, facecolor="white")

"""
The data used in this basic preprocessing and clustering tutorial was collected from
bone marrow mononuclear cells of healthy human donors and was part of openproblem’s
NeurIPS 2021 benchmarking dataset [Luecken et al., 2021].
The samples used in this tutorial were measured using the
10X Multiome Gene Expression and Chromatin Accessability kit.
"""

EXAMPLE_DATA = pooch.create(
    path=pooch.os_cache("scverse_tutorials"),
    base_url="doi:10.6084/m9.figshare.22716739.v1/",
)
EXAMPLE_DATA.load_registry_from_doi()


samples = {
    "s1d1": "s1d1_filtered_feature_bc_matrix.h5",
    "s1d3": "s1d3_filtered_feature_bc_matrix.h5",
}
adatas = {}

for sample_id, filename in samples.items():
    path = EXAMPLE_DATA.fetch(filename)
    sample_adata = sc.read_10x_h5(path)
    sample_adata.var_names_make_unique()
    adatas[sample_id] = sample_adata

adata = ad.concat(adatas, label="sample")
adata.obs_names_make_unique()
print(adata.obs["sample"].value_counts())
print(adata)



# The data contains ~8,000 cells per sample and 36,601 measured genes.
# We’ll now investigate these with a basic preprocessing and clustering workflow.

"""
AnnData object with n_obs × n_vars = cells × genes
│
├── X                 (the main data matrix)
├── obs               (cell-level metadata)
├── var               (gene-level metadata)
├── layers            (additional data matrices)
├── obsm              (multi-dimensional embeddings, e.g. PCA, UMAP)
├── varm              (gene embeddings, e.g. PCA loadings)
├── uns               (unstructured info, e.g. clustering colors)
└── raw               (optional frozen copy of original data)
"""

# explanation of adata .obs:
"""
>>> adata.obs
                   sample
AAACCCAAGGATGGCT-1   s1d1
AAACCCAAGGCCTAGA-1   s1d1
AAACCCAAGTGAGTGC-1   s1d1
AAACCCACAAGAGGCT-1   s1d1
AAACCCACATCGTGGC-1   s1d1
...                   ...
TTTGTTGAGAGTCTGG-1   s1d3
TTTGTTGCAGACAATA-1   s1d3
TTTGTTGCATGTTACG-1   s1d3
TTTGTTGGTAGTCACT-1   s1d3
TTTGTTGTCGCGCTGA-1   s1d3

[17125 rows x 1 columns]

* each row -> one cell
"""

# Explanation of cell barcodes:
"""
* AAACCCAAGTGAGTGC-1:
cell barcode, unique identifier for each cell
used to index .obs, aligned with rows of .X
Row i of adata.X corresponds to cell barcode i in adata.obs.index.
such barcodes are technical identifiers, not biologically meaningful

In single-cell RNA-seq (like 10x Genomics),
thousands of cells are captured individually inside microdroplets.
Each droplet contains:
one cell (ideally),
one bead coated with oligonucleotides (short DNA sequences).
Each oligo has:
a cell barcode (e.g. TTTGTTGCAGACAATA),
a unique molecular identifier (UMI) (distinguishes molecules within the same cell),
and a poly-T tail to capture mRNA.

When the mRNA from that cell is reverse-transcribed,
every resulting cDNA molecule gets tagged with the same cell barcode
(because it came from the same droplet/bead).
Later, sequencing reads with that same barcode are grouped together —
that’s how we know all those transcripts came from the same cell.
TTTGTTGCAGACAATA-1
TTTGTTGCAGACAATA → the cell barcode, a 16-nucleotide sequence unique
(within that experiment) to a single bead/droplet.
-1 → an internal index (10x uses -1, -2, … if multiple libraries were merged).
These barcodes are assigned by the 10x library prep kit,
not by the cell’s genome or transcriptome.
They’re synthetic, not biologically derived.
"""

# explanation of adata .var:
"""
>>> adata.var
Empty DataFrame
Columns: []
Index: [MIR1302-2HG, FAM138A, OR4F5, AL627309.1, AL627309.3, AL627309.2, AL627309.5, AL627309.4, AP006222.2, AL732372.1, OR4F29, AC114498.1, OR4F16, AL669831.2, LINC01409, FAM87B, LINC01128, LINC00115, FAM41C, AL645608.6, AL645608.2, AL645608.4, LINC02593, SAMD11, NOC2L, KLHL17, PLEKHN1, PERM1, AL645608.7, HES4, ISG15, AL645608.1, AGRN, AL645608.5, AL645608.8, RNF223, C1orf159, AL390719.3, LINC01342, AL390719.2, TTLL10-AS1, TTLL10, TNFRSF18, TNFRSF4, SDF4, B3GALT6, C1QTNF12, AL162741.1, UBE2J2, LINC01786, SCNN1D, ACAP3, PUSL1, INTS11, AL139287.1, CPTP, TAS1R3, DVL1, MXRA8, AURKAIP1, CCNL2, MRPL20-AS1, MRPL20, AL391244.2, ANKRD65, AL391244.1, TMEM88B, LINC01770, VWA1, ATAD3C, ATAD3B, ATAD3A, TMEM240, SSU72, AL645728.1, FNDC10, AL691432.4, AL691432.2, MIB2, MMP23B, CDK11B, FO704657.1, SLC35E2B, CDK11A, SLC35E2A, NADK, GNB1, AL109917.1, CALML6, TMEM52, CFAP74, AL391845.2, GABRD, AL391845.1, PRKCZ, AL590822.2, PRKCZ-AS1, FAAP20, AL590822.1, SKI, ...]

[36601 rows x 0 columns]
>>>
These are the Genes measured in the dataset.
Each row corresponds to one gene (or transcript, depending on the reference used).
The index (adata.var.index) contains the gene symbols or transcript IDs.
The columns (adata.var.columns) would contain gene-level metadata,
but in this case, there are no additional annotations provided, so it’s empty.
"""


# explanation of adata.layers:
# explanation of adata.obsm:
# explanation of adata.varm:
# explanation of adata.uns:
# explanation of adata.raw:

# Quality control (QC) metrics
"""
In single-cell analysis, quality control (QC) metrics are summary statistics used to identify poor-quality cells or genes before downstream analysis (normalization, clustering, etc.).
Why QC metrics are needed?
Single-cell experiments (e.g., 10x Genomics) are noisy.
Some “cells” in the data might actually be:
- Dead cells (RNA leaking, degraded),
- Doublets (two cells captured together),
- Empty droplets (ambient RNA),
- Low-capture cells (very few reads),
- Cells with too much mitochondrial RNA (damaged membranes).
If you don’t remove these, your downstream steps (normalization, clustering, marker finding) become unreliable.
That’s why the first step after loading data is:
Compute QC metrics → filter out bad cells and genes.

UMIs, Unique Molecular Identifier:
Each unique UMI is interpreted as one original mRNA molecule for that gene in that cell.
It’s a short random DNA sequence (usually 8–12 nucleotides) that is attached to each 
individual mRNA molecule before amplification during single-cell RNA sequencing.
(may be included in multiple reads after PCR amplification).

Quality control metrics are provided in both adata.obs and adata.var after running sc.pp.calculate_qc_metrics().
adata.obs -> PER CELL METRICS
- total_counts: Total number of counts (UMIs, Unique Molecular Identifier) per cell.
- n_genes_by_counts: Number of genes detected (non-zero counts) per cell.
- pct_counts_<gene_set>: Percentage of counts from specific gene sets (e.g., mitochondrial genes) per cell.
adata.var -> PER GENE METRICS
- total_counts: Total number of counts (UMIs) per gene across all cells.
- n_cells_by_counts: Number of cells in which the gene is detected (non-zero counts).
- mean_counts: Mean expression level across all cells
"""

# calculating QC metrics
"""
The scanpy function calculate_qc_metrics() calculates common quality control (QC) metrics,
which are largely based on calculateQCMetrics from scater [McCarthy et al., 2017].
One can pass specific gene population to calculate_qc_metrics()
in order to calculate proportions of counts for these populations.
Mitochondrial, ribosomal and hemoglobin genes are defined by distinct prefixes as listed below.
"""

# why differentiating between mitochondrial, ribosomal, and hemoglobin genes ?
"""
Why we look at specific gene categories ?

- Mitochondrial genes: High percentage of mitochondrial gene expression often indicates damaged or stressed cells, as these cells may have compromised membranes leading to leakage of cytoplasmic RNA.
- Ribosomal genes: High ribosomal gene expression can indicate active protein synthesis, but excessively high levels may suggest technical artifacts or specific cell states.
- Hemoglobin genes: In datasets containing blood cells, high hemoglobin gene expression can indicate contamination from red blood cells or specific cell types like erythrocytes.
By calculating the percentage of counts from these gene categories,
we can identify and filter out low-quality cells or specific cell types that may confound downstream analyses

We look at biologically interpretable subsets of genes to detect problematic cells or 
unwanted cell types.

The independence is conceptual, not necessarily mathematical / computational.
You compute them separately because each one diagnoses a different biological or technical source of variation.

Why you don’t want to lump them together blindly:
If you combined them into one total (say, “% of mitochondrial + ribosomal + hemoglobin”), you would:
lose the ability to distinguish why a cell looks abnormal,
risk discarding biologically relevant cell types (e.g., erythroid cells or plasma cells) that should have high expression in one of those categories.
For example:
A dying T-cell might have high mitochondrial fraction.
A healthy plasma cell might have high ribosomal fraction.
A red blood cell will have huge hemoglobin fraction.
If you filter by the sum, you’d remove all three, even though only the first one (dying cell) is genuinely low-quality.
That’s why QC is done independently per gene category — to let you decide which biological signal to remove.
"""

"""
>>> adata.var
Empty DataFrame
Columns: []
Index: [MIR1302-2HG, FAM138A, OR4F5, AL627309.1, AL627309.3, AL627309.2, AL627309.5, AL627309.4, AP006222.2, AL732372.1, OR4F29, AC114498.1, OR4F16, AL669831.2, LINC01409, FAM87B, LINC01128, LINC00115, FAM41C, AL645608.6, AL645608.2, AL645608.4, LINC02593, SAMD11, NOC2L, KLHL17, PLEKHN1, PERM1, AL645608.7, HES4, ISG15, AL645608.1, AGRN, AL645608.5, AL645608.8, RNF223, C1orf159, AL390719.3, LINC01342, AL390719.2, TTLL10-AS1, TTLL10, TNFRSF18, TNFRSF4, SDF4, B3GALT6, C1QTNF12, AL162741.1, UBE2J2, LINC01786, SCNN1D, ACAP3, PUSL1, INTS11, AL139287.1, CPTP, TAS1R3, DVL1, MXRA8, AURKAIP1, CCNL2, MRPL20-AS1, MRPL20, AL391244.2, ANKRD65, AL391244.1, TMEM88B, LINC01770, VWA1, ATAD3C, ATAD3B, ATAD3A, TMEM240, SSU72, AL645728.1, FNDC10, AL691432.4, AL691432.2, MIB2, MMP23B, CDK11B, FO704657.1, SLC35E2B, CDK11A, SLC35E2A, NADK, GNB1, AL109917.1, CALML6, TMEM52, CFAP74, AL391845.2, GABRD, AL391845.1, PRKCZ, AL590822.2, PRKCZ-AS1, FAAP20, AL590822.1, SKI, ...]

[36601 rows x 0 columns]
>>>
"""

adata2 = adata.copy()

# mitochondrial genes, "MT-" for human, "Mt-" for mouse
adata.var["mt"] = adata.var_names.str.startswith("MT-")
# ribosomal genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")



"""
>>> adata.var
                mt   ribo     hb
MIR1302-2HG  False  False  False
FAM138A      False  False  False
OR4F5        False  False  False
AL627309.1   False  False  False
AL627309.3   False  False  False
...            ...    ...    ...
AC141272.1   False  False  False
AC023491.2   False  False  False
AC007325.1   False  False  False
AC007325.4   False  False  False
AC007325.2   False  False  False

[36601 rows x 3 columns]
>>>
"""


sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)



"""
>>> adata.var
                mt   ribo     hb  ...  pct_dropout_by_counts  total_counts  log1p_total_counts
MIR1302-2HG  False  False  False  ...              99.994161           1.0            0.693147
FAM138A      False  False  False  ...             100.000000           0.0            0.000000
OR4F5        False  False  False  ...             100.000000           0.0            0.000000
AL627309.1   False  False  False  ...              99.766423          40.0            3.713572
AL627309.3   False  False  False  ...              99.982482           3.0            1.386294
...            ...    ...    ...  ...                    ...           ...                 ...
AC141272.1   False  False  False  ...              99.982482           3.0            1.386294
AC023491.2   False  False  False  ...             100.000000           0.0            0.000000
AC007325.1   False  False  False  ...              99.994161           1.0            0.693147
AC007325.4   False  False  False  ...              99.351825         113.0            4.736198
AC007325.2   False  False  False  ...              99.994161           1.0            0.693147

[36601 rows x 9 columns]
>>>
"""

"""
setting qc_vars=["mt", "ribo", "hb"]
adds new columns to adata.obs, for example:
- total_counts_mt	total counts from mitochondrial genes
- pct_counts_mt	% of total counts that are mitochondrial
- total_counts_ribo, pct_counts_ribo	same for ribosomal
- total_counts_hb, pct_counts_hb	same for hemoglobin
So qc_vars adds extra per-cell summaries.
It doesn’t change your n_cells_by_counts, mean_counts, etc.
"""

sc.pp.calculate_qc_metrics(adata2, inplace=True, log1p=True)
# >>> adata2.var.columns
# Index(['mt', 'ribo', 'hb', 'n_cells_by_counts', 'mean_counts',
#        'log1p_mean_counts', 'pct_dropout_by_counts', 'total_counts',
#        'log1p_total_counts'],
#       dtype='object')

"""
Gene-level QC metrics added to adata.var:
- total_counts → Total UMI count for this gene across all cells. sum of counts across all cells
- mean_counts → Mean (average) UMI count for this gene across all cells. average 
expression count (including zeros). It is about EXPRESSION LEVEL, genetic average 
abundance. (HOW MUCH)
- n_cells_by_counts -> Number of cells in which this gene has a nonzero count, 
how many cells express this gene (UMI count > 0). This measures the extend in cells, 
but now how strongly the gene is expressed on average (different to mean_counts !). 
It is about PREVALENCE. (WHERE)
- log1p_mean_counts → Natural log of (1 + mean_counts). log-transformed mean counts
- log1p_total_counts → Natural log of (1 + total_counts). log-transformed total counts
- pct_dropout_by_counts → Percentage of cells with zero counts for this gene. 
proportion of cells that do not express this gene (zero UMI count). High dropout 
percentage indicates low expression or technical issues.
"""

"""
Cell-level QC metrics added to adata.obs:

"""
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
    save="_qc_violin.png",
    show=False,
)

"""
a count = one observed RNA molecule (or transcript)
This scatter plot provides an overview of single-cell quality metrics, with each dot representing one cell (or droplet).
The x-axis shows the total number of RNA molecules detected per cell (total_counts), while the y-axis indicates how many genes were detected in that cell (n_genes_by_counts).
The color scale encodes the percentage of mitochondrial transcripts (pct_counts_mt), which serves as an indicator of cell stress or damage.
In a typical dataset, most cells form a diagonal trend—cells with higher total counts also tend to express more genes, reflecting deeper sequencing or larger transcriptomes.
Cells appearing in the lower-left region, with few total counts and few genes, likely represent empty droplets or low-quality captures.
Conversely, cells with extremely high total counts and high gene numbers may be doublets—instances where multiple cells were captured together.
Cells with high mitochondrial percentages (brightly colored) often correspond to stressed or dying cells.
Overall, this plot is a key diagnostic tool for identifying and filtering out low-quality or anomalous cells before downstream analysis.
"""
sc.pl.scatter(
    adata,
    "total_counts",
    "n_genes_by_counts",
    color="pct_counts_mt",
    save="_qc_scatter.png",
    show=False,
)


"""
Based on the QC metric plots, one could now remove cells that have too many mitochondrial genes expressed or too many total counts by setting manual or automatic thresholds.
However, sometimes what appears to be poor QC metrics can be driven by real biology so we suggest starting with a very permissive filtering strategy and revisiting it at a later point.
We therefore now only filter cells with less than 100 genes expressed and genes that are detected in less than 3 cells.
"""


# 1. Filter genes: keep only those expressed in at least 3 cells
sc.pp.filter_genes(adata, min_cells=3)

# 2. Filter cells based on QC metrics

adata = adata[adata.obs["n_genes_by_counts"] < 10000, :]  # remove potential doublets
# == sc.pp.filter_cells(adata, max_genes=10000)
adata = adata[adata.obs["n_genes_by_counts"] > 100, :]    # remove low-complexity cells
# == sc.pp.filter_cells(adata, min_genes=100)    # remove low-complexity cells
adata = adata[adata.obs["pct_counts_mt"] < 5, :]          # remove stressed/dying cells


sc.pl.scatter(
    adata,
    "total_counts",
    "n_genes_by_counts",
    color="pct_counts_mt",
    save="_qc_scatter_filtered.png",
    show=False,
)

"""As a next step, we run a doublet detection algorithm. Identifying doublets is crucial as they can lead to misclassifications or distortions in downstream analysis steps. Scanpy contains the doublet detection method Scrublet [Wolock et al., 2019]. Scrublet predicts cell doublets using a nearest-neighbor classifier of observed transcriptomes and simulated doublets. scanpy.pp.scrublet() adds doublet_score and predicted_doublet to .obs. One can now either filter directly on predicted_doublet or use the doublet_score later during clustering to filter clusters with high doublet scores.
"""
sc.pp.scrublet(adata, batch_key="sample")




"""
The next preprocessing step is normalization.
A common approach is count depth scaling with subsequent log plus one (log1p) transformation.
Count depth scaling normalizes the data to a “size factor” such as the 
MEDIAN COUNT DEPTH
in the dataset, ten thousand (CP10k) or one million (CPM, counts per million).
The size factor for count depth scaling can be controlled via target_sum in pp.normalize_total.
We are applying median count depth normalization with log1p transformation (AKA log1PF).

The median count depth is a measure of sequencing depth, or more precisely, 
an indicator of how deeply each cell was sequenced.
In single-cell RNA-seq, sequencing depth refers to how many total reads (or UMIs) 
were captured and assigned to a given cell.
"""


# adata.obs["total_counts"] -> rows, cells, obs
# adata.var['total_counts'] -> genes, columns, vars

# Saving count data on the .layers attribute (auxiliary data matrices), so that we
# keep a copy of the raw counts before normalization
adata.layers["counts"] = adata.X.copy()

# mean of total counts across all cells
median_count_depth = int(adata.obs["total_counts"].median())

# Normalizing to median total counts
sc.pp.normalize_total(
    adata,
    target_sum=median_count_depth,
)
# Logarithmize the data
sc.pp.log1p(adata)


"""
As a next step, we want to reduce the dimensionality of the dataset and only include the most informative genes.
This step is commonly known as feature selection.
The scanpy function pp.highly_variable_genes annotates highly variable genes by reproducing the implementations of Seurat [Satija et al., 2015], Cell Ranger [Zheng et al., 2017], and Seurat v3 [Stuart et al., 2019] depending on the chosen flavor.
"""

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    batch_key="sample",
)

"""
Several new entries are added:

In adata.var:
- highly_variable -> Boolean flag (True for selected genes)
- means -> Mean expression of each gene
- variances -> Raw variance of each gene
- variances_norm -> Normalized/adjusted variance
- highly_variable_nbatches -> In how many batches the gene was found variable
- highly_variable_rank -> Rank of variability (lower = more variable)
"""

sc.pl.highly_variable_genes(
    adata,
    save="_highly_variable_genes.png",
    show=False,
)


"""
PCA
- Each cell as one data point (observation)
- Each gene as one feature (dimension)
-> Looking for combinations of gene expression patterns that explain most of the 
variance in the data !
-> PCA components of gene expression !
-> your axis (features) are expression levels of genes
-> PCA finds orthogonal directions in GENE space that best explain the variance 
across cells.

Let us inspect the contribution of single PCs to the total variance in the data.
This gives us information about how many PCs we should consider in order to compute the neighborhood relations of cells, e.g. used in the clustering function leiden() or tsne().
In our experience, there does not seem to be signifigant downside to overestimating the numer of principal components.
"""
sc.tl.pca(adata)
sc.pl.pca_variance_ratio(
    adata, n_pcs=50, log=True,
    save="_pca_variance_ratio.png",
    show=False,
)

sc.pl.pca(
    adata,
    color=["sample", "sample", "pct_counts_mt", "pct_counts_mt"],
    dimensions=[(0, 1), (2, 3), (0, 1), (2, 3)],
    ncols=2,
    size=2,
    show=False,
    save="_pca_1.png",
)

sc.pl.pca(
    adata,
    color=["pct_counts_mt"],
    show=False,
    save="_pca_pct_counts_mt.png",
)

"""Where are the gene loadings stored?
PCA also produces loadings, which tell you how strongly each gene contributes to each principal component.
adata.varm["PCs"]
with shape (n_genes, n_pcs).
- Each column corresponds to one principal component.
- Each row corresponds to one gene’s contribution to that component.
You can inspect or visualize these loadings to see which genes drive specific PCs.
"""
# check PCA loadings
sc.pl.pca(
    adata,
    color='pct_counts_mt',
    show=False,
    save="_pca_pct_counts_mt_loadings.png",
)
sc.pl.pca_loadings(adata, show=False, save="_pca_loadings.png", )





"""
Nearest neighbor graph construction and visualization
Next, we compute the neighborhood graph of cells based on the PCA representation.
sc.pp.neighbors(adata)
This graph can then be embedded in two dimensions for visualiztion with UMAP [McInnes et al., 2018]:

Even though the data considered in this tutorial includes two different samples,
we only observe a minor batch effect and we can continue with clustering and annotation of our data.
If you inspect batch effects in your UMAP it can be beneficial to integrate across samples and perform batch correction/integration.
We recommend checking out scanorama and scvi-tools for batch integration.
"""

sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(
    adata,
    color="sample",
    # Setting a smaller point size to get prevent overlap
    size=2,
    show=False,
    save="_umap.png",
)

"""
Clustering
As with Seurat and many other frameworks, we recommend the Leiden graph-clustering method
(community detection based on optimizing modularity) [Traag et al., 2019].
Note that Leiden clustering directly clusters the neighborhood graph of cells,
which we already computed in the previous section.
"""

# Using the igraph implementation and a fixed number of iterations can be significantly faster,
# especially for larger datasets
sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
sc.pl.umap(
    adata,
    color=["leiden"],
    show=False,
    save="_umap_leiden.png",
)




"""
Re-assess quality control and cell filtering
As indicated before, we will now re-assess our filtering strategy by visualizing different QC metrics using UMAP.


"""
sc.pl.umap(
    adata,
    color=["leiden", "predicted_doublet", "doublet_score"],
    # increase horizontal space between panels
    wspace=0.5,
    size=3,
    show=False,
    save="_umap_leiden_doublet.png",
)

sc.pl.umap(
    adata,
    color=["leiden", "log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts"],
    wspace=0.5,
    ncols=2,
    show=False,
    save="_umap_leiden_gene_features.png",
)


"""
Manual cell-type annotation

Cell type annotation is laborous and repetitive task, one which typically requires multiple rounds of subclustering and re-annotation.
It’s difficult to show the entirety of the process in this tutorial, but we aim to show how the tools scanpy provides assist in this process.

We have now reached a point where we have obtained a set of cells with decent quality, and we can proceed to their annotation to known cell types.
Typically, this is done using genes that are exclusively expressed by a given cell type, or in other words these genes are the marker genes of the cell types, and are thus used to distinguish the heterogeneous groups of cells in our data.
Previous efforts have collected and curated various marker genes into available resources, such as CellMarker, TF-Marker, and PanglaoDB.
The cellxgene gene expression tool can also be quite useful to see which cell types a gene has been expressed in across many existing datasets.

Commonly and classically, cell type annotation uses those marker genes subsequent to the grouping of the cells into clusters.
So, let’s generate a set of clustering solutions which we can then use to annotate our cell types.
Here, we will use the Leiden clustering algorithm which will extract cell communities from our nearest neighbours graph.
"""


for res in [0.02, 0.5, 2.0]:
    sc.tl.leiden(adata, key_added=f"leiden_res_{res:4.2f}", resolution=res, flavor="igraph")

"""
Notably, the number of clusters that we define is largely arbitrary, and so is the resolution parameter that we use to control for it.
As such, the number of clusters is ultimately bound to the stable and biologically-meaningful groups that we can ultimately distringuish, typically done by experts in the corresponding field or by using expert-curated prior knowledge in the form of markers.
Though UMAPs should not be over-interpreted, here we can already see that in the highest resolution our data is over-clustered, while the lowest resolution is likely grouping cells which belong to distinct cell identities.
"""

sc.pl.umap(
    adata,
    color=["leiden_res_0.02", "leiden_res_0.50", "leiden_res_2.00"],
    legend_loc="on data",
    show=False,
    save="_umap_leiden_resolutions.png",
)

"""
Let’s define a set of marker genes for the main cell types that we expect to see in this dataset.
These were adapted from Single Cell Best Practices annotation chapter, for a more detailed overview and best practices in cell type annotation, we refer the user to it.
"""

"""
So when you see something like "CD14+ Mono": ["FCN1", "CD14"],
it means: “Cells expressing the genes FCN1 and CD14 are likely CD14+ monocytes.”
"""
marker_genes = {
    "CD14+ Mono": ["FCN1", "CD14"],
    "CD16+ Mono": ["TCF7L2", "FCGR3A", "LYN"],
    # Note: DMXL2 should be negative
    "cDC2": ["CST3", "COTL1", "LYZ", "DMXL2", "CLEC10A", "FCER1A"],
    "Erythroblast": ["MKI67", "HBA1", "HBB"],
    # Note HBM and GYPA are negative markers
    "Proerythroblast": ["CDK6", "SYNGR1", "HBM", "GYPA"],
    "NK": ["GNLY", "NKG7", "CD247", "FCER1G", "TYROBP", "KLRG1", "FCGR3A"],
    "ILC": ["ID2", "PLCG2", "GNLY", "SYNE1"],
    "Naive CD20+ B": ["MS4A1", "IL4R", "IGHD", "FCRL1", "IGHM"],
    # Note IGHD and IGHM are negative markers
    "B cells": [
        "MS4A1",
        "ITGB1",
        "COL4A4",
        "PRDM1",
        "IRF4",
        "PAX5",
        "BCL11A",
        "BLK",
        "IGHD",
        "IGHM",
    ],
    "Plasma cells": ["MZB1", "HSP90B1", "FNDC3B", "PRDM1", "IGKC", "JCHAIN"],
    # Note PAX5 is a negative marker
    "Plasmablast": ["XBP1", "PRDM1", "PAX5"],
    "CD4+ T": ["CD4", "IL7R", "TRBC2"],
    "CD8+ T": ["CD8A", "CD8B", "GZMK", "GZMA", "CCL5", "GZMB", "GZMH", "GZMA"],
    "T naive": ["LEF1", "CCR7", "TCF7"],
    "pDC": ["GZMB", "IL3RA", "COBLL1", "TCF4"],
}

"""
Cells and interpretation:

"""

sc.pl.dotplot(
    adata,
    marker_genes,
    groupby="leiden_res_0.02",
    standard_scale="var",
    show=False,
    save="_dotplot_marker_genes_leiden_res_0.02.png",
)


sc.pl.dotplot(
    adata,
    marker_genes,
    groupby="leiden_res_0.50",
    standard_scale="var",
    show=False,
    save="_dotplot_marker_genes_leiden_res_0.5.png",
)



sc.pl.dotplot(
    adata,
    marker_genes,
    groupby="leiden_res_2.00",
    standard_scale="var",
    show=False,
    save="_dotplot_marker_genes_leiden_res_2.0.png",
)






