import os
import scanpy as sc
import snapatac2 as snap
from snapatac2.datasets import _datasets, datasets
from pathlib import Path
import pandas as pd
import numpy as np
import anndata as ad
import mudata as md
import argparse


# Init args
parser = argparse.ArgumentParser()
parser.add_argument('-a','--path_tmp', required=True)
parser.add_argument('-b','--path_annot', required=True)
parser.add_argument('-c','--path_geneids', required=True)
parser.add_argument('-d','--organism', required=True)
parser.add_argument('-e','--path_peaks', required=True)
parser.add_argument('-f','--path_output', required=True)
parser.add_argument('-g','--path_multi', required=True)
args = vars(parser.parse_args())

path_tmp = args['path_tmp']
path_annot = args['path_annot']
path_geneids = args['path_geneids']
organism = args['organism']
path_peaks = args['path_peaks']
path_output = args['path_output']
path_multi = args['path_multi']

# Read gene ids
path_geneids = os.path.join(path_geneids, organism + '.csv')
geneids = pd.read_csv(path_geneids).set_index('symbol')['id'].to_dict()

# Change default cache dir
if not os.path.exists(path_tmp):
    os.mkdir(path_tmp)
_datasets = datasets()
_datasets.path = Path(path_tmp)

# Read annots
obs = pd.read_csv(path_annot, index_col=0)

# Read data
rna = sc.read_10x_h5(path_multi, genome="GRCh38", gex_only=True)
del rna.obs
rna.var.index.name = None

# Rename barcodes RNA
sample_id = 'smpl'
rna.obs_names = [sample_id + '_' + o.split('-1')[0] for o in rna.obs_names]

# Filter faulty gene symbols
ensmbls = np.array([geneids[g] if g in geneids else '' for g in rna.var_names])
msk = ensmbls != ''
rna = rna[:, msk].copy()

# Basic QC
sc.pp.filter_cells(rna, min_genes=100)
sc.pp.filter_genes(rna, min_cells=3)
del rna.obs['n_genes']

# Remove duplicated genes based on num of cells
to_remove = []
for dup in rna.var.index[rna.var.index.duplicated()]:
    tmp = rna.var.loc[dup]
    max_idx = tmp.set_index('gene_ids')['n_cells'].idxmax()
    to_remove.extend(tmp['gene_ids'][tmp['gene_ids'] != max_idx].values)
rna = rna[:, ~rna.var['gene_ids'].isin(to_remove)].copy()
del rna.var

# Read atac data
atac = ad.read_h5ad(path_peaks)

# Filter annotation and RNA data based on ATAC barcodes
obs = obs[obs.index.isin(atac.obs_names)]
rna = rna[rna.obs_names.isin(atac.obs_names)]

# Add celltype annotation to ATAC data
rna.obs['celltype'] = obs['celltype']
rna.obs['batch'] = obs['batch']
rna.obs_names = obs.index

atac.obs['celltype'] = obs['celltype']
atac.obs['batch'] = obs['batch']
atac.obs_names = obs.index

# Create mdata
mdata = md.MuData(
    {'rna': rna, 'atac': atac,},
    obs=obs
)

# Write
mdata.write(path_output)
