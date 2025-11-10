import pandas as pd
import numpy as np
import mudata as mu
from tqdm import tqdm
import sys
import os
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_cats, f_beta_score
import argparse


# Init args
parser = argparse.ArgumentParser()
parser.add_argument('-a','--grn_path', required=True)
parser.add_argument('-b','--resource_path', required=True)
parser.add_argument('-f','--out_path', required=True)
parser.add_argument('-s','--subset_path', required=False, default=None, help='Path to save cell-type-specific subset')
parser.add_argument('-c','--confusion_path', required=False, default=None, help='Path to save confusion matrix')
args = vars(parser.parse_args())

grn_path = args['grn_path']
resource_path = args['resource_path']
out_path = args['out_path']
subset_path = args['subset_path']
confusion_path = args['confusion_path']


grn_name = os.path.basename(grn_path).replace('.grn.csv', '')
data_path = os.path.join(os.path.dirname(os.path.dirname(grn_path)), 'mdata.h5mu')
dataset = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(data_path))))
case = os.path.basename(os.path.dirname(data_path))
resource_name = os.path.splitext(os.path.basename(resource_path))[0]

# Read grn
grn = pd.read_csv(grn_path)

if grn.shape[0] > 0:
    # Read resource and filter by cats
    db = pd.read_csv(resource_path, header=None, sep='\t')
    db.columns = ['gene', 'ctype']
    cats = load_cats(dataset, case)
    if resource_name in cats:
        cats = [re.escape(c) for c in cats[resource_name]]
        print('Filtering for {0} cats'.format(len(cats)))
        db = db[db['ctype'].str.contains('|'.join(cats))]
        
        # Save cell-type-specific subset if path provided
        if subset_path is not None and subset_path != '' and case != 'all':
            os.makedirs(os.path.dirname(subset_path), exist_ok=True)
            db.to_csv(subset_path, index=False)
            print(f'Saved cell-type-specific subset to {subset_path}')
    
    # Filter resource by measured genes
    genes = mu.read(os.path.join(data_path, 'mod', 'rna')).var_names.astype('U')
    db = db[db['gene'].astype('U').isin(genes)]
    
    # Compute evaluation
    y_pred = grn['source'].unique().astype('U')
    y = db['gene'].unique().astype('U')
    
    tp_genes = np.intersect1d(y_pred, y)
    fp_genes = np.setdiff1d(y_pred, y)
    fn_genes = np.setdiff1d(y, y_pred)
    
    tp = len(tp_genes)
    fp = len(fp_genes)
    fn = len(fn_genes)
    
    if tp > 0.:
        prc = tp / (tp + fp)
        rcl = tp / (tp + fn)
        f01 = f_beta_score(prc, rcl)
    else:
        prc, rcl, f01 = 0., 0., 0.
    
    # Save confusion matrix if path provided
    if confusion_path is not None:
        confusion_df = pd.DataFrame({
            'name': [grn_name],
            'tp': [tp],
            'fp': [fp],
            'fn': [fn],
            'tp_genes': ['|'.join(sorted(tp_genes)) if len(tp_genes) > 0 else ''],
            'fp_genes': ['|'.join(sorted(fp_genes)) if len(fp_genes) > 0 else ''],
            'fn_genes': ['|'.join(sorted(fn_genes)) if len(fn_genes) > 0 else '']
        })
        os.makedirs(os.path.dirname(confusion_path), exist_ok=True)
        confusion_df.to_csv(confusion_path, index=False)
        print(f'Saved confusion matrix to {confusion_path}')
    
    df = pd.DataFrame([[grn_name, prc, rcl, f01, tp, fp, fn]], 
                      columns=['name', 'prc', 'rcl', 'f01', 'tp', 'fp', 'fn'])
else:
    df = pd.DataFrame([[grn_name, np.nan, np.nan, np.nan, 0, 0, 0]], 
                      columns=['name', 'prc', 'rcl', 'f01', 'tp', 'fp', 'fn'])
    
    # Save empty confusion matrix if path provided
    if confusion_path is not None:
        confusion_df = pd.DataFrame({
            'name': [grn_name],
            'tp': [0],
            'fp': [0],
            'fn': [0],
            'tp_genes': [''],
            'fp_genes': [''],
            'fn_genes': ['']
        })
        os.makedirs(os.path.dirname(confusion_path), exist_ok=True)
        confusion_df.to_csv(confusion_path, index=False)

# Write
df.to_csv(out_path, index=False)
