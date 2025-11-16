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


def _safe_auc_inputs(y_true: np.ndarray, scores: np.ndarray):
    """Return y_true, scores after sorting by descending score.
    If y_true has <2 classes, return None to signal undefined AUC.
    """
    if y_true.ndim != 1 or scores.ndim != 1 or y_true.size != scores.size:
        raise ValueError("y_true and scores must be 1D arrays of equal length")
    # Need both classes present
    if np.unique(y_true).size < 2:
        return None
    order = np.argsort(-scores, kind="mergesort")  # stable
    return y_true[order], scores[order]


def _compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    prepared = _safe_auc_inputs(y_true, scores)
    if prepared is None:
        return np.nan
    y_sorted, s_sorted = prepared
    # Binary labels assumed 0/1
    pos = (y_sorted == 1).astype(float)
    neg = 1.0 - pos
    cum_pos = np.cumsum(pos)
    cum_neg = np.cumsum(neg)
    total_pos = cum_pos[-1]
    total_neg = cum_neg[-1]
    if total_pos == 0 or total_neg == 0:
        return np.nan
    tpr = cum_pos / total_pos
    fpr = cum_neg / total_neg
    # Insert (0,0) point for completeness
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    # Trapezoidal integration
    return float(np.trapz(tpr, fpr))


def _compute_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    prepared = _safe_auc_inputs(y_true, scores)
    if prepared is None:
        return np.nan
    y_sorted, _ = prepared
    pos = (y_sorted == 1).astype(float)
    cum_pos = np.cumsum(pos)
    idx = np.arange(1, y_sorted.size + 1, dtype=float)
    total_pos = cum_pos[-1]
    if total_pos == 0:
        return np.nan
    precision = cum_pos / idx
    recall = cum_pos / total_pos
    # Prepend (recall=0, precision=1) conventional start
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    # Integrate precision w.r.t recall (descending scores already applied)
    return float(np.trapz(precision, recall))


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
    
    # Compute evaluation (binary confusion metrics)
    y_pred = grn['source'].unique().astype('U')
    y = db['gene'].unique().astype('U')

    tp_genes = np.intersect1d(y_pred, y)
    fp_genes = np.setdiff1d(y_pred, y)
    fn_genes = np.setdiff1d(y, y_pred)

    tp = len(tp_genes)
    fp = len(fp_genes)
    fn = len(fn_genes)

    if tp > 0.:
        prc = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rcl = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f01 = f_beta_score(prc, rcl)
    else:
        prc, rcl, f01 = 0., 0., 0.

    # AUC metrics: derive continuous TF scores
    score_col = None
    for candidate in ('score', 'Score'):
        if candidate in grn.columns:
            score_col = candidate
            break
    # Universe: all genes considered for evaluation (union of y and y_pred)
    universe = np.unique(np.concatenate([y, y_pred])).astype('U')
    # Assign scores: for genes that appear as TF sources, use max edge score; else 0.
    if score_col is not None:
        tf_scores = grn.groupby('source')[score_col].max().to_dict()
        scores = np.array([tf_scores.get(g, 0.0) for g in universe], dtype=float)
        # Optional normalization (min-max) if variability exists
        s_min = scores.min()
        s_max = scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)
    else:
        scores = np.array([1.0 if g in y_pred else 0.0 for g in universe], dtype=float)
    labels = np.array([1 if g in y else 0 for g in universe], dtype=int)
    auroc = _compute_auroc(labels, scores)
    auprc = _compute_auprc(labels, scores)
    
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
    
    df = pd.DataFrame([[grn_name, prc, rcl, f01, auprc, auroc, tp, fp, fn]], 
                      columns=['name', 'prc', 'rcl', 'f01', 'auprc', 'auroc', 'tp', 'fp', 'fn'])
else:
    df = pd.DataFrame([[grn_name, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, 0]], 
                      columns=['name', 'prc', 'rcl', 'f01', 'auprc', 'auroc', 'tp', 'fp', 'fn'])
    
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
