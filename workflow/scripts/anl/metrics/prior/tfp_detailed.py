from itertools import combinations
import scipy.stats as ss
import numpy as np
import pandas as pd
import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import f_beta_score


def _safe_auc_inputs(y_true: np.ndarray, scores: np.ndarray):
    if y_true.ndim != 1 or scores.ndim != 1 or y_true.size != scores.size:
        raise ValueError("y_true and scores must be 1D arrays of equal length")
    if np.unique(y_true).size < 2:
        return None
    order = np.argsort(-scores, kind="mergesort")
    return y_true[order], scores[order]


def _compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    prepared = _safe_auc_inputs(y_true, scores)
    if prepared is None:
        return np.nan
    y_sorted, _ = prepared
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
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
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
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def compute_pval(tf_a, tf_b, grn):
    trg_a = set(grn[grn['source'] == tf_a]['target'])
    trg_b = set(grn[grn['source'] == tf_b]['target'])
    total = set(grn['target'])
    a = len(trg_a & trg_b)
    b = len(trg_a - trg_b)
    c = len(trg_b - trg_a)
    d = len(total - (trg_a | trg_b))
    # Always run Fisher (even if a==0) to produce a valid p-value
    s, p = ss.fisher_exact([[a, b], [c, d]], alternative='greater')
    return s, p


def find_pairs_with_stats(grn):
    rows = []
    for tf_a, tf_b in combinations(grn['source'].unique(), r=2):
        s, p = compute_pval(tf_a, tf_b, grn)
        rows.append([tf_a, tf_b, s, p])
    df = pd.DataFrame(rows, columns=['tf_a', 'tf_b', 'stat', 'pval'])
    if df.shape[0] == 0:
        return df
    df['padj'] = ss.false_discovery_control(df['pval'], method='bh')
    return df


# Init args
parser = argparse.ArgumentParser()
parser.add_argument('-a','--grn_path', required=True)
parser.add_argument('-b','--resource_path', required=True)
parser.add_argument('-p','--thr_pval', required=True, type=float)
parser.add_argument('-f','--out_path', required=True)
parser.add_argument('-c','--confusion_path', required=False, default=None)
args = vars(parser.parse_args())

# Read
grn = pd.read_csv(args['grn_path']).drop_duplicates(['source', 'target'])
tfp = pd.read_csv(args['resource_path'], sep='\t', header=None)

# Process
tfs = set(tfp[0]) | set(tfp[1])
grn = grn[grn['source'].isin(tfs)].drop_duplicates(['source', 'target'])
tfp_pairs = set(['|'.join(sorted([a, b])) for a, b in zip(tfp[0], tfp[1])])
grn_name = os.path.basename(args['grn_path']).replace('.grn.csv', '')

if grn.shape[0] > 1:  # Need at least 2 TFs in grn
    pair_df = find_pairs_with_stats(grn)
    if pair_df.shape[0] > 0:
        # Continuous score for AUC: transform adjusted p-values (higher score = more significant)
        # Avoid -log10(0) -> inf; cap at 50.
        with np.errstate(divide='ignore'):
            pair_df['pair_score'] = -np.log10(pair_df['padj'].clip(lower=1e-300))
        pair_df.loc[~np.isfinite(pair_df['pair_score']), 'pair_score'] = 50.0
        # Classification threshold for binary metrics (retain original behavior)
        sig_pairs = set(['|'.join(sorted([a, b])) for a, b, padj in zip(pair_df['tf_a'], pair_df['tf_b'], pair_df['padj']) if padj < args['thr_pval']])
        tp_pairs = sig_pairs & tfp_pairs
        fp_pairs = sig_pairs - tfp_pairs
        fn_pairs = tfp_pairs - sig_pairs
        tp = len(tp_pairs)
        fp = len(fp_pairs)
        fn = len(fn_pairs)
        if tp > 0:
            rcl = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prc = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f01 = f_beta_score(prc, rcl)
        else:
            prc, rcl, f01 = 0., 0., 0.
        # AUC metrics over all evaluated pairs
        all_pairs = ['|'.join(sorted([a, b])) for a, b in zip(pair_df['tf_a'], pair_df['tf_b'])]
        labels = np.array([1 if p in tfp_pairs else 0 for p in all_pairs], dtype=int)
        scores = pair_df['pair_score'].to_numpy(dtype=float)
        # Normalize scores to [0,1] for stability
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores_norm = (scores - s_min) / (s_max - s_min)
        else:
            scores_norm = scores
        auroc = _compute_auroc(labels, scores_norm)
        auprc = _compute_auprc(labels, scores_norm)
    else:
        tp = fp = fn = 0
        prc = rcl = f01 = np.nan
        auroc = auprc = np.nan
        tp_pairs = fp_pairs = fn_pairs = set()

    if args['confusion_path'] is not None:
        confusion_df = pd.DataFrame({
            'name': [grn_name],
            'tp': [tp],
            'fp': [fp],
            'fn': [fn],
            'tp_pairs': ['|'.join(sorted(tp_pairs)) if len(tp_pairs) > 0 else ''],
            'fp_pairs': ['|'.join(sorted(fp_pairs)) if len(fp_pairs) > 0 else ''],
            'fn_pairs': ['|'.join(sorted(fn_pairs)) if len(fn_pairs) > 0 else '']
        })
        os.makedirs(os.path.dirname(args['confusion_path']), exist_ok=True)
        confusion_df.to_csv(args['confusion_path'], index=False)

    df = pd.DataFrame([[grn_name, prc, rcl, f01, auprc, auroc, tp, fp, fn]],
                      columns=['name', 'prc', 'rcl', 'f01', 'auprc', 'auroc', 'tp', 'fp', 'fn'])
else:
    df = pd.DataFrame([[grn_name, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, 0]],
                      columns=['name', 'prc', 'rcl', 'f01', 'auprc', 'auroc', 'tp', 'fp', 'fn'])

    if args['confusion_path'] is not None:
        confusion_df = pd.DataFrame({
            'name': [grn_name],
            'tp': [0],
            'fp': [0],
            'fn': [0],
            'tp_pairs': [''],
            'fp_pairs': [''],
            'fn_pairs': ['']
        })
        os.makedirs(os.path.dirname(args['confusion_path']), exist_ok=True)
        confusion_df.to_csv(args['confusion_path'], index=False)

# Write
df.to_csv(args['out_path'], index=False)
