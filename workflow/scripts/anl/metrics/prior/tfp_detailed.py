from itertools import combinations
import scipy.stats as ss
import numpy as np
import pandas as pd
import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import f_beta_score


def compute_pval(tf_a, tf_b, grn):
    trg_a = set(grn[grn['source'] == tf_a]['target'])
    trg_b = set(grn[grn['source'] == tf_b]['target'])
    total = set(grn['target'])
    a = len(trg_a & trg_b)
    if a > 0:
        b = len(trg_a - trg_b)
        c = len(trg_b - trg_a)
        d = len(total - (trg_a | trg_b))
        s, p = ss.fisher_exact([[a, b], [c, d]], alternative='greater')
    else:
        s, p = 0, np.nan
    return s, p


def find_pairs(grn, thr_pval):
    df = []
    for tf_a, tf_b in combinations(grn['source'].unique(), r=2):
        s, p = compute_pval(tf_a, tf_b, grn)
        df.append([tf_a, tf_b, s, p])
    df = pd.DataFrame(df, columns=['tf_a', 'tf_b', 'stat', 'pval']).dropna()
    if df.shape[0] > 0:
        df['padj'] = ss.false_discovery_control(df['pval'], method='bh')
        df = df[df['padj'] < thr_pval]
        pairs = set(['|'.join(sorted([a, b])) for a, b in zip(df['tf_a'], df['tf_b'])])
    else:
        pairs = set()
    return pairs


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
grn = grn[grn['source'].isin(tfs)]
tfp = set(['|'.join(sorted([a, b])) for a, b in zip(tfp[0], tfp[1])])
grn_name = os.path.basename(args['grn_path']).replace('.grn.csv', '')

if grn.shape[0] > 1:  # Need at least 2 TFs in grn
    # Find pairs
    p_grn = find_pairs(grn, thr_pval=args['thr_pval'])
    
    # Compute metrics
    tp_pairs = p_grn & tfp
    fp_pairs = p_grn - tfp
    fn_pairs = tfp - p_grn
    
    tp = len(tp_pairs)
    fp = len(fp_pairs)
    fn = len(fn_pairs)
    
    if tp > 0:
        rcl = tp / (tp + fn)
        prc = tp / (tp + fp)
        f01 = f_beta_score(prc, rcl)
    else:
        prc, rcl, f01 = 0., 0., 0.
    
    # Save confusion matrix if path provided
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
    
    df = pd.DataFrame([[grn_name, prc, rcl, f01, tp, fp, fn]], 
                      columns=['name', 'prc', 'rcl', 'f01', 'tp', 'fp', 'fn'])
else:
    df = pd.DataFrame([[grn_name, np.nan, np.nan, np.nan, 0, 0, 0]], 
                      columns=['name', 'prc', 'rcl', 'f01', 'tp', 'fp', 'fn'])
    
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
