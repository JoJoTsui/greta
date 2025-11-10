from itertools import combinations
import scipy.stats as ss
import numpy as np
import pandas as pd
import argparse
import sys
import os
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


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--grn', required=True, help='GRN file path')
parser.add_argument('-b', '--db', required=True, help='TFP database file path')
parser.add_argument('-p', '--pval', type=float, required=True, help='P-value threshold')
parser.add_argument('-f', '--output', required=True, help='Output file path')
args = parser.parse_args()

# Read
grn = pd.read_csv(args.grn).drop_duplicates(['source', 'target'])
tfp = pd.read_csv(args.db, sep='\t', header=None)

# Process
tfs = set(tfp[0]) | set(tfp[1])
grn = grn[grn['source'].isin(tfs)]
tfp = set(['|'.join(sorted([a, b])) for a, b in zip(tfp[0], tfp[1])])
grn_name = os.path.basename(args.grn).replace('.grn.csv', '')

if grn.shape[0] > 1:  # Need at least 2 TFs in grn
    # Find pairs
    p_grn = find_pairs(grn, thr_pval=args.pval)
    
    # Compute F score
    tp = len(p_grn & tfp)
    if tp > 0:
        fp = len(p_grn - tfp)
        fn = len(tfp - p_grn)
        rcl = tp / (tp + fn)
        prc = tp / (tp + fp)
        f01 = f_beta_score(prc, rcl)
    else:
        prc, rcl, f01 = 0., 0., 0.
    df = pd.DataFrame([[grn_name, prc, rcl, f01]], columns=['name', 'prc', 'rcl', 'f01'])
else:
    df = pd.DataFrame([[grn_name, np.nan, np.nan, np.nan]], columns=['name', 'prc', 'rcl', 'f01'])

# Write
df.to_csv(args.output, index=False)
