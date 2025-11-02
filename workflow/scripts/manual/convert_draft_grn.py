#!/usr/bin/env python3
"""Convert draft GRN edges into a GRETA-compatible GRN table."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set, Tuple

try:
    import pandas as pd  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - import guard for runtime feedback
    raise SystemExit("pandas is required to run convert_draft_grn.py") from exc

try:
    import pyranges as pr  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency for CRE annotation
    pr = None


Pair = Tuple[str, str]


def _load_draft(
    path: Path,
    tf_col: str,
    target_col: str,
    score_col: Optional[str],
    score_threshold: float,
    top_k: Optional[int],
) -> pd.DataFrame:
    """Read the draft GRN and keep the relevant columns."""
    df = pd.read_csv(path)
    missing = {tf_col, target_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in draft file: {', '.join(sorted(missing))}")

    if score_col and score_col not in df.columns:
        raise ValueError(f"Score column '{score_col}' not found in draft file")

    df = df[[tf_col, target_col] + ([score_col] if score_col else [])].copy()
    df.rename(columns={tf_col: "source", target_col: "target"}, inplace=True)

    if score_col:
        df.rename(columns={score_col: "score"}, inplace=True)
        df = df[pd.to_numeric(df["score"], errors="coerce").notna()]
        df["score"] = df["score"].astype(float)
        if score_threshold is not None:
            df = df[df["score"] >= score_threshold]
    else:
        df["score"] = 1.0

    if top_k is not None:
        df = (
            df.sort_values(["source", "score"], ascending=[True, False])
            .groupby("source", as_index=False)
            .head(top_k)
            .reset_index(drop=True)
        )

    return df.drop_duplicates(["source", "target"])  # keep highest score per pair


def _parse_target_list(raw: str) -> Iterable[str]:
    """Extract gene names from a string representation of target tuples."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, (list, tuple, set, frozenset)):
        genes = []
        for item in parsed:
            if isinstance(item, (list, tuple)) and item:
                genes.append(str(item[0]))
            else:
                genes.append(str(item))
        return genes
    return []


def _load_filter_pairs(path: Path) -> Set[Pair]:
    """Load TF-target pairs from a pruned SCENIC table."""
    df = pd.read_csv(path, header=1)
    df.columns = [str(col).strip() for col in df.columns]
    if "Unnamed: 0" in df.columns and "TF" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "TF"})
    if "Unnamed: 1" in df.columns and "MotifID" not in df.columns:
        df = df.rename(columns={"Unnamed: 1": "MotifID"})

    tf_candidates = [c for c in df.columns if c.lower() == "tf"]
    tf_col = tf_candidates[0] if tf_candidates else df.columns[0]
    tgt_candidate = [c for c in df.columns if "target" in c.lower()]
    if not tgt_candidate:
        raise ValueError("Could not find a TargetGenes column in the filter file")
    tgt_col = tgt_candidate[0]

    working = df[[tf_col, tgt_col]].dropna()
    working = working[working[tf_col].astype(str).str.strip().str.lower() != "tf"]

    pairs: Set[Pair] = set()
    for _, row in working.iterrows():
        tf = str(row[tf_col]).strip()
        for gene in _parse_target_list(str(row[tgt_col])):
            gene = gene.strip()
            if gene:
                pairs.add((tf, gene))
    return pairs


def _apply_filter(grn: pd.DataFrame, pairs: Set[Pair]) -> pd.DataFrame:
    if not pairs:
        return grn
    filter_df = pd.DataFrame(list(pairs), columns=["source", "target"])
    return grn.merge(filter_df, on=["source", "target"], how="inner")


def _load_reg_pairs(path: Path) -> pd.DataFrame:
    """Load TF-target pairs from a SCENIC regulon table."""
    reg = pd.read_csv(path, header=None)
    if reg.shape[0] < 3 or reg.shape[1] <= 8:
        raise ValueError("Unexpected format for regulon file")
    reg = reg.iloc[2:, [0, 8]].copy()
    reg.columns = ["source", "target"]
    reg.dropna(inplace=True)
    reg["target"] = reg["target"].astype(str)
    reg = reg[reg["target"].str.strip() != ""]
    reg["target"] = reg["target"].str.split(",")
    reg = reg.explode("target")
    reg["target"] = reg["target"].str.replace(r"[\[\]\(\)' ]", "", regex=True)
    reg = reg[reg["target"] != ""]
    return reg.drop_duplicates()


def _load_promoters(path: Path) -> pd.DataFrame:
    if pr is None:
        raise RuntimeError("pyranges is required to annotate CRE coordinates")
    proms = pr.read_bed(str(path)).df
    if "Name" not in proms.columns:
        raise ValueError("Promoter BED file must contain a Name column with gene identifiers")
    proms["cre"] = (
        proms["Chromosome"].astype(str)
        + "-"
        + proms["Start"].astype(int).astype(str)
        + "-"
        + proms["End"].astype(int).astype(str)
    )
    return proms.rename(columns={"Name": "target"})[["target", "cre"]]


def _annotate_with_promoters(
    grn: pd.DataFrame,
    promoters_path: Optional[Path],
    reg_path: Optional[Path],
    require_cre: bool,
) -> pd.DataFrame:
    if promoters_path is None:
        return grn

    proms = _load_promoters(promoters_path)
    merged = grn

    if reg_path is not None:
        reg_pairs = _load_reg_pairs(reg_path)
        merged = merged.merge(reg_pairs, on=["source", "target"], how="inner")

    merged = merged.merge(proms, on="target", how="inner")
    merged = merged[["source", "cre", "target", "score"]]

    if require_cre and merged.empty:
        raise RuntimeError("No CRE overlaps found after promoter annotation")

    if merged.empty and not require_cre:
        return grn.assign(cre=pd.NA)[["source", "cre", "target", "score"]]

    return merged.drop_duplicates(["source", "cre", "target"])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True, type=Path, help="Path to draft_grn.csv")
    parser.add_argument("--output", required=True, type=Path, help="Destination GRETA GRN path")
    parser.add_argument("--tf-column", default="TF", help="Column name for TF identifiers")
    parser.add_argument("--target-column", default="target", help="Column name for target genes")
    parser.add_argument("--score-column", default="importance", help="Column name for edge scores")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Minimum score to keep")
    parser.add_argument("--top-k", type=int, default=None, help="Keep top-K targets per TF")
    parser.add_argument("--filter", type=Path, default=None, help="Optional pruned regulon file to filter edges")
    parser.add_argument("--promoters", type=Path, default=None, help="Optional promoter BED file for CRE annotation")
    parser.add_argument("--reg", type=Path, default=None, help="Optional SCENIC regulon CSV for TF-target filtering")
    parser.add_argument("--require-cre", action="store_true", help="Fail if no CRE can be assigned")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    grn = _load_draft(
        path=args.draft,
        tf_col=args.tf_column,
        target_col=args.target_column,
        score_col=args.score_column,
        score_threshold=args.score_threshold,
        top_k=args.top_k,
    )

    if args.filter:
        pairs = _load_filter_pairs(args.filter)
        grn = _apply_filter(grn, pairs)

    grn = _annotate_with_promoters(
        grn=grn,
        promoters_path=args.promoters,
        reg_path=args.reg,
        require_cre=args.require_cre,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grn.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
