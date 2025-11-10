#!/usr/bin/env python3
"""Compute GRETA prior metrics (DETAILED) for a GRN across selected databases.

This manual runner mirrors the Snakemake detailed targets, calling:
- tfm_detailed.py for TF-marker metrics (writes scores + confusion + optional subset)
- tfp_detailed.py for TF-protein interaction metrics (writes scores + confusion)

Behavioral parity with Snakefile (tcell case):
- case == 'tcell' => tfm on {hpa, tfmdb}; tfp on {intact}
- outputs under: anl/metrics/prior_detailed/<metric>/<db>/<dataset>.<case>/
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class MetricSpec:
    name: str
    script: Path
    db_root: Path
    resource_suffix: str
    command_type: str  # 'tfm' or 'tfp'


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grn", required=True, type=Path, help="Path to the GRETA-formatted GRN CSV")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset identifier (e.g. pbmc10k)")
    parser.add_argument("--case", type=str, default=None, help="Case identifier (e.g. tcell)")
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=["tfm", "tfp"],
        default=["tfm", "tfp"],
        help="Subset of metrics to run",
    )
    parser.add_argument("--db-root", type=Path, default=None, help="Override path to the database root directory")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output directory for metric results (defaults to anl/metrics/prior_detailed)",
    )
    parser.add_argument("--tfp-threshold", type=float, default=0.01, help="FDR threshold for TFP metric")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python executable to use")
    parser.add_argument("--skip-existing", action="store_true", help="Skip metrics with existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them")
    return parser.parse_args(argv)


def _infer_dataset_case(grn_path: Path) -> tuple[str, str]:
    parts = grn_path.resolve().parts
    if "dts" in parts:
        idx = parts.index("dts")
        try:
            dataset = parts[idx + 1]
            if parts[idx + 2] != "cases":
                raise ValueError
            case = parts[idx + 3]
            return dataset, case
        except (IndexError, ValueError) as exc:
            raise ValueError("Could not infer dataset/case from GRN path") from exc
    raise ValueError("GRN path does not contain a dts/<dataset>/cases/<case> component")


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _build_specs(repo_root: Path, db_root_override: Optional[Path]) -> Dict[str, MetricSpec]:
    db_root = (db_root_override or (repo_root / "dbs" / "hg38")).resolve()
    return {
        "tfm": MetricSpec(
            name="tfm",
            script=repo_root / "workflow" / "scripts" / "anl" / "metrics" / "prior" / "tfm_detailed.py",
            db_root=db_root / "tfm",
            resource_suffix=".tsv",
            command_type="tfm",
        ),
        "tfp": MetricSpec(
            name="tfp",
            script=repo_root / "workflow" / "scripts" / "anl" / "metrics" / "prior" / "tfp_detailed.py",
            db_root=db_root / "tfp",
            resource_suffix=".tsv",
            command_type="tfp",
        ),
    }


def _allowed_dbs_for(dataset: str, case: str, metric_name: str) -> Optional[set[str]]:
    """Return a set of allowed database names for a given dataset/case/metric.

    Mirrors the Snakefile prior_metric_targets_detailed behavior for tcell:
    - tfm: {hpa, tfmdb}
    - tfp: {intact}
    None means no restriction.
    """
    if case == "tcell":
        if metric_name == "tfm":
            return {"hpa", "tfmdb"}
        if metric_name == "tfp":
            return {"intact"}
        return set()
    return None


def _iter_resources(spec: MetricSpec, allowed_dbs: Optional[set[str]] = None) -> Iterable[Path]:
    if not spec.db_root.exists():
        return []
    for child in sorted(p for p in spec.db_root.iterdir() if p.is_dir()):
        if allowed_dbs is not None:
            if len(allowed_dbs) == 0 or child.name not in allowed_dbs:
                continue
        candidate = (child / f"{child.name}{spec.resource_suffix}").resolve()
        if candidate.exists():
            yield candidate


def _command_for_spec(
    spec: MetricSpec,
    python_exe: Path,
    grn_path: Path,
    resource_path: Path,
    out_path: Path,
    subset_path: Optional[Path],
    confusion_path: Path,
    tfp_threshold: float,
) -> List[str]:
    if spec.command_type == "tfm":
        cmd = [
            str(python_exe),
            str(spec.script),
            "-a",
            str(grn_path),
            "-b",
            str(resource_path),
            "-f",
            str(out_path),
            "-c",
            str(confusion_path),
        ]
        if subset_path is not None:
            cmd.extend(["-s", str(subset_path)])
        return cmd
    # tfp detailed
    return [
        str(python_exe),
        str(spec.script),
        "-a",
        str(grn_path),
        "-b",
        str(resource_path),
        "-p",
        str(tfp_threshold),
        "-f",
        str(out_path),
        "-c",
        str(confusion_path),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    grn_path = args.grn.resolve()
    if not grn_path.exists():
        raise FileNotFoundError(f"GRN not found: {grn_path}")

    dataset = args.dataset
    case = args.case
    if dataset is None or case is None:
        dataset, case = _infer_dataset_case(grn_path)

    repo_root = Path(__file__).resolve().parents[3]
    output_root = (args.output_root or (repo_root / "anl" / "metrics" / "prior_detailed")).resolve()

    expected_runs_dir = repo_root / "dts" / dataset / "cases" / case / "runs"
    cleanup_link: Optional[Path] = None
    metrics_grn_path = grn_path
    if not _is_relative_to(grn_path, expected_runs_dir):
        expected_runs_dir.mkdir(parents=True, exist_ok=True)
        link_path = expected_runs_dir / grn_path.name
        if link_path.exists():
            if link_path.is_symlink() and link_path.resolve() == grn_path:
                metrics_grn_path = link_path
            else:
                idx = 1
                while True:
                    alternative = expected_runs_dir / f"{grn_path.stem}.{idx}{grn_path.suffix}"
                    if not alternative.exists():
                        link_path = alternative
                        break
                    idx += 1
                link_path.symlink_to(grn_path)
                metrics_grn_path = link_path
                cleanup_link = link_path
        else:
            link_path.symlink_to(grn_path)
            metrics_grn_path = link_path
            cleanup_link = link_path

    mdata_path = expected_runs_dir.parent / "mdata.h5mu"
    if not mdata_path.exists():
        print(f"Warning: expected metadata file {mdata_path} not found; metric scripts may fail", file=sys.stderr)

    specs = _build_specs(repo_root, args.db_root)
    metrics = args.metrics or list(specs.keys())

    grn_name = grn_path.name.replace(".grn.csv", "")
    python_exe = args.python.resolve()

    planned: List[tuple[List[str], Path, Path]] = []  # (cmd, out_dir, confusion_csv)
    try:
        for metric_name in metrics:
            spec = specs[metric_name]
            if not spec.script.exists():
                raise FileNotFoundError(f"Metric script missing: {spec.script}")
            allowed = _allowed_dbs_for(dataset, case, spec.name)
            for resource_path in _iter_resources(spec, allowed_dbs=allowed):
                db_name = resource_path.parent.name
                out_dir = output_root / spec.name / db_name / f"{dataset}.{case}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{grn_name}.scores.csv"
                confusion_row = out_dir / f"{grn_name}.confusion.csv"
                subset_path: Optional[Path] = None
                if spec.name == "tfm":
                    subset_path = out_dir / f"{dataset}.{case}.subset.csv"
                if args.skip_existing and out_path.exists():
                    continue
                cmd = _command_for_spec(
                    spec,
                    python_exe,
                    metrics_grn_path,
                    resource_path,
                    out_path,
                    subset_path,
                    confusion_row,
                    args.tfp_threshold,
                )
                planned.append((cmd, out_dir, confusion_row))

        if not planned:
            print("No metric jobs to run", file=sys.stderr)
            return 0

        if args.dry_run:
            for cmd, out_dir, conf_csv in planned:
                print(f"DRY-RUN: {' '.join(cmd)} -> {out_dir}")
            return 0

        # Run jobs
        for cmd, out_dir, conf_csv in planned:
            print(f"Running {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=repo_root)
            print(f"Wrote outputs under {out_dir}")

        # Lightweight aggregation: write/refresh confusion_agg.csv per out_dir
        for _, out_dir, _ in planned:
            confs = sorted(out_dir.glob("*.confusion.csv"))
            if not confs:
                continue
            # Concatenate rows (no heavy deps): simply copy header from first and append others' rows
            agg_path = out_dir / f"{dataset}.{case}.confusion_agg.csv"
            with agg_path.open("w", encoding="utf-8") as fout:
                header_written = False
                for i, f in enumerate(confs):
                    with f.open("r", encoding="utf-8") as fin:
                        for j, line in enumerate(fin):
                            if j == 0 and header_written:
                                continue
                            fout.write(line)
                    header_written = True
            print(f"Aggregated confusion -> {agg_path}")

        return 0
    finally:
        if cleanup_link is not None and cleanup_link.exists():
            cleanup_link.unlink()


if __name__ == "__main__":
    sys.exit(main())
