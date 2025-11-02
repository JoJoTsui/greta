#!/usr/bin/env python3
"""Compute GRETA prior metrics for a GRN across all available databases."""

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
    command_type: str  # 'gnm', 'tfm', or 'tfp'
    group: Optional[str] = None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grn", required=True, type=Path, help="Path to the GRETA-formatted GRN CSV")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset identifier (e.g. pbmc10k)")
    parser.add_argument("--case", type=str, default=None, help="Case identifier (e.g. tcell)")
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=["tfm", "tfp", "tfb", "cre", "c2g"],
        default=None,
        help="Subset of metrics to run",
    )
    parser.add_argument("--db-root", type=Path, default=None, help="Override path to the database root directory")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output directory for metric results",
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
            script=repo_root / "workflow" / "scripts" / "anl" / "metrics" / "prior" / "tfm.py",
            db_root=db_root / "tfm",
            resource_suffix=".tsv",
            command_type="tfm",
        ),
        "tfp": MetricSpec(
            name="tfp",
            script=repo_root / "workflow" / "scripts" / "anl" / "metrics" / "prior" / "tfp.py",
            db_root=db_root / "tfp",
            resource_suffix=".tsv",
            command_type="tfp",
        ),
        "tfb": MetricSpec(
            name="tfb",
            script=repo_root / "workflow" / "scripts" / "anl" / "metrics" / "prior" / "gnm.py",
            db_root=db_root / "tfb",
            resource_suffix=".bed",
            command_type="gnm",
            group="source",
        ),
        "cre": MetricSpec(
            name="cre",
            script=repo_root / "workflow" / "scripts" / "anl" / "metrics" / "prior" / "gnm.py",
            db_root=db_root / "cre",
            resource_suffix=".bed",
            command_type="gnm",
            group=None,
        ),
        "c2g": MetricSpec(
            name="c2g",
            script=repo_root / "workflow" / "scripts" / "anl" / "metrics" / "prior" / "gnm.py",
            db_root=db_root / "c2g",
            resource_suffix=".bed",
            command_type="gnm",
            group="target",
        ),
    }


def _iter_resources(spec: MetricSpec) -> Iterable[Path]:
    if not spec.db_root.exists():
        return []
    resources: List[Path] = []
    for child in sorted(spec.db_root.iterdir()):
        if not child.is_dir():
            continue
        candidate = (child / f"{child.name}{spec.resource_suffix}").resolve()
        if candidate.exists():
            resources.append(candidate)
    return resources


def _command_for_spec(
    spec: MetricSpec,
    python_exe: Path,
    grn_path: Path,
    resource_path: Path,
    output_path: Path,
    tfp_threshold: float,
) -> List[str]:
    if spec.command_type == "tfp":
        return [
            str(python_exe),
            str(spec.script),
            str(grn_path),
            str(resource_path),
            str(tfp_threshold),
            str(output_path),
        ]
    cmd = [
        str(python_exe),
        str(spec.script),
        "-a",
        str(grn_path),
        "-b",
        str(resource_path),
    ]
    if spec.group:
        cmd.extend(["-d", spec.group])
    cmd.extend(["-f", str(output_path)])
    return cmd


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
    output_root = (args.output_root or (repo_root / "anl" / "metrics" / "prior")).resolve()

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

    planned = []
    try:
        for metric_name in metrics:
            spec = specs[metric_name]
            if not spec.script.exists():
                raise FileNotFoundError(f"Metric script missing: {spec.script}")
            for resource_path in _iter_resources(spec):
                db_name = resource_path.parent.name
                out_dir = output_root / spec.name / db_name / f"{dataset}.{case}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{grn_name}.scores.csv"
                if args.skip_existing and out_path.exists():
                    continue
                cmd = _command_for_spec(
                    spec,
                    python_exe,
                    metrics_grn_path,
                    resource_path,
                    out_path,
                    args.tfp_threshold,
                )
                planned.append((cmd, out_path))

        if not planned:
            print("No metric jobs to run", file=sys.stderr)
            return 0

        if args.dry_run:
            for cmd, out_path in planned:
                print(f"DRY-RUN: {' '.join(cmd)} -> {out_path}")
            return 0

        for cmd, out_path in planned:
            print(f"Running {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=repo_root)
            print(f"Wrote {out_path}")

        return 0
    finally:
        if cleanup_link is not None and cleanup_link.exists():
            cleanup_link.unlink()


if __name__ == "__main__":
    sys.exit(main())
