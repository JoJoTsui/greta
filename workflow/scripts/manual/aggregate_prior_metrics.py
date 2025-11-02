#!/usr/bin/env python3
"""Aggregate prior metric outputs and produce GRETA-style summaries."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset identifier (e.g. pbmc10k)")
    parser.add_argument("--case", required=True, help="Case identifier (e.g. tcell)")
    parser.add_argument("--input-root", type=Path, default=None, help="Root directory with metric outputs")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional override for summary CSV path")
    parser.add_argument("--metrics", nargs="*", default=None, help="Subset of metric families to aggregate")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python executable to use")
    parser.add_argument("--add-info", action="store_true", help="Add metadata columns to aggregated tables")
    parser.add_argument("--skip-existing", action="store_true", help="Skip aggregation when outputs already exist")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them")
    return parser.parse_args(argv)


def _collect_inputs(dataset_case_dir: Path) -> List[Path]:
    if not dataset_case_dir.exists():
        return []
    return sorted(p for p in dataset_case_dir.glob("*.scores.csv") if p.is_file())


def _aggregate_metric(
    python_exe: Path,
    aggregator: Path,
    input_files: Iterable[Path],
    output_path: Path,
    add_info: bool,
    dry_run: bool,
    repo_root: Path,
) -> None:
    if not input_files:
        return
    cmd: List[str] = [str(python_exe), str(aggregator), "-i", *(str(p) for p in input_files)]
    if add_info:
        cmd.append("-a")
    cmd.extend(["-o", str(output_path)])
    if dry_run:
        print(f"DRY-RUN: {' '.join(cmd)}")
        return
    print(f"Aggregating {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=repo_root)
    print(f"Wrote {output_path}")


def _ensure_repo_view(
    src: Path,
    repo_root: Path,
    metric_type: str,
    task: str,
    db: str,
    dataset: str,
    case: str,
) -> Tuple[Path, bool]:
    src = src.resolve()
    repo_base = repo_root / "anl" / "metrics" / metric_type / task / db
    repo_base.mkdir(parents=True, exist_ok=True)
    dest = repo_base / f"{dataset}.{case}.scores.csv"
    try:
        if dest.exists() or dest.is_symlink():
            if dest.resolve() == src:
                return dest, False
            dest.unlink()
    except FileNotFoundError:
        pass
    try:
        dest.symlink_to(src)
        return dest, True
    except OSError:
        dest.write_bytes(src.read_bytes())
        return dest, True


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    input_root = (args.input_root or (repo_root / "anl" / "metrics" / "prior")).resolve()

    aggregator_script = repo_root / "workflow" / "scripts" / "anl" / "metrics" / "aggregate.py"
    summary_script = repo_root / "workflow" / "scripts" / "anl" / "metrics" / "test.py"
    if not aggregator_script.exists():
        raise FileNotFoundError(f"Missing aggregate.py script at {aggregator_script}")
    if not summary_script.exists():
        raise FileNotFoundError(f"Missing test.py script at {summary_script}")

    metrics = args.metrics or [d.name for d in input_root.iterdir() if d.is_dir()]
    metric_type = input_root.name

    aggregated_paths: List[Path] = []
    summary_inputs: List[Path] = []
    cleanup_targets: List[Path] = []
    for metric in sorted(metrics):
        metric_dir = input_root / metric
        if not metric_dir.exists():
            continue
        for db_dir in sorted(d for d in metric_dir.iterdir() if d.is_dir()):
            dataset_case_dir = db_dir / f"{args.dataset}.{args.case}"
            input_files = _collect_inputs(dataset_case_dir)
            if not input_files:
                continue
            aggregate_out = db_dir / f"{args.dataset}.{args.case}.scores.csv"
            if args.skip_existing and aggregate_out.exists():
                repo_view, cleanup = _ensure_repo_view(
                    aggregate_out,
                    repo_root,
                    metric_type,
                    metric,
                    db_dir.name,
                    args.dataset,
                    args.case,
                )
                summary_inputs.append(repo_view)
                aggregated_paths.append(aggregate_out)
                if cleanup:
                    cleanup_targets.append(repo_view)
                continue
            _aggregate_metric(
                python_exe=args.python,
                aggregator=aggregator_script,
                input_files=input_files,
                output_path=aggregate_out,
                add_info=args.add_info,
                dry_run=args.dry_run,
                repo_root=repo_root,
            )
            repo_view, cleanup = _ensure_repo_view(
                aggregate_out,
                repo_root,
                metric_type,
                metric,
                db_dir.name,
                args.dataset,
                args.case,
            )
            summary_inputs.append(repo_view)
            aggregated_paths.append(aggregate_out)
            if cleanup:
                cleanup_targets.append(repo_view)

    if not aggregated_paths:
        print("No aggregated outputs produced", file=sys.stderr)
        return 0

    summary_inputs = sorted(dict.fromkeys(summary_inputs))
    aggregated_paths = sorted(dict.fromkeys(aggregated_paths))

    summary_out = args.summary_out or (repo_root / "anl" / "metrics" / "summary" / f"{args.dataset}.{args.case}.csv")
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(args.python), str(summary_script), "-m", *(str(p) for p in summary_inputs), "-o", str(summary_out)]
    try:
        if args.dry_run:
            print(f"DRY-RUN: {' '.join(cmd)}")
            return 0

        print(f"Summarising {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=repo_root)
        print(f"Wrote {summary_out}")
        return 0
    finally:
        for target in cleanup_targets:
            try:
                if target.is_symlink() or target.exists():
                    target.unlink()
            except FileNotFoundError:
                continue


if __name__ == "__main__":
    sys.exit(main())
