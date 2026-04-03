import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "evaluate_quantized_proxy.py"


TARGETS = {
    "fp8": {"quantizer_type": "fp8_e4m3fn", "bit_width": 8, "group_size": 128},
    "int8": {"quantizer_type": "uniform_groupwise", "bit_width": 8, "group_size": 128},
    "int4": {"quantizer_type": "uniform_groupwise", "bit_width": 4, "group_size": 128},
    "int2": {"quantizer_type": "uniform_int2_groupwise", "bit_width": 2, "group_size": 128},
    "ternary": {"quantizer_type": "ternary_groupwise", "bit_width": 2, "group_size": 128},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill missing proxy-quant evaluation artifacts.")
    parser.add_argument("--mode", choices=["lora", "fullft"], required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-quant-stats", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-name", default="proxy_backfill_summary.csv")
    return parser.parse_args()


def _candidate_targets(mode: str, run_name: str) -> list[str]:
    if mode == "lora":
        if run_name.endswith("__no_quant_ft"):
            return ["fp8", "int4", "int2", "ternary"]
        if "__quant_int4_" in run_name:
            return ["int4", "int2", "ternary"]
        if "__quant_fp8_" in run_name:
            return ["fp8", "int2", "ternary"]
        if "__quant_int2_" in run_name:
            return ["int2"]
        if "__quant_ternary_" in run_name:
            return ["ternary"]
        return []

    if run_name.endswith("__no_quant"):
        return ["int8", "int4", "int2", "ternary"]
    for name in ("int8", "int4", "int2", "ternary"):
        token = f"__{name}__"
        if token in run_name:
            return [name]
    return []


def _artifact_dir(mode: str, run_dir: Path) -> Path:
    return run_dir / ("adapter" if mode == "lora" else "model")


def _run_eval(config_path: Path, artifact_dir: Path, output_path: Path, target_name: str, device: str, skip_quant_stats: bool) -> None:
    target = TARGETS[target_name]
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--config",
        str(config_path),
        "--artifact-dir",
        str(artifact_dir),
        "--output",
        str(output_path),
        "--device",
        device,
        "--quantizer-type",
        target["quantizer_type"],
        "--bit-width",
        str(target["bit_width"]),
        "--group-size",
        str(target["group_size"]),
    ]
    if skip_quant_stats:
        command.append("--skip-quant-stats")
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def _collect_rows(output_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(output_root.glob("runs/*/quantized_proxy_metrics*.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_name": metrics_path.parent.name,
                "target": metrics_path.stem.replace("quantized_proxy_metrics__", ""),
                "float_loss": payload["active_metrics"]["eval_loss"],
                "quant_loss": payload["proxy_quantized_metrics"]["eval_loss"],
                "eval_loss_delta": payload["eval_loss_delta"],
                "primary_metric_name": payload["primary_metric_name"],
                "active_primary_metric": payload["active_primary_metric"],
                "proxy_primary_metric": payload["proxy_primary_metric"],
                "primary_metric_delta": payload["primary_metric_delta"],
                "num_proxy_quantized_layers": payload["num_proxy_quantized_layers"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    runs_root = output_root / "runs"
    configs_root = output_root / "configs"

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "final_metrics.json").exists():
            continue
        artifact_dir = _artifact_dir(args.mode, run_dir)
        if not artifact_dir.exists():
            continue
        config_path = configs_root / f"{run_dir.name}.json"
        if not config_path.exists():
            continue
        for target_name in _candidate_targets(args.mode, run_dir.name):
            output_path = run_dir / f"quantized_proxy_metrics__{target_name}.json"
            if output_path.exists() and not args.force:
                continue
            _run_eval(
                config_path=config_path,
                artifact_dir=artifact_dir,
                output_path=output_path,
                target_name=target_name,
                device=args.device,
                skip_quant_stats=args.skip_quant_stats,
            )

    summary_rows = _collect_rows(output_root)
    summary_path = output_root / args.summary_name
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "target",
                "float_loss",
                "quant_loss",
                "eval_loss_delta",
                "primary_metric_name",
                "active_primary_metric",
                "proxy_primary_metric",
                "primary_metric_delta",
                "num_proxy_quantized_layers",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
