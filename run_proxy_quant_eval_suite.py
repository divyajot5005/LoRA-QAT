import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


TARGETS_BY_PHASE = {
    "no_quant_ft": [
        {"name": "int4", "quantizer_type": "uniform_groupwise", "bit_width": 4, "group_size": 128},
        {"name": "fp8", "quantizer_type": "fp8_e4m3fn", "bit_width": 8, "group_size": 128},
    ],
    "quant_int4_ft": [
        {"name": "int4", "quantizer_type": "uniform_groupwise", "bit_width": 4, "group_size": 128},
    ],
    "quant_fp8_ft": [
        {"name": "fp8", "quantizer_type": "fp8_e4m3fn", "bit_width": 8, "group_size": 128},
    ],
    "quant_int4_act_ft": [
        {"name": "int4", "quantizer_type": "uniform_groupwise", "bit_width": 4, "group_size": 128},
    ],
    "quant_fp8_act_ft": [
        {"name": "fp8", "quantizer_type": "fp8_e4m3fn", "bit_width": 8, "group_size": 128},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run proxy quantized evaluation for completed H100-suite adapters.")
    parser.add_argument("--output-root", default="outputs/cloud_instruction_suite", help="Suite output root.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use.")
    parser.add_argument("--models", nargs="*", help="Optional subset of model families.")
    parser.add_argument("--tasks", nargs="*", help="Optional subset of task names.")
    parser.add_argument("--phases", nargs="*", help="Optional subset of finetune phases.")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run(command: list[str], workdir: Path) -> None:
    subprocess.run(command, cwd=workdir, check=True)


def _selected(value: str, allowed: list[str] | None) -> bool:
    return not allowed or value in set(allowed)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    repo_root = Path(__file__).resolve().parent
    rows: list[dict[str, object]] = []

    for config_path in sorted((output_root / "configs").glob("*.json")):
        run_name = config_path.stem
        parts = run_name.split("__")
        if len(parts) < 3:
            continue
        model_family = parts[0]
        task_name = parts[1]
        phase = "__".join(parts[2:])
        if phase not in TARGETS_BY_PHASE:
            continue
        if not _selected(model_family, args.models):
            continue
        if not _selected(task_name, args.tasks):
            continue
        if not _selected(phase, args.phases):
            continue

        config = _load_json(config_path)
        run_dir = Path(config["training"]["output_dir"])
        adapter_dir = run_dir / "adapter"
        if not adapter_dir.exists():
            continue

        for target in TARGETS_BY_PHASE[phase]:
            metrics_path = run_dir / f"quantized_proxy_metrics__{target['name']}.json"
            _run(
                [
                    args.python,
                    "evaluate_quantized_proxy.py",
                    "--config",
                    str(config_path),
                    "--adapter-dir",
                    str(adapter_dir),
                    "--output",
                    str(metrics_path),
                    "--quantizer-type",
                    target["quantizer_type"],
                    "--bit-width",
                    str(target["bit_width"]),
                    "--group-size",
                    str(target["group_size"]),
                ],
                repo_root,
            )

            payload = _load_json(metrics_path)
            rows.append(
                {
                    "run_name": run_name,
                    "phase": phase,
                    "target": target["name"],
                    "active_primary_metric": payload["active_primary_metric"],
                    "proxy_primary_metric": payload["proxy_primary_metric"],
                    "primary_metric_delta": payload["primary_metric_delta"],
                    "raw_quant_error": payload["raw_quant_error"],
                    "avg_distance_to_grid": payload["avg_distance_to_grid"],
                }
            )

    if not rows:
        return

    summary_csv = output_root / "proxy_summary.csv"
    summary_md = output_root / "proxy_summary.md"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Proxy Quantization Summary",
        "",
        "| Run | Target | Active | Proxy | Delta | Raw Quant Error | Avg Distance |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['target']} | {row['active_primary_metric']:.6f} | "
            f"{row['proxy_primary_metric']:.6f} | {row['primary_metric_delta']:.6f} | "
            f"{row['raw_quant_error']:.6e} | {row['avg_distance_to_grid']:.6e} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"proxy_summary_csv={summary_csv}")
    print(f"proxy_summary_md={summary_md}")


if __name__ == "__main__":
    main()
