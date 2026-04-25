import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


EXPERIMENTS = [
    {
        "family": "qwen",
        "label": "qwen3-8b",
        "base_model": "Qwen/Qwen3-8B",
        "adapters": {
            "sft": {
                42: "yrhong/qwen3-8b-sft-seed42",
                43: "yrhong/qwen3-8b-sft-seed43",
                44: "yrhong/qwen3-8b-sft-seed44",
            },
            "diat": {
                42: "yrhong/qwen3-8b-diat-seed42",
                43: "yrhong/qwen3-8b-diat-seed43",
                44: "yrhong/qwen3-8b-diat-seed44",
            },
        },
    },
    {
        "family": "llama",
        "label": "llama-3.1-8b-instruct",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "adapters": {
            "sft": {
                42: "yrhong/llama-3.1-8b-instruct-sft-seed42",
                43: "yrhong/llama-3.1-8b-instruct-sft-seed43",
                44: "yrhong/llama-3.1-8b-instruct-sft-seed44",
            },
            "diat": {
                42: "yrhong/llama-3.1-8b-instruct-diat-seed42",
                43: "yrhong/llama-3.1-8b-instruct-diat-seed43",
                44: "yrhong/llama-3.1-8b-instruct-diat-seed44",
            },
        },
    },
]


CSV_COLUMNS = [
    "dataset",
    "dataset_path",
    "family",
    "variant",
    "prompt_style",
    "seed",
    "backend",
    "model",
    "adapter",
    "model_type",
    "status",
    "returncode",
    "num_samples",
    "accuracy",
    "correct",
    "total",
    "summary_file",
    "results_file",
    "command",
]


def get_project_root():
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def get_eval_script(project_root):
    candidates = [
        project_root / "eval_diat.py",
        project_root / "src" / "eval_diat.py",
        project_root / "scripts" / "eval_diat.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "eval_diat.py not found in expected locations: "
        + ", ".join(str(path) for path in candidates)
    )


def get_default_dataset_dir(project_root):
    return project_root / "dataset"


def parse_args():
    project_root = get_project_root()
    parser = argparse.ArgumentParser(
        description="Run all DIAT evaluation experiments for one dataset or all datasets in a folder."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        nargs="+",
        default=None,
        help="One or more dataset files to evaluate (.json or .jsonl).",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(get_default_dataset_dir(project_root)),
        help="Directory containing dataset files when --dataset_path is not set. Defaults to ./dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "results"),
        help="Root output directory passed to eval_diat.py.",
    )
    parser.add_argument("--backend", type=str, choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--debug_n", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["qwen", "llama"],
        default=["qwen", "llama"],
        help="Subset of model families to run.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["base", "sft", "diat"],
        default=["base", "sft", "diat"],
        help="Subset of experiment variants to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Seeds to run for adapter variants.",
    )
    parser.add_argument(
        "--base_prompt_styles",
        nargs="+",
        choices=["default", "instruction"],
        default=["default", "instruction"],
        help="Prompt styles to evaluate for base models.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Keep running remaining experiments after a failed command.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def collect_dataset_files(args):
    if args.dataset_path:
        dataset_files = []
        for raw_path in args.dataset_path:
            dataset_path = Path(raw_path).resolve()
            if not dataset_path.is_file():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            dataset_files.append(dataset_path)
        return dataset_files

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. Create ./dataset and add .json/.jsonl files, or pass --dataset_dir with a different location."
        )

    dataset_files = sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )
    if not dataset_files:
        raise FileNotFoundError(
            f"No .json or .jsonl datasets found in {dataset_dir}."
        )
    return dataset_files


def build_runs(args):
    selected_families = set(args.families)
    selected_variants = set(args.variants)
    selected_seeds = set(args.seeds)
    selected_base_prompt_styles = list(dict.fromkeys(args.base_prompt_styles))

    runs = []
    for experiment in EXPERIMENTS:
        if experiment["family"] not in selected_families:
            continue

        if "base" in selected_variants:
            for prompt_style in selected_base_prompt_styles:
                runs.append(
                    {
                        "family": experiment["family"],
                        "variant": "base",
                        "prompt_style": prompt_style,
                        "seed": None,
                        "model": experiment["base_model"],
                        "adapter": None,
                    }
                )

        for variant_name, variant_adapters in experiment["adapters"].items():
            if variant_name not in selected_variants:
                continue
            for seed, adapter in sorted(variant_adapters.items()):
                if seed not in selected_seeds:
                    continue
                runs.append(
                    {
                        "family": experiment["family"],
                        "variant": variant_name,
                        "prompt_style": "default",
                        "seed": seed,
                        "model": experiment["base_model"],
                        "adapter": adapter,
                    }
                )
    return runs


def sanitize_name(name: str):
    return name.replace("/", "_")


def build_model_name(model, adapter):
    base_name = sanitize_name(model)
    if adapter:
        adapter_name = sanitize_name(os.path.basename(os.path.normpath(adapter)))
        return f"{base_name}/tuned_{adapter_name}"
    return f"{base_name}/base"


def infer_eval_output_paths(output_root, dataset_path, run, backend):
    model_name = build_model_name(run["model"], run["adapter"])

    dataset_tag = str(dataset_path).replace(os.sep, "_")
    output_dir = output_root / model_name / backend / run["prompt_style"] / dataset_tag
    return output_dir / "test_results.jsonl", output_dir / "summary.json"


def build_command(eval_script, args, dataset_path, run):
    cmd = [
        sys.executable,
        str(eval_script),
        "--model",
        run["model"],
        "--backend",
        args.backend,
        "--dataset_path",
        str(dataset_path),
        "--output_dir",
        args.output_dir,
        "--prompt_style",
        run["prompt_style"],
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--debug_n",
        str(args.debug_n),
    ]

    if run["adapter"]:
        cmd.extend(["--adapter", run["adapter"]])

    if args.backend == "hf":
        if args.load_in_4bit:
            cmd.append("--load_in_4bit")
        if args.load_in_8bit:
            cmd.append("--load_in_8bit")
    else:
        cmd.extend(
            [
                "--batch_size",
                str(args.batch_size),
                "--tensor_parallel_size",
                str(args.tensor_parallel_size),
                "--gpu_memory_utilization",
                str(args.gpu_memory_utilization),
                "--max_model_len",
                str(args.max_model_len),
                "--max_num_seqs",
                str(args.max_num_seqs),
                "--dtype",
                args.dtype,
            ]
        )

    return cmd


def load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def build_csv_row(record):
    summary = load_json_file(record.get("summary_file", "")) if record.get("status") == "ok" else {}
    experiment_info = summary.get("experiment_info", {})
    results = summary.get("results", {})
    output_paths = summary.get("output_paths", {})
    dataset_path = record.get("dataset_path", "")

    return {
        "dataset": Path(dataset_path).name,
        "dataset_path": dataset_path,
        "family": record.get("family"),
        "variant": record.get("variant"),
        "prompt_style": experiment_info.get(
            "prompt_style", record.get("prompt_style", "")
        ),
        "seed": "" if record.get("seed") is None else record.get("seed"),
        "backend": record.get("backend"),
        "model": record.get("model"),
        "adapter": record.get("adapter") or "",
        "model_type": experiment_info.get("model_type", ""),
        "status": record.get("status", ""),
        "returncode": record.get("returncode", ""),
        "num_samples": experiment_info.get("num_samples", ""),
        "accuracy": results.get("accuracy", ""),
        "correct": results.get("correct", ""),
        "total": results.get("total", ""),
        "summary_file": record.get("summary_file", ""),
        "results_file": output_paths.get("results_file", record.get("results_file", "")),
        "command": " ".join(record.get("command", [])),
    }


def write_results_csv(records, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(build_csv_row(record))


def write_suite_outputs(suite_summary, suite_summary_path, suite_csv_path, final_csv_path):
    suite_summary["output_paths"] = {
        "suite_summary_file": str(suite_summary_path),
        "suite_csv_file": str(suite_csv_path),
        "final_csv_file": str(final_csv_path),
    }

    with open(suite_summary_path, "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2)

    records = suite_summary.get("records", [])
    write_results_csv(records, suite_csv_path)
    write_results_csv(records, final_csv_path)


def main():
    args = parse_args()
    project_root = get_project_root()
    eval_script = get_eval_script(project_root)

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load_in_4bit or --load_in_8bit.")

    dataset_files = collect_dataset_files(args)
    runs = build_runs(args)
    if not runs:
        raise ValueError("No experiment runs selected with the current filters.")

    suite_started_at = datetime.now().isoformat()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    suite_log_dir = output_root / "_suite_runs"
    suite_log_dir.mkdir(parents=True, exist_ok=True)
    suite_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_summary_path = suite_log_dir / f"suite_{suite_stamp}.json"
    suite_csv_path = suite_log_dir / f"suite_{suite_stamp}.csv"
    final_csv_path = output_root / "final_summary.csv"
    args.output_dir = str(output_root)

    print(f"Datasets: {len(dataset_files)}")
    print(f"Experiment variants: {len(runs)}")
    print(f"Backend: {args.backend}")
    print(
        "Note: Llama-3.1-8B-Instruct requires collaborators to have accepted the model license on Hugging Face."
    )

    records = []
    failure_count = 0
    for dataset_path in dataset_files:
        for run in runs:
            run_name = (
                f"{run['family']}:{run['variant']}:{run['prompt_style']}"
                if run["seed"] is None
                else f"{run['family']}:{run['variant']}:seed{run['seed']}:{run['prompt_style']}"
            )
            cmd = build_command(eval_script, args, dataset_path, run)
            print("\n" + "=" * 80)
            print(f"Running {run_name} on {dataset_path.name}")
            print(" ".join(cmd))

            record = {
                "dataset_path": str(dataset_path),
                "family": run["family"],
                "variant": run["variant"],
                "prompt_style": run["prompt_style"],
                "seed": run["seed"],
                "model": run["model"],
                "adapter": run["adapter"],
                "backend": args.backend,
                "command": cmd,
            }
            results_file, summary_file = infer_eval_output_paths(
                output_root, dataset_path, run, args.backend
            )
            record["results_file"] = str(results_file)
            record["summary_file"] = str(summary_file)

            if args.dry_run:
                record["status"] = "dry_run"
                records.append(record)
                continue

            completed = subprocess.run(cmd, cwd=str(project_root))
            record["returncode"] = completed.returncode
            record["status"] = "ok" if completed.returncode == 0 else "failed"
            records.append(record)

            if completed.returncode != 0:
                failure_count += 1
                if not args.continue_on_error:
                    suite_summary = {
                        "started_at": suite_started_at,
                        "finished_at": datetime.now().isoformat(),
                        "backend": args.backend,
                        "output_dir": str(output_root),
                        "dataset_files": [str(path) for path in dataset_files],
                        "num_runs_requested": len(dataset_files) * len(runs),
                        "num_failures": failure_count,
                        "records": records,
                    }
                    write_suite_outputs(
                        suite_summary,
                        suite_summary_path,
                        suite_csv_path,
                        final_csv_path,
                    )
                    raise SystemExit(completed.returncode)

    suite_summary = {
        "started_at": suite_started_at,
        "finished_at": datetime.now().isoformat(),
        "backend": args.backend,
        "output_dir": str(output_root),
        "dataset_files": [str(path) for path in dataset_files],
        "num_runs_requested": len(dataset_files) * len(runs),
        "num_failures": failure_count,
        "records": records,
    }
    write_suite_outputs(
        suite_summary,
        suite_summary_path,
        suite_csv_path,
        final_csv_path,
    )

    print("\n" + "=" * 80)
    print(f"Suite summary saved to: {suite_summary_path}")
    print(f"Suite CSV saved to: {suite_csv_path}")
    print(f"Final CSV saved to: {final_csv_path}")
    print(f"Failures: {failure_count}")

    if failure_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
