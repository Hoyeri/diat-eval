# DIAT Evaluation Package

This repository provides a simple script for evaluating SFT and DIAT model variants on your own multiple-choice dataset.

The trained adapters are already uploaded to Hugging Face. You do not need to train models or manually download adapter files. The evaluation script loads the required base models and adapters, runs inference on the dataset files you provide, and writes the results under `./results`.

## What This Runs

The main script is:

```text
scripts/run_all_diat_experiments.py
```

By default, it evaluates every `.json` dataset file under `./dataset` using the full model suite:

| family | base model | evaluated variants |
| --- | --- | --- |
| `qwen` | `Qwen/Qwen3-8B` | `base`, `sft` seeds 42/43/44, `diat` seeds 42/43/44 |
| `llama` | `meta-llama/Llama-3.1-8B-Instruct` | `base`, `sft` seeds 42/43/44, `diat` seeds 42/43/44 |

For the SFT and DIAT variants, the script loads the corresponding Hugging Face adapter repositories automatically.

## Quick Start

You only need this workflow:

```bash
git clone https://github.com/Hoyeri/diat-eval.git
cd diat-eval

# Optional but recommended: create and activate your own environment first.
# conda create -n diat-eval python=3.10 -y
# conda activate diat-eval

pip install tqdm torch transformers peft huggingface_hub
pip install vllm

# Before running the full suite, make sure your Hugging Face account has access
# to meta-llama/Llama-3.1-8B-Instruct:
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
huggingface-cli login

# Remove the example files, then place your real dataset file(s) here.
rm -f dataset/*.json dataset/*.jsonl
cp /path/to/your_dataset.json dataset/

python scripts/run_all_diat_experiments.py \
  --backend vllm \
  --dataset_dir dataset \
  --output_dir results
```

After the run finishes, check:

```text
results/final_summary.csv
```

## Step 1: Clone the Repository

Clone this repository and move into the repository root:

```bash
git clone https://github.com/Hoyeri/diat-eval.git
cd diat-eval
```

All commands below should be run from the `diat-eval` directory.

## Step 2: Prepare Your Dataset

Put your evaluation dataset files under:

```text
dataset/
```

The repository includes sample files only to show the expected format. Before running the full suite on your real data, remove the sample files from `dataset/` and copy your actual dataset files into that directory.

Example:

```bash
rm -f dataset/*.json dataset/*.jsonl
cp /path/to/your_dataset.json dataset/
```

This matters because `run_all_diat_experiments.py` evaluates every `.json` and `.jsonl` file under `dataset/` when you use `--dataset_dir dataset`. If sample files are still there, they will also be evaluated.

If you prefer to evaluate one specific file without deleting the samples, pass it explicitly:

```bash
python scripts/run_all_diat_experiments.py \
  --backend vllm \
  --dataset_path dataset/your_dataset.json \
  --output_dir results
```

## Dataset Format

The evaluator expects a multiple-choice dataset in `.json` or `.jsonl` format. The recommended format is a `.json` file containing a JSON array of question objects:

```json
[
  {
    "id": 0,
    "question": "What is 2 + 2?",
    "opa": "3",
    "opb": "4",
    "answer": "4",
    "answer_idx": "B"
  },
  {
    "id": 1,
    "question": "Which planet is known as the Red Planet?",
    "opa": "Venus",
    "opb": "Jupiter",
    "opc": "Mars",
    "opd": "Saturn",
    "answer": "Mars",
    "answer_idx": "C"
  }
]
```

Rules:

- `question` is required.
- Options must be named `opa`, `opb`, `opc`, ..., up to `opg`.
- Each question must have 2 to 7 options.
- Option fields must be contiguous. For example, `opa`, `opb`, `opd` without `opc` is invalid.
- `answer_idx` is recommended and should be the correct letter, such as `"A"`, `"B"`, or `"C"`.
- If `answer_idx` is missing, the evaluator tries to match the `answer` text against the option texts.

## Step 3: Install Dependencies

Install the common dependencies:

```bash
pip install tqdm torch transformers peft huggingface_hub
```

If you plan to use the vLLM backend:

```bash
pip install vllm
```

If you plan to use the Hugging Face backend with 4-bit or 8-bit loading:

```bash
pip install bitsandbytes accelerate
```

Authenticate with Hugging Face before running evaluation:

```bash
huggingface-cli login
```

Important: `meta-llama/Llama-3.1-8B-Instruct` requires access approval before evaluation. Request access at [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), then log in with the approved Hugging Face account.

## Step 4: Choose a Backend

This repository supports two evaluation backends: `vllm` and `hf`.

Use `vllm` when:

- you have a GPU environment where vLLM installs cleanly;
- you want to run the full suite faster;
- you are running the main evaluation rather than debugging model loading.

Recommended full-suite command:

```bash
python scripts/run_all_diat_experiments.py \
  --backend vllm \
  --dataset_dir dataset \
  --output_dir results \
  --batch_size 32 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.9
```

Use `hf` when:

- vLLM is not available in your environment;
- you want the standard Transformers/PEFT loading path;
- you are debugging a small run or checking compatibility.

Recommended Hugging Face command:

```bash
python scripts/run_all_diat_experiments.py \
  --backend hf \
  --dataset_dir dataset \
  --output_dir results \
  --load_in_4bit
```

The `hf` backend can be slower, especially for the full model suite. `--load_in_4bit` is useful when GPU memory is limited.

## Step 5: Run the Full Evaluation

After your real dataset files are in `dataset/`, run:

```bash
python scripts/run_all_diat_experiments.py \
  --backend vllm \
  --dataset_dir dataset \
  --output_dir results
```

This evaluates all configured base, SFT, and DIAT models on every `.json` file in `dataset/`.

## Results

All generated outputs are written under:

```text
results/
```

The most useful file is:

```text
results/final_summary.csv
```

This file contains one row per completed evaluation run, including the dataset name, model family, variant, seed, backend, accuracy, number correct, and total number of evaluated samples.

Each individual run also writes:

```text
results/<base_model_name>/base/<backend>/<dataset_tag>/
  test_results.jsonl
  summary.json

results/<base_model_name>/tuned_<adapter_name>/<backend>/<dataset_tag>/
  test_results.jsonl
  summary.json
```

The full suite additionally writes timestamped metadata:

```text
results/_suite_runs/suite_<timestamp>.json
results/_suite_runs/suite_<timestamp>.csv
```

Use `results/final_summary.csv` for the latest run summary. Use the timestamped files under `results/_suite_runs/` if you need to inspect previous runs.

## Notes

- If a run fails after some models have completed, partial results and suite metadata are still written under `results/`.
