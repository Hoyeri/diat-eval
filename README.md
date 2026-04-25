# DIAT Evaluation Package

This repository provides a simple script for evaluating base, SFT, and DIAT model variants on your own multiple-choice dataset.

The trained adapters are already uploaded to Hugging Face. You do not need to train models or manually download adapter files. The evaluation script loads the required base models and adapters, runs inference on the dataset files you provide, and writes the results under `./results`.

## What This Runs

The main script is:

```text
scripts/run_all_diat_experiments.py
```

By default, it evaluates every `.json` or `.jsonl` dataset file under `./dataset` using the full model suite:

| family | base model | evaluated variants |
| --- | --- | --- |
| `qwen` | `Qwen/Qwen3-8B` | `base` with `default` and `instruction` prompts, `sft` seeds 42/43/44 with `default`, `diat` seeds 42/43/44 with `default` |
| `llama` | `meta-llama/Llama-3.1-8B-Instruct` | `base` with `default` and `instruction` prompts, `sft` seeds 42/43/44 with `default`, `diat` seeds 42/43/44 with `default` |

For the SFT and DIAT variants, the script loads the corresponding Hugging Face adapter repositories automatically.
For base models, the suite supports two prompt styles: `default` and `instruction`. The `instruction` prompt explicitly tells the model to ignore noise such as trap details, misleading clues, and diagnostically irrelevant information.

## Quick Start

You only need this workflow:

```bash
git clone https://github.com/Hoyeri/diat-eval.git
cd diat-eval

# Optional but recommended: create and activate your own environment first.
# conda create -n diat-eval python=3.10 -y
# conda activate diat-eval

pip install tqdm torch transformers peft huggingface_hub

# Recommended default backend: install vLLM with a method that matches
# your local CUDA / driver environment.
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

If your datasets live somewhere else, pass that directory explicitly with `--dataset_dir /path/to/your_datasets`.

If you prefer to evaluate one specific file without deleting the samples, pass it explicitly:

```bash
python scripts/run_all_diat_experiments.py \
  --backend vllm \
  --dataset_path dataset/your_dataset.json \
  --output_dir results
```

## Dataset Format

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

Recommended default backend: `vllm`

```bash
pip install vllm
```

`vllm`, `torch`, and `bitsandbytes` are sensitive to your local CUDA and driver environment. Install versions that match your system rather than assuming one pinned combination will work everywhere.

If `vllm` is unavailable or does not run cleanly in your environment, use the Hugging Face fallback backend:

```bash
pip install accelerate
```

If you plan to use the Hugging Face backend with `--load_in_4bit` or `--load_in_8bit` as a lower-memory fallback:

```bash
pip install bitsandbytes
```

Authenticate with Hugging Face before running evaluation:

```bash
huggingface-cli login
```

Important: `meta-llama/Llama-3.1-8B-Instruct` requires access approval before evaluation. Request access at [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), then log in with the approved Hugging Face account.

## Step 4: Choose a Backend

This repository supports two evaluation backends: `vllm` and `hf`.

Use `vllm` when:

- it installs cleanly in your GPU environment;
- you want the recommended default path for full evaluation;
- you want faster evaluation throughput than the Hugging Face fallback.

Recommended vLLM full-suite command:

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

- vLLM is not available or does not run cleanly in your environment;
- you want the standard Transformers/PEFT loading path;
- you are debugging a small run or checking compatibility.

Recommended Hugging Face full-suite command:

```bash
python scripts/run_all_diat_experiments.py \
  --backend hf \
  --dataset_dir dataset \
  --output_dir results
```

The `hf` backend is the fallback path when `vllm` is unavailable, and it uses standard Transformers/PEFT loading with default `bf16` evaluation. It can be slower than `vllm`, especially for the full model suite. If the model does not fit in GPU memory or fails to load in the default mode, rerun with `--load_in_4bit` or `--load_in_8bit` as a lower-memory fallback.

## Step 5: Run the Full Evaluation

After your real dataset files are in `dataset/`, run one of the full-suite commands from Step 4.

That run evaluates all configured base, SFT, and DIAT models on every `.json` or `.jsonl` file in `dataset/`.
By default, each base model is evaluated twice, once with the original `default` prompt and once with the `instruction` prompt. Adapter variants remain on the `default` prompt.

## Prompt Styles

The suite-level script uses:

- `default` and `instruction` for `base` runs
- `default` only for `sft` and `diat` runs

If you run the single evaluator directly instead of the suite runner, choose the prompt explicitly:

```bash
python src/eval_diat.py \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --dataset_path dataset/your_dataset.json \
  --output_dir results \
  --prompt_style instruction
```

## Results

All generated outputs are written under:

```text
results/
```

The most useful file is:

```text
results/final_summary.csv
```

This file contains one row per completed evaluation run, including the dataset name, model family, variant, prompt style, seed, backend, accuracy, number correct, and total number of evaluated samples.

If you want to share evaluation outputs after the run finishes, compress the `results/` directory and share the archive (for example, via Google Drive). 

Each individual run also writes:

```text
results/<model_name>/<backend>/<prompt_style>/<dataset_tag>/
  test_results.jsonl
  summary.json
```

Where:

- `<model_name>` is either `<base_model_name>/base` or `<base_model_name>/tuned_<adapter_name>`
- `<prompt_style>` is `default` or `instruction`

Examples:

```text
results/Qwen_Qwen3-8B/base/vllm/default/<dataset_tag>/
results/Qwen_Qwen3-8B/base/vllm/instruction/<dataset_tag>/
results/Qwen_Qwen3-8B/tuned_qwen3-8b-sft-seed42/vllm/default/<dataset_tag>/
```

All runs store `prompt_style` explicitly as a directory level: `<model_name>/<backend>/<prompt_style>/<dataset_tag>/`. Base runs may use `default` or `instruction`, while adapter runs currently use `default`.

The full suite additionally writes timestamped metadata:

```text
results/_suite_runs/suite_<timestamp>.json
results/_suite_runs/suite_<timestamp>.csv
```

Use `results/final_summary.csv` for the latest run summary. Use the timestamped files under `results/_suite_runs/` if you need to inspect previous runs.

## Notes

- If a run fails after some models have completed, partial results and suite metadata are still written under `results/`.
