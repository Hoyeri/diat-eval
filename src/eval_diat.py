import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
from datetime import datetime

from tqdm import tqdm


OPTION_KEYS = ["opa", "opb", "opc", "opd", "ope", "opf", "opg"]
MIN_OPTIONS = 2
MAX_OPTIONS = 7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HF/PEFT or vLLM models on MCQ datasets"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--backend", type=str, choices=["hf", "vllm"], default="hf")
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load HF backend weights in 4-bit quantization",
    )
    quant_group.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load HF backend weights in 8-bit quantization",
    )
    parser.add_argument("--dataset_path", type=str, default="./data/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--debug_n", type=int, default=5, help="Print raw outputs for first N samples"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of prompts per vLLM generate call",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization target",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Optional vLLM max model length override",
    )
    return parser.parse_args()


def sanitize_name(name: str):
    return name.replace("/", "_")


def infer_model_type(adapter_path=None):
    return "peft_adapter" if adapter_path else "base_model"


def build_prompt(question_data, valid_letters):
    letters_str = ", ".join(valid_letters)
    system_prompt = (
        "You are a medical AI assistant used for decision-making in a clinical setting.\n"
        "Reason step-by-step through this problem, and provide your final answer\n"
        f"(Return only the letter ({letters_str}) of your choice and nothing else.)"
    )

    options_block = "\n".join(
        f"{letter}) {question_data['options'][letter]}" for letter in valid_letters
    )

    return f"""{system_prompt}

Question:
{question_data['question']}

Options:
{options_block}

Answer:
"""


def print_debug_output(generated_text):
    print("\n" + "=" * 60)
    print("[DEBUG] FULL MODEL OUTPUT (raw):")
    print(generated_text)
    print("=" * 60 + "\n")


def get_installed_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_runtime_versions():
    versions = {"python": platform.python_version()}
    for package_name in ["torch", "transformers", "peft", "vllm"]:
        version = get_installed_version(package_name)
        if version is not None:
            versions[package_name] = version
    return versions


def resolve_hf_quantization_mode(args):
    if args.load_in_4bit:
        return "4bit", "user_requested"
    if args.load_in_8bit:
        return "8bit", "user_requested"
    if args.adapter:
        return "4bit", "default_for_adapter"
    return "bf16", "default_for_base"


def load_hf_inference_model(args):
    base_model = args.model
    adapter_path = args.adapter
    print("Loading model...")
    print(" - Backend: hf")
    print(f" - Base model: {base_model}")
    print(f" - Adapter: {adapter_path if adapter_path else 'None (base only)'}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "HF backend requires `torch` and `transformers` to be installed."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_mode, quantization_source = resolve_hf_quantization_mode(args)
    print(f" - HF load mode: {quantization_mode} ({quantization_source})")

    model_load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if quantization_mode == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_load_kwargs["quantization_config"] = bnb_config
    elif quantization_mode == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model_load_kwargs["quantization_config"] = bnb_config
    else:
        model_load_kwargs["torch_dtype"] = torch.bfloat16

    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "HF backend with `--adapter` requires the `peft` package."
            ) from exc

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_load_kwargs,
        )

        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_load_kwargs,
        )

    model.eval()
    backend_meta = {
        "eos_token_id": tokenizer.eos_token_id,
        "hf_quantization_mode": quantization_mode,
        "hf_quantization_source": quantization_source,
    }
    return model, tokenizer, infer_model_type(adapter_path), backend_meta


def build_vllm_import_error(exc):
    msg = "vLLM backend requires the `vllm` package to be importable."
    spec = importlib.util.find_spec("vllm")
    search_paths = list(getattr(spec, "submodule_search_locations", []) or [])
    if search_paths:
        msg += f" Python currently resolves `vllm` to: {search_paths}."
    msg += f" Original error: {exc}"
    return msg


def resolve_vllm_adapter_path(adapter_ref):
    if adapter_ref is None or os.path.exists(adapter_ref):
        return adapter_ref

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "vLLM adapter references that are not local paths require "
            "`huggingface_hub` to be installed so the adapter can be downloaded."
        ) from exc

    return snapshot_download(repo_id=adapter_ref)


def load_vllm_backend(args):
    print("Loading model...")
    print(" - Backend: vllm")
    print(f" - Base model: {args.model}")
    print(f" - Adapter: {args.adapter if args.adapter else 'None (base only)'}")

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError(build_vllm_import_error(exc)) from exc

    engine_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
    }
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    if args.adapter:
        engine_kwargs["enable_lora"] = True

    llm = LLM(**engine_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    lora_request = None
    if args.adapter:
        adapter_path = resolve_vllm_adapter_path(args.adapter)
        try:
            from vllm.lora.request import LoRARequest
        except Exception as exc:
            raise RuntimeError(
                "vLLM adapter mode requires LoRA support to be importable. "
                "This path expects a PEFT-style adapter directory and a model "
                "that supports LoRA in vLLM."
            ) from exc
        adapter_name = sanitize_name(os.path.basename(os.path.normpath(adapter_path)))
        lora_request = LoRARequest(adapter_name or "adapter", 1, adapter_path)

    backend_meta = {
        "temperature": 0.0,
        "batch_size": args.batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }
    return llm, sampling_params, lora_request, infer_model_type(args.adapter), backend_meta


def parse_answer_letter(generated_text: str, valid_letters):


    text = (generated_text or "").strip()
    if not text:
        return None
    letters_pat = "".join(re.escape(letter) for letter in valid_letters)

    explicit = re.findall(
        r"(?:"
        r"correct\s+answer\s*(?:is|:)?|"
        r"the\s+answer\s*(?:is|:)?|"
        r"answer\s*(?:is|:)?"
        rf")\s*[:\-]?\s*\(?\s*([{letters_pat}])\s*\)?",
        text,
        re.IGNORECASE,
    )
    if explicit:
        return explicit[-1].upper()

    last_line = text.splitlines()[-1].strip()
    m = re.match(
        rf"^Answer:\s*\(?\s*([{letters_pat}])\s*\)?\s*\.?\s*$",
        last_line,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    line_letters = re.findall(
        rf"^\s*\(?\s*([{letters_pat}])\s*\)?\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if line_letters:
        return line_letters[-1].upper()

    punct_letters = re.findall(
        rf"\b([{letters_pat}])\b\s*[\)\].:]",
        text,
        re.IGNORECASE
    )
    if punct_letters:
        return punct_letters[-1].upper()

    return None


def normalize_answer_idx(answer_idx, valid_letters, options=None, answer_text=None):
    if answer_idx is not None:
        if isinstance(answer_idx, int):
            if 0 <= answer_idx < len(valid_letters):
                return valid_letters[answer_idx]
        if isinstance(answer_idx, str):
            s = answer_idx.strip().upper()
            if s in valid_letters:
                return s
            if s.isdigit():
                idx = int(s)
                if 0 <= idx < len(valid_letters):
                    return valid_letters[idx]

    if answer_text is not None and options:
        target = str(answer_text).strip()
        for letter, text in options.items():
            if target == str(text).strip():
                return letter
        target_lower = target.lower()
        for letter, text in options.items():
            if target_lower == str(text).strip().lower():
                return letter

    return None


def normalize_options_from_sample(sample):
    options = []
    seen_gap = False

    for key in OPTION_KEYS:
        if key in sample and sample[key] is not None and str(sample[key]).strip() != "":
            if seen_gap:
                raise ValueError(
                    "Option keys must be contiguous (e.g., opa, opb, opc...)."
                )
            options.append(str(sample[key]))
        else:
            if options:
                seen_gap = True

    if not (MIN_OPTIONS <= len(options) <= MAX_OPTIONS):
        raise ValueError(
            f"Option count must be between {MIN_OPTIONS} and {MAX_OPTIONS}, got {len(options)}."
        )

    letters = [chr(ord("A") + i) for i in range(len(options))]
    return {letters[i]: options[i] for i in range(len(options))}, letters


def normalize_sample(sample):
    if not isinstance(sample, dict):
        raise ValueError("Each sample must be a JSON object.")

    question = sample.get("question")
    if question is None:
        raise ValueError("Missing 'question' field.")

    options, valid_letters = normalize_options_from_sample(sample)

    gold_answer = normalize_answer_idx(
        sample.get("answer_idx"),
        valid_letters=valid_letters,
        options=options,
        answer_text=sample.get("answer"),
    )

    return question, options, valid_letters, gold_answer


def get_hf_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        if hasattr(model, "device"):
            return model.device
        raise RuntimeError("Unable to determine model input device.") from exc


def generate_hf_response(model, tokenizer, prompt, max_new_tokens=1024):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = get_hf_input_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    new_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return new_text.strip()


def batch_items(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    if content.lstrip().startswith("["):
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON array input must be a list of objects.")
        return data

    return [json.loads(line) for line in content.splitlines() if line.strip()]


def prepare_samples(raw_data):
    prepared_samples = []
    for sample in raw_data:
        question, options, valid_letters, gold_answer = normalize_sample(sample)
        prepared_samples.append(
            {
                "question": question,
                "options": options,
                "valid_letters": valid_letters,
                "gold_answer": gold_answer,
                "prompt": build_prompt(
                    {"question": question, "options": options}, valid_letters
                ),
            }
        )
    return prepared_samples


def generate_outputs_hf(args, prepared_samples):
    model, tokenizer, model_type, backend_meta = load_hf_inference_model(args)

    generated_outputs = []
    for i, sample in enumerate(tqdm(prepared_samples, desc="Evaluating", unit="sample")):
        generated = generate_hf_response(
            model,
            tokenizer,
            sample["prompt"],
            max_new_tokens=args.max_new_tokens,
        )
        if i < args.debug_n:
            print_debug_output(generated)
        generated_outputs.append(generated)

    return generated_outputs, model_type, backend_meta


def generate_outputs_vllm(args, prepared_samples):
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer.")

    llm, sampling_params, lora_request, model_type, backend_meta = load_vllm_backend(args)

    prompts = [sample["prompt"] for sample in prepared_samples]
    generated_outputs = []

    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    for _, prompt_batch in tqdm(
        batch_items(prompts, args.batch_size),
        total=num_batches,
        desc="Evaluating",
        unit="batch",
    ):
        batch_outputs = llm.generate(
            prompt_batch,
            sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
        )
        for output in batch_outputs:
            generated = output.outputs[0].text.strip() if output.outputs else ""
            sample_idx = len(generated_outputs)
            if sample_idx < args.debug_n:
                print_debug_output(generated)
            generated_outputs.append(generated)

    return generated_outputs, model_type, backend_meta


def run_evaluation(args):
    if args.backend != "hf" and (args.load_in_4bit or args.load_in_8bit):
        raise ValueError("--load_in_4bit/--load_in_8bit are only supported with --backend hf.")

    raw_data = load_dataset(args.dataset_path)
    print(f"Evaluating {len(raw_data)} samples")
    prepared_samples = prepare_samples(raw_data)

    base_name = sanitize_name(args.model)
    if args.adapter:
        adapter_name = sanitize_name(os.path.basename(os.path.normpath(args.adapter)))
        model_name = f"{base_name}/tuned_{adapter_name}"
    else:
        model_name = f"{base_name}/base"

    dataset_tag = args.dataset_path.replace(os.sep, "_")
    output_dir = os.path.join(args.output_dir, model_name, args.backend, dataset_tag)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "test_results.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")

    if args.backend == "hf":
        generated_outputs, model_type, backend_meta = generate_outputs_hf(
            args, prepared_samples
        )
    else:
        generated_outputs, model_type, backend_meta = generate_outputs_vllm(
            args, prepared_samples
        )

    model_info = {
        "base_model": args.model,
        "adapter": args.adapter,
        "model_type": model_type,
        "model_name": model_name,
        "backend": args.backend,
        "dataset_path": args.dataset_path,
    }
    if "hf_quantization_mode" in backend_meta:
        model_info["hf_quantization_mode"] = backend_meta["hf_quantization_mode"]

    if len(generated_outputs) != len(prepared_samples):
        raise RuntimeError(
            f"Generated {len(generated_outputs)} outputs for {len(prepared_samples)} samples."
        )

    results = []
    correct = 0
    total = 0

    for i, (sample, generated) in enumerate(zip(prepared_samples, generated_outputs)):
        question = sample["question"]
        options = sample["options"]
        valid_letters = sample["valid_letters"]
        gold_answer = sample["gold_answer"]

        pred_answer = parse_answer_letter(generated, valid_letters)
        if pred_answer is not None:
            pred_answer = pred_answer.upper()

        is_correct = pred_answer == gold_answer if gold_answer else None
        status = "NO_GOLD" if is_correct is None else ("CORRECT" if is_correct else "WRONG")

        if is_correct is not None:
            correct += int(is_correct)
            total += 1

        print(
            f"[{i:04d}] "
            f"PRED={pred_answer} | "
            f"GT={gold_answer} | "
            f"{status}"
        )

        results.append(
            {
                **model_info,
                "question": question,
                "options": options,
                "generated_text": generated,
                "pred_answer": pred_answer,
                "gold_answer": gold_answer,
                "is_correct": is_correct,
            }
        )

    acc = correct / total if total > 0 else 0.0

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
    }
    if args.backend == "hf":
        generation_config.update(
            {
                "do_sample": False,
                "eos_token_id": backend_meta["eos_token_id"],
                "hf_quantization_mode": backend_meta["hf_quantization_mode"],
                "hf_quantization_source": backend_meta["hf_quantization_source"],
            }
        )
    else:
        generation_config.update(
            {
                "temperature": backend_meta["temperature"],
                "batch_size": backend_meta["batch_size"],
                "tensor_parallel_size": backend_meta["tensor_parallel_size"],
                "gpu_memory_utilization": backend_meta["gpu_memory_utilization"],
            }
        )
        if backend_meta["max_model_len"] is not None:
            generation_config["max_model_len"] = backend_meta["max_model_len"]

    summary = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "base_model": args.model,
            "adapter": args.adapter,
            "backend": args.backend,
            "model_type": model_type,
            "dataset_path": args.dataset_path,
            "num_samples": len(raw_data),
        },
        "generation_config": generation_config,
        "runtime_versions": get_runtime_versions(),
        "results": {
            "accuracy": acc,
            "correct": correct,
            "total": total,
        },
        "output_paths": {
            "results_file": output_path,
            "summary_file": summary_path,
        },
    }
    if args.backend == "vllm" and args.adapter:
        summary["compatibility_notes"] = [
            "vLLM LoRA evaluation expects a PEFT-style adapter directory and a base model that supports LoRA in vLLM."
        ]

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n====================")
    print(f"Final Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"Results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print("====================")


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
