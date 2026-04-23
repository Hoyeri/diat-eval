# Dataset Folder

Place your real evaluation dataset file(s) in this folder.

The evaluation script reads `.json` files from this directory when you run:

```bash
python scripts/run_all_diat_experiments.py \
  --backend vllm \
  --dataset_dir dataset \
  --output_dir results
```

Before running evaluation, remove the sample file and copy your real dataset file(s) here:

```bash
rm -f dataset/*.json dataset/*.jsonl
cp /path/to/your_dataset.json dataset/
```
