# Dataset Folder

Place your real evaluation dataset file(s) in this folder.

The evaluation script reads `.json` and `.jsonl` files from this directory when you run:

```bash
python scripts/run_all_diat_experiments.py
```

Before running evaluation, remove the sample files and copy your real dataset file(s) here:

```bash
rm -f dataset/*.json dataset/*.jsonl
cp /path/to/your_dataset.json dataset/
```
