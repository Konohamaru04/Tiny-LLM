# Public dataset snapshot

`configs/public_datasets.yaml` pins exact Hugging Face revisions. The downloader
uses the public Dataset Viewer API, verifies that each repository still resolves
to the expected commit, rate-limits requests, resumes interrupted downloads,
deduplicates normalized records, and writes SHA-256 checksums to
`data/public_sft/manifest.json`.

Snapshot prepared on 2026-07-24:

| Stage | Source | Pinned revision | Included | Terms noted by source |
|---|---|---|---:|---|
| Pretraining | [FineWeb-Edu sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | 5,000 documents | ODC-By 1.0 and Common Crawl terms |
| Reasoning SFT | [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | `e4e141ec9dea9f8326f4d347be56105859b2bd68` | 1,000 records | Apache-2.0 |
| Tool SFT | [Dolci-Instruct-SFT-Tool-Use](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-Tool-Use) | `dc042846f0f2de0f15eedae3d6ced04223ed47eb` | 2,000 records | ODC-By 1.0 and Ai2 guidelines |
| Thinking/tool SFT | [Tülu MAX SFT](https://huggingface.co/datasets/allenai/tmax-sft) | `9d6ab7471ffa2884b2ef0cd2b7e5e22a027ff1b4` | 300 records | ODC-By, Ai2 guidelines, and upstream model terms |

The combined deterministic SFT split contains 3,127 training and 173
validation examples. This is a practical starter subset, not a mirror of the
full upstream corpora.

The repository also includes 297 unique locally supplied Markdown documents in
pretraining. Their duplicate directory copies are removed by content hash.
Their provenance is not described by the Hugging Face manifest and must be
reviewed separately before redistribution.

## Refresh

```powershell
python scripts/download_public_datasets.py --config configs/public_datasets.yaml --stage all
```

Individual stages:

```powershell
python scripts/download_public_datasets.py --config configs/public_datasets.yaml --stage pretrain
python scripts/download_public_datasets.py --config configs/public_datasets.yaml --stage sft
```

Changing a source revision is an explicit review step: update the pinned commit
and rerun validation after reading the new dataset card and terms. The script
fails closed instead of silently accepting upstream drift.

The license column is an engineering inventory, not legal advice. Dataset
records can contain material governed by upstream websites, model providers,
or other rights in addition to the dataset repository's declared license.
