# CFC-LLMs

Official release for reproducing the experiments in the CVPR 2026 paper "Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling".

## Public Release Workflow

1. Clone the GitHub repository:

```bash
git clone https://github.com/FlynnYe/CFC-LLMs.git
cd CFC-LLMs
```

2. Download the data from Google Drive.
3. Place the downloaded files under `data/raw/` using the layout in [data/README.md](data/README.md).
4. Run the reproduction script.

## Repository Layout

- `synthetic/`: synthetic pipeline
- `triviaqa/`: TriviaQA evaluation
- `gsm8k/`: GSM8K evaluation
- `flick8k/`: Flickr8k evaluation
- `scripts/`: top-level reproduction, figure generation, normalization, and verification utilities

## Dependencies

```bash
pip install -r requirements.txt
```

## External Data

Large scored sampling files are not included in the GitHub repository. Download the archive from Google Drive, place it at the repository root, and extract it so that it creates `data/raw/` using the layout in [data/README.md](data/README.md), or pass explicit paths to `scripts/reproduce_all.py`.

Google Drive archive:
- [CFC-LLMs_data_raw.tar.gz](https://drive.google.com/file/d/1XT5Otn7ZWSoIj3Ct7u9sEjgqx8RUzP7I/view?usp=sharing)

Extract it at the repository root:

```bash
tar -xzf CFC-LLMs_data_raw.tar.gz
```

## Reproduce All Results

From the repository root:

```bash
python scripts/reproduce_all.py \
  --triviaqa-base-dir data/raw/triviaqa \
  --gsm8k-score-dir data/raw/gsm8k \
  --flickr-coverage-cache data/raw/flickr8k/cached_coverage.json \
  --flickr-candidate-cache data/raw/flickr8k/cached_candidates.json \
  --verify
```

## Reproduce Individual Parts

Synthetic only:

```bash
python synthetic/reproduce_paper_synthetic.py --output-dir outputs/synthetic
```

TriviaQA only:

```bash
python triviaqa/evaluate_paper_setting.py \
  --base-dir data/raw/triviaqa \
  --model-glob meta-llama__Llama-2-13b-hf \
  --run-dir outputs/triviaqa/paper_setting
```

GSM8K only:

```bash
python gsm8k/evaluate_paper_setting.py \
  --score-dir data/raw/gsm8k \
  --run-dir outputs/gsm8k/paper_setting
```

Flickr8k only:

```bash
python flick8k/evaluate_paper_setting.py \
  --coverage-cache data/raw/flickr8k/cached_coverage.json \
  --candidate-cache data/raw/flickr8k/cached_candidates.json \
  --run-dir outputs/flick8k/paper_setting
```
## Cite Our Paper

If you find this repository useful for your research, please cite our paper:

```bibtex
@inproceedings{ye2026conditional,
  title={Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling},
  author={Ye, Kai and Pan, Qingtao and Li, Shuo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3627--3635},
  year={2026}
}
```

