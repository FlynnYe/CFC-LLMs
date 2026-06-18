# External Data

This release expects the large scored sampling files to be downloaded separately and placed under `data/raw/`.

Expected layout:

```text
data/raw/
  triviaqa/
    seed_0/
      meta-llama__Llama-2-13b-hf/
        samples*_scored.jsonl
    ...
    seed_19/
      meta-llama__Llama-2-13b-hf/
        samples*_scored.jsonl
  gsm8k/
    gsm8k_with_rm_*.jsonl
  flick8k/
    cached_coverage.json
    cached_candidates.json
```

Google Drive archive:

- [CFC-LLMs_data_raw.tar.gz](https://drive.google.com/file/d/1XT5Otn7ZWSoIj3Ct7u9sEjgqx8RUzP7I/view?usp=sharing)

After download, extract it at the repository root:

```bash
tar -xzf CFC-LLMs_data_raw.tar.gz
```
