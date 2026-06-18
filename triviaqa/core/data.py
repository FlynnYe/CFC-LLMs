from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import string
from typing import Iterable

import numpy as np


ARTIFACT_VERSION = 'v2'


@dataclass(frozen=True)
class CandidateRecord:
    text: str
    score: float
    correct: bool
    freq: int

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'score': float(self.score),
            'correct': bool(self.correct),
            'freq': int(self.freq),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> 'CandidateRecord':
        return cls(
            text=str(payload['text']),
            score=float(payload['score']),
            correct=bool(payload['correct']),
            freq=int(payload['freq']),
        )


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    question_id: str
    question: str
    candidates: tuple[CandidateRecord, ...]
    num_sample_seeds: int
    max_freq_share: float
    normalized_entropy: float
    unique_count: int
    min_score: float
    mean_score: float
    max_score: float
    success_score: float
    has_correct: bool

    def to_dict(self) -> dict:
        return {
            'doc_id': self.doc_id,
            'question_id': self.question_id,
            'question': self.question,
            'candidates': [candidate.to_dict() for candidate in self.candidates],
            'num_sample_seeds': int(self.num_sample_seeds),
            'max_freq_share': float(self.max_freq_share),
            'normalized_entropy': float(self.normalized_entropy),
            'unique_count': int(self.unique_count),
            'min_score': float(self.min_score),
            'mean_score': float(self.mean_score),
            'max_score': float(self.max_score),
            'success_score': float(self.success_score),
            'has_correct': bool(self.has_correct),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> 'DocumentRecord':
        return cls(
            doc_id=str(payload['doc_id']),
            question_id=str(payload.get('question_id', payload['doc_id'])),
            question=str(payload.get('question', '')),
            candidates=tuple(CandidateRecord.from_dict(item) for item in payload['candidates']),
            num_sample_seeds=int(payload['num_sample_seeds']),
            max_freq_share=float(payload['max_freq_share']),
            normalized_entropy=float(payload['normalized_entropy']),
            unique_count=int(payload['unique_count']),
            min_score=float(payload['min_score']),
            mean_score=float(payload['mean_score']),
            max_score=float(payload['max_score']),
            success_score=float(payload['success_score']),
            has_correct=bool(payload['has_correct']),
        )


@dataclass
class _CandidateAggregate:
    losses: list[float]
    correct: bool
    freq: int


def parse_int_spec(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    if ',' in raw:
        return [int(token.strip()) for token in raw.split(',') if token.strip()]
    if '-' in raw:
        start, end = raw.split('-', 1)
        return list(range(int(start), int(end) + 1))
    return [int(raw)]


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def normalize_text(text: str) -> str:
    lowered = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = [word for word in lowered.split() if word not in {'a', 'an', 'the'}]
    return ' '.join(words)


def extract_prediction_text(example: dict) -> str | None:
    filtered = example.get('filtered_resps')
    if isinstance(filtered, list) and filtered and isinstance(filtered[0], str):
        return filtered[0]
    responses = example.get('resps')
    if isinstance(responses, list) and responses:
        if isinstance(responses[0], list) and responses[0] and isinstance(responses[0][0], str):
            return responses[0][0]
        if isinstance(responses[0], str):
            return responses[0]
    return None


def score_from_example(example: dict) -> float | None:
    scores = example.get('scores') or {}
    if 'avg_logprob' not in scores:
        return None
    try:
        avg_logprob = float(scores['avg_logprob'])
    except Exception:
        return None
    return float(1.0 - math.exp(avg_logprob))


def find_latest_scored_file(base_dir: Path, sample_seed: int, model_glob: str) -> Path | None:
    seed_dir = base_dir / f'seed_{sample_seed}'
    patterns = [
        f'{model_glob}/samples*_prompt*_scored.jsonl',
        f'{model_glob}/samples*_scored.jsonl',
    ]
    for pattern in patterns:
        files = sorted(seed_dir.glob(pattern))
        if files:
            return files[-1]
    return None


def _aggregate_candidate_score(losses: list[float], string_agg: str) -> float:
    arr = np.asarray(losses, dtype=float)
    if arr.size == 0:
        return 1.0
    if string_agg == 'min':
        return float(np.min(arr))
    if string_agg == 'max':
        return float(np.max(arr))
    if string_agg in {'median', 'q50'}:
        return float(np.quantile(arr, 0.5))
    if string_agg == 'q25':
        return float(np.quantile(arr, 0.25))
    if string_agg == 'q75':
        return float(np.quantile(arr, 0.75))
    return float(np.mean(arr))


def _normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0.0]
    if probs.size <= 1:
        return 0.0
    entropy = float(-np.sum(probs * np.log(probs)))
    return float(entropy / math.log(probs.size))


def _cache_key(base_dir: Path, model_glob: str, sample_seeds: list[int], string_agg: str) -> str:
    payload = {
        'version': ARTIFACT_VERSION,
        'base_dir': str(base_dir.resolve()),
        'model_glob': model_glob,
        'sample_seeds': sample_seeds,
        'string_agg': string_agg,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:12]
    base_slug = base_dir.name.replace('/', '_')
    model_slug = model_glob.replace('/', '_').replace('*', 'star')
    return f'{base_slug}_{model_slug}_{digest}.jsonl.gz'


def build_records(
    base_dir: Path,
    model_glob: str,
    sample_seeds: list[int],
    string_agg: str,
) -> list[DocumentRecord]:
    by_doc: dict[str, dict] = {}
    expected_seed_count = len(sample_seeds)

    for sample_seed in sample_seeds:
        scored_path = find_latest_scored_file(base_dir, sample_seed=sample_seed, model_glob=model_glob)
        if scored_path is None:
            raise FileNotFoundError(f'No scored JSONL found for sample seed {sample_seed} under {base_dir}.')
        seen_doc_ids: set[str] = set()
        for example in _iter_jsonl(scored_path):
            raw_doc_id = example.get('doc_id')
            if raw_doc_id is None:
                raw_doc_id = example.get('doc', {}).get('question_id') or example.get('doc_hash')
            if raw_doc_id is None:
                continue
            doc_id = str(raw_doc_id)
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            prediction = extract_prediction_text(example)
            if not isinstance(prediction, str):
                continue
            normalized = normalize_text(prediction)
            if not normalized:
                continue
            score = score_from_example(example)
            if score is None:
                continue
            question = str(example.get('doc', {}).get('question', ''))
            question_id = str(example.get('doc', {}).get('question_id', doc_id))
            correct = bool(example.get('exact_match'))

            record = by_doc.setdefault(
                doc_id,
                {
                    'doc_id': doc_id,
                    'question_id': question_id,
                    'question': question,
                    'seed_ids': set(),
                    'candidates': {},
                },
            )
            record['seed_ids'].add(sample_seed)
            candidate_map: dict[str, _CandidateAggregate] = record['candidates']
            current = candidate_map.get(normalized)
            if current is None:
                candidate_map[normalized] = _CandidateAggregate(losses=[score], correct=correct, freq=1)
            else:
                current.losses.append(score)
                current.correct = bool(current.correct or correct)
                current.freq += 1

    complete_records: list[DocumentRecord] = []
    for payload in by_doc.values():
        if len(payload['seed_ids']) < expected_seed_count:
            continue
        candidate_list: list[CandidateRecord] = []
        freq_counts: list[int] = []
        weighted_scores: list[float] = []
        success_scores: list[float] = []
        for text, aggregate in payload['candidates'].items():
            score = _aggregate_candidate_score(aggregate.losses, string_agg=string_agg)
            candidate = CandidateRecord(
                text=text,
                score=score,
                correct=aggregate.correct,
                freq=aggregate.freq,
            )
            candidate_list.append(candidate)
            freq_counts.append(candidate.freq)
            weighted_scores.extend([candidate.score] * candidate.freq)
            if candidate.correct:
                success_scores.append(candidate.score)

        # Keep first-seen unique-string order from the raw samples. The truncated
        # CFC variant is defined as a post-filter prefix in this candidate order.
        if not candidate_list:
            continue
        freq_arr = np.asarray(freq_counts, dtype=float)
        weighted_arr = np.asarray(weighted_scores, dtype=float)
        max_freq_share = float(np.max(freq_arr) / np.sum(freq_arr)) if freq_arr.size else 0.0
        success_score = float(min(success_scores)) if success_scores else 1.0
        complete_records.append(
            DocumentRecord(
                doc_id=payload['doc_id'],
                question_id=payload['question_id'],
                question=payload['question'],
                candidates=tuple(candidate_list),
                num_sample_seeds=expected_seed_count,
                max_freq_share=max_freq_share,
                normalized_entropy=_normalized_entropy(freq_arr),
                unique_count=len(candidate_list),
                min_score=float(min(candidate.score for candidate in candidate_list)),
                mean_score=float(np.mean(weighted_arr)) if weighted_arr.size else 1.0,
                max_score=float(max(candidate.score for candidate in candidate_list)),
                success_score=success_score,
                has_correct=bool(success_scores),
            )
        )

    complete_records.sort(key=lambda record: (record.doc_id.isdigit(), int(record.doc_id) if record.doc_id.isdigit() else record.doc_id))
    return complete_records


def load_or_build_records(
    base_dir: str | Path,
    model_glob: str,
    sample_seeds: list[int],
    string_agg: str = 'mean',
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
) -> tuple[list[DocumentRecord], Path]:
    base_path = Path(base_dir)
    cache_root = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parents[1] / 'cache'
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / _cache_key(base_path, model_glob=model_glob, sample_seeds=sample_seeds, string_agg=string_agg)

    if cache_path.exists() and not force_rebuild:
        records: list[DocumentRecord] = []
        with gzip.open(cache_path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    records.append(DocumentRecord.from_dict(json.loads(line)))
        return records, cache_path

    records = build_records(base_path, model_glob=model_glob, sample_seeds=sample_seeds, string_agg=string_agg)
    with gzip.open(cache_path, 'wt', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict()))
            handle.write('\n')
    return records, cache_path


def split_records(records: list[DocumentRecord], num_docs: int, split_seed: int) -> tuple[list[DocumentRecord], list[DocumentRecord]]:
    chosen = list(records)
    rng = random.Random(split_seed)
    rng.shuffle(chosen)
    if num_docs > 0:
        chosen = chosen[: min(num_docs, len(chosen))]
    midpoint = len(chosen) // 2
    return chosen[:midpoint], chosen[midpoint:]
