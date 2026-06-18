from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable

import numpy as np


ARTIFACT_VERSION = 'v1'


@dataclass(frozen=True)
class CandidateRecord:
    text: str
    score: float
    correct: bool
    rank: int

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'score': float(self.score),
            'correct': bool(self.correct),
            'rank': int(self.rank),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> 'CandidateRecord':
        return cls(
            text=str(payload['text']),
            score=float(payload['score']),
            correct=bool(payload['correct']),
            rank=int(payload['rank']),
        )


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    question_id: str
    question: str
    gt: str
    candidates: tuple[CandidateRecord, ...]
    num_sample_seeds: int
    max_freq_share: float
    normalized_entropy: float
    unique_count: int
    prompt_length: float
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
            'gt': self.gt,
            'candidates': [candidate.to_dict() for candidate in self.candidates],
            'num_sample_seeds': int(self.num_sample_seeds),
            'max_freq_share': float(self.max_freq_share),
            'normalized_entropy': float(self.normalized_entropy),
            'unique_count': int(self.unique_count),
            'prompt_length': float(self.prompt_length),
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
            gt=str(payload.get('gt', '')),
            candidates=tuple(CandidateRecord.from_dict(item) for item in payload['candidates']),
            num_sample_seeds=int(payload['num_sample_seeds']),
            max_freq_share=float(payload['max_freq_share']),
            normalized_entropy=float(payload['normalized_entropy']),
            unique_count=int(payload['unique_count']),
            prompt_length=float(payload.get('prompt_length', 0.0)),
            min_score=float(payload['min_score']),
            mean_score=float(payload['mean_score']),
            max_score=float(payload['max_score']),
            success_score=float(payload['success_score']),
            has_correct=bool(payload['has_correct']),
        )


@dataclass
class _CandidateAggregate:
    scores: list[float]
    correct: bool
    first_rank: int


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
            if line.strip():
                yield json.loads(line)


def _stable_loss_from_reward(reward: float) -> float:
    # Monotone map from RM reward to a bounded loss in [0, 1]:
    # smaller loss <=> larger reward.
    x = float(reward)
    if x >= 0.0:
        z = math.exp(-x)
        return float(z / (1.0 + z))
    z = math.exp(x)
    return float(1.0 / (1.0 + z))


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


def normalize_answer(text: str) -> str:
    norm = str(text).strip().lower()
    norm = norm.replace(',', '')
    norm = norm.replace('$', '')
    norm = ' '.join(norm.split())
    if norm.endswith('.'):
        norm = norm[:-1].strip()
    return norm


def _score_path_list(score_dir: Path, score_glob: str) -> list[Path]:
    paths = sorted(score_dir.glob(score_glob))
    if not paths:
        raise FileNotFoundError(f'No score files matched {score_glob!r} under {score_dir}.')
    return paths


def _cache_key(score_dir: Path, score_glob: str, max_samples: int, collapse_answers: bool) -> str:
    payload = {
        'version': ARTIFACT_VERSION,
        'score_dir': str(score_dir.resolve()),
        'score_glob': score_glob,
        'max_samples': int(max_samples),
        'collapse_answers': bool(collapse_answers),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:12]
    return f'gsm8k_{digest}.jsonl.gz'


def build_records(
    score_dir: Path,
    score_glob: str = 'gsm8k_with_rm_*.jsonl',
    max_samples: int = 20,
    collapse_answers: bool = False,
) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    for path in _score_path_list(score_dir, score_glob):
        for example in _iter_jsonl(path):
            preds = list(example.get('pred') or [])
            rewards = list(example.get('verifier_score') or [])
            correctness = list(example.get('score') or [])
            usable = min(len(preds), len(rewards), len(correctness))
            if max_samples > 0:
                usable = min(usable, int(max_samples))
            if usable <= 0:
                continue

            trimmed_preds = [str(value) for value in preds[:usable]]
            trimmed_scores = [_stable_loss_from_reward(float(value)) for value in rewards[:usable]]
            trimmed_correct = [bool(value) for value in correctness[:usable]]
            normalized_answers = [normalize_answer(value) for value in trimmed_preds]

            if collapse_answers:
                agg: dict[str, _CandidateAggregate] = {}
                raw_text: dict[str, str] = {}
                for idx, (text, norm, score, correct) in enumerate(zip(trimmed_preds, normalized_answers, trimmed_scores, trimmed_correct)):
                    current = agg.get(norm)
                    if current is None:
                        agg[norm] = _CandidateAggregate(scores=[score], correct=correct, first_rank=idx)
                        raw_text[norm] = text
                    else:
                        current.scores.append(score)
                        current.correct = bool(current.correct or correct)
                ordered = sorted(agg.items(), key=lambda item: item[1].first_rank)
                candidates = tuple(
                    CandidateRecord(
                        text=raw_text[norm],
                        score=float(np.mean(state.scores)),
                        correct=bool(state.correct),
                        rank=int(state.first_rank),
                    )
                    for norm, state in ordered
                )
                freq_counter = Counter(normalized_answers)
            else:
                candidates = tuple(
                    CandidateRecord(
                        text=text,
                        score=score,
                        correct=correct,
                        rank=idx,
                    )
                    for idx, (text, score, correct) in enumerate(zip(trimmed_preds, trimmed_scores, trimmed_correct))
                )
                freq_counter = Counter(normalized_answers)

            freq_arr = np.asarray(list(freq_counter.values()), dtype=float)
            sample_score_arr = np.asarray(trimmed_scores, dtype=float)
            success_scores = [candidate.score for candidate in candidates if candidate.correct]
            max_freq_share = float(np.max(freq_arr) / np.sum(freq_arr)) if freq_arr.size else 0.0
            prompt = str(example.get('question', ''))
            records.append(
                DocumentRecord(
                    doc_id=str(example.get('idx')),
                    question_id=str(example.get('idx')),
                    question=prompt,
                    gt=str(example.get('gt', '')),
                    candidates=candidates,
                    num_sample_seeds=int(usable),
                    max_freq_share=max_freq_share,
                    normalized_entropy=_normalized_entropy(freq_arr),
                    unique_count=int(len(freq_counter)),
                    prompt_length=float(len(prompt.split())),
                    min_score=float(np.min(sample_score_arr)),
                    mean_score=float(np.mean(sample_score_arr)),
                    max_score=float(np.max(sample_score_arr)),
                    success_score=float(min(success_scores)) if success_scores else 1.0,
                    has_correct=bool(success_scores),
                )
            )
    records.sort(key=lambda record: int(record.doc_id))
    return records


def load_or_build_records(
    score_dir: str | Path,
    score_glob: str = 'gsm8k_with_rm_*.jsonl',
    max_samples: int = 20,
    collapse_answers: bool = False,
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
) -> tuple[list[DocumentRecord], Path]:
    score_path = Path(score_dir)
    cache_root = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parents[1] / 'cache'
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / _cache_key(
        score_path,
        score_glob=score_glob,
        max_samples=max_samples,
        collapse_answers=collapse_answers,
    )

    if cache_path.exists() and not force_rebuild:
        records: list[DocumentRecord] = []
        with gzip.open(cache_path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    records.append(DocumentRecord.from_dict(json.loads(line)))
        return records, cache_path

    records = build_records(
        score_path,
        score_glob=score_glob,
        max_samples=max_samples,
        collapse_answers=collapse_answers,
    )
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
