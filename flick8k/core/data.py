from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
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
    image_id: str
    image_path: str
    reference_captions: tuple[str, ...]
    candidates: tuple[CandidateRecord, ...]
    num_candidates: int
    num_references: int
    unique_count: int
    avg_caption_length: float
    caption_length_std: float
    min_score: float
    mean_score: float
    max_score: float
    score_spread: float
    success_score: float
    has_correct: bool

    def to_dict(self) -> dict:
        return {
            'doc_id': self.doc_id,
            'image_id': self.image_id,
            'image_path': self.image_path,
            'reference_captions': list(self.reference_captions),
            'candidates': [candidate.to_dict() for candidate in self.candidates],
            'num_candidates': int(self.num_candidates),
            'num_references': int(self.num_references),
            'unique_count': int(self.unique_count),
            'avg_caption_length': float(self.avg_caption_length),
            'caption_length_std': float(self.caption_length_std),
            'min_score': float(self.min_score),
            'mean_score': float(self.mean_score),
            'max_score': float(self.max_score),
            'score_spread': float(self.score_spread),
            'success_score': float(self.success_score),
            'has_correct': bool(self.has_correct),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> 'DocumentRecord':
        return cls(
            doc_id=str(payload['doc_id']),
            image_id=str(payload.get('image_id', payload['doc_id'])),
            image_path=str(payload.get('image_path', '')),
            reference_captions=tuple(str(item) for item in payload.get('reference_captions', [])),
            candidates=tuple(CandidateRecord.from_dict(item) for item in payload['candidates']),
            num_candidates=int(payload['num_candidates']),
            num_references=int(payload['num_references']),
            unique_count=int(payload['unique_count']),
            avg_caption_length=float(payload['avg_caption_length']),
            caption_length_std=float(payload['caption_length_std']),
            min_score=float(payload['min_score']),
            mean_score=float(payload['mean_score']),
            max_score=float(payload['max_score']),
            score_spread=float(payload['score_spread']),
            success_score=float(payload['success_score']),
            has_correct=bool(payload['has_correct']),
        )


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


def _load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _cache_key(coverage_cache: Path, candidate_cache: Path | None, max_candidates: int) -> str:
    payload = {
        'version': ARTIFACT_VERSION,
        'coverage_cache': str(coverage_cache.resolve()),
        'candidate_cache': None if candidate_cache is None else str(candidate_cache.resolve()),
        'max_candidates': int(max_candidates),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:12]
    return f'flick8k_{digest}.jsonl.gz'


def build_records(
    coverage_cache: str | Path,
    candidate_cache: str | Path | None = None,
    max_candidates: int = 0,
) -> list[DocumentRecord]:
    coverage_path = Path(coverage_cache)
    candidate_path = None if candidate_cache is None else Path(candidate_cache)

    coverage_payload = _load_json(coverage_path)
    coverage_rows = list(coverage_payload['data'])

    candidate_by_image: dict[str, dict] = {}
    if candidate_path is not None:
        candidate_payload = _load_json(candidate_path)
        candidate_by_image = {str(item['image']): item for item in candidate_payload['data']}

    records: list[DocumentRecord] = []
    for item in coverage_rows:
        image_id = str(item['image'])
        candidates_raw = list(item.get('candidates') or [])
        coverage_flags = list(item.get('candidate_coverage') or [])
        usable = min(len(candidates_raw), len(coverage_flags))
        if max_candidates > 0:
            usable = min(usable, int(max_candidates))
        if usable <= 0:
            continue

        meta = candidate_by_image.get(image_id, {})
        image_path = str(meta.get('image_path', ''))
        reference_captions = tuple(str(text) for text in meta.get('reference_captions', []))

        candidates = tuple(
            CandidateRecord(
                text=str(candidates_raw[idx]['caption']),
                score=float(candidates_raw[idx]['score']),
                correct=bool(coverage_flags[idx]),
                rank=idx,
            )
            for idx in range(usable)
        )

        scores = np.asarray([candidate.score for candidate in candidates], dtype=float)
        lengths = np.asarray([len(candidate.text.split()) for candidate in candidates], dtype=float)
        success_scores = [candidate.score for candidate in candidates if candidate.correct]
        score_spread = float(np.max(scores) - np.min(scores)) if scores.size else 0.0
        failure_slack = max(score_spread, 1e-6)
        failure_score = float(np.max(scores) + failure_slack) if scores.size else 1.0

        records.append(
            DocumentRecord(
                doc_id=image_id,
                image_id=image_id,
                image_path=image_path,
                reference_captions=reference_captions,
                candidates=candidates,
                num_candidates=int(len(candidates)),
                num_references=int(len(reference_captions)),
                unique_count=int(len(candidates)),
                avg_caption_length=float(np.mean(lengths)) if lengths.size else 0.0,
                caption_length_std=float(np.std(lengths)) if lengths.size else 0.0,
                min_score=float(np.min(scores)) if scores.size else 0.0,
                mean_score=float(np.mean(scores)) if scores.size else 0.0,
                max_score=float(np.max(scores)) if scores.size else 0.0,
                score_spread=score_spread,
                success_score=float(min(success_scores)) if success_scores else failure_score,
                has_correct=bool(success_scores),
            )
        )

    records.sort(key=lambda record: record.image_id)
    return records


def load_or_build_records(
    coverage_cache: str | Path,
    candidate_cache: str | Path | None = None,
    max_candidates: int = 0,
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
) -> tuple[list[DocumentRecord], Path]:
    coverage_path = Path(coverage_cache)
    candidate_path = None if candidate_cache is None else Path(candidate_cache)
    cache_root = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parents[1] / 'cache'
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / _cache_key(coverage_path, candidate_path, max_candidates=max_candidates)

    if cache_path.exists() and not force_rebuild:
        records: list[DocumentRecord] = []
        with gzip.open(cache_path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    records.append(DocumentRecord.from_dict(json.loads(line)))
        return records, cache_path

    records = build_records(
        coverage_cache=coverage_path,
        candidate_cache=candidate_path,
        max_candidates=max_candidates,
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
