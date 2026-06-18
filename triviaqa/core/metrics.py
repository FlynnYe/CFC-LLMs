from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DocumentRecord


@dataclass
class EvaluationMetrics:
    ecr: float
    apss: float
    gsc: float
    selection_rate: float
    group_coverages: np.ndarray
    bin_edges: np.ndarray

    def to_dict(self) -> dict:
        return {
            'ecr': float(self.ecr),
            'apss': float(self.apss),
            'gsc': float(self.gsc),
            'selection_rate': float(self.selection_rate),
            'group_coverages': [float(value) for value in self.group_coverages],
            'bin_edges': [float(value) for value in self.bin_edges],
        }


def _equal_frequency_edges(values: np.ndarray, bins: int) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=float)
    return np.quantile(values, np.linspace(0.0, 1.0, int(bins) + 1))


def evaluate_predictions(
    records: list[DocumentRecord],
    selected_masks: list[np.ndarray],
    difficulty: np.ndarray,
    group_bins: int = 5,
) -> EvaluationMetrics:
    if len(records) != len(selected_masks):
        raise ValueError('records and selected_masks must have the same length.')
    if difficulty.shape[0] != len(records):
        raise ValueError('difficulty must have one value per record.')

    covered = np.zeros(len(records), dtype=bool)
    set_sizes = np.zeros(len(records), dtype=float)
    totals = np.zeros(len(records), dtype=float)
    for idx, (record, mask) in enumerate(zip(records, selected_masks)):
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape[0] != len(record.candidates):
            raise ValueError('Each selection mask must align with the record candidate count.')
        set_sizes[idx] = float(np.sum(mask))
        totals[idx] = float(len(record.candidates))
        if np.any(mask):
            covered[idx] = any(candidate.correct for candidate, selected in zip(record.candidates, mask) if selected)

    edges = _equal_frequency_edges(np.asarray(difficulty, dtype=float), bins=group_bins)
    group_coverages = np.empty(group_bins, dtype=float)
    for group in range(group_bins):
        lo = edges[group]
        hi = edges[group + 1] + 1e-12
        group_mask = (difficulty >= lo) & (difficulty <= hi)
        group_coverages[group] = float(np.mean(covered[group_mask])) if np.any(group_mask) else np.nan

    return EvaluationMetrics(
        ecr=float(np.mean(covered)) if covered.size else float('nan'),
        apss=float(np.mean(set_sizes)) if set_sizes.size else float('nan'),
        gsc=float(np.nanmin(group_coverages)) if group_coverages.size else float('nan'),
        selection_rate=float(np.sum(set_sizes) / np.sum(totals)) if np.sum(totals) > 0 else float('nan'),
        group_coverages=group_coverages,
        bin_edges=edges,
    )
