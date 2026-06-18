from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import SyntheticDataset
from .methods import PredictionResult


@dataclass
class MethodMetrics:
    method: str
    ecr: float
    apss: float
    gsc: float
    group_coverages: np.ndarray
    bin_edges: np.ndarray
    true_ecr: float
    true_gsc: float
    true_group_coverages: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            'method': self.method,
            'ecr': float(self.ecr),
            'apss': float(self.apss),
            'gsc': float(self.gsc),
            'group_coverages': self.group_coverages.tolist(),
            'bin_edges': self.bin_edges.tolist(),
            'true_ecr': float(self.true_ecr),
            'true_gsc': float(self.true_gsc),
            'true_group_coverages': self.true_group_coverages.tolist(),
        }


def _proxy_success(result: PredictionResult, dataset: SyntheticDataset) -> np.ndarray:
    thresholds = np.asarray(result.thresholds, dtype=float)
    if np.isnan(thresholds).any():
        if 'accept_v_max' not in result.extras:
            raise ValueError(f'Method {result.name} has no threshold or frontier for proxy coverage.')
        frontier = np.asarray(result.extras['accept_v_max'], dtype=float).reshape(-1)
        return dataset.S <= frontier
    return dataset.S <= thresholds.reshape(-1)


def _true_success(result: PredictionResult, dataset: SyntheticDataset) -> np.ndarray:
    return np.any(np.asarray(result.accept_mask, dtype=bool) & dataset.A.astype(bool), axis=1)


def _compute_group_coverages(success: np.ndarray, t: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(t, np.linspace(0.0, 1.0, int(bins) + 1))
    coverages = np.empty(int(bins), dtype=float)
    for b in range(int(bins)):
        lo = edges[b]
        hi = edges[b + 1] + 1e-12
        idx = (t >= lo) & (t <= hi)
        coverages[b] = float(np.mean(success[idx])) if np.any(idx) else np.nan
    return coverages, edges


def evaluate_prediction_result(
    result: PredictionResult,
    dataset: SyntheticDataset,
    bins: int = 10,
) -> MethodMetrics:
    proxy_success = _proxy_success(result, dataset)
    true_success = _true_success(result, dataset)
    proxy_group, edges = _compute_group_coverages(proxy_success, dataset.T, bins=bins)
    true_group, _ = _compute_group_coverages(true_success, dataset.T, bins=bins)
    return MethodMetrics(
        method=result.name,
        ecr=float(np.mean(proxy_success)),
        apss=float(np.mean(result.set_size)),
        gsc=float(np.nanmin(proxy_group)),
        group_coverages=proxy_group,
        bin_edges=edges,
        true_ecr=float(np.mean(true_success)),
        true_gsc=float(np.nanmin(true_group)),
        true_group_coverages=true_group,
    )


def aggregate_method_metrics(metrics: list[MethodMetrics]) -> dict[str, Any]:
    if not metrics:
        raise ValueError('metrics must be non-empty')
    group_coverages = np.stack([metric.group_coverages for metric in metrics], axis=0)
    true_group_coverages = np.stack([metric.true_group_coverages for metric in metrics], axis=0)
    gsc_seed = np.asarray([metric.gsc for metric in metrics], dtype=float)
    true_gsc_seed = np.asarray([metric.true_gsc for metric in metrics], dtype=float)

    return {
        'method': metrics[0].method,
        'ecr_mean': float(np.mean([metric.ecr for metric in metrics])),
        'ecr_std': float(np.std([metric.ecr for metric in metrics])),
        'apss_mean': float(np.mean([metric.apss for metric in metrics])),
        'apss_std': float(np.std([metric.apss for metric in metrics])),
        'gsc_seed_mean': float(np.mean(gsc_seed)),
        'gsc_seed_std': float(np.std(gsc_seed)),
        'gsc_min_mean_bin': float(np.nanmin(np.mean(group_coverages, axis=0))),
        'group_coverages_mean': np.mean(group_coverages, axis=0).tolist(),
        'group_coverages_std': np.std(group_coverages, axis=0).tolist(),
        'true_ecr_mean': float(np.mean([metric.true_ecr for metric in metrics])),
        'true_ecr_std': float(np.std([metric.true_ecr for metric in metrics])),
        'true_gsc_seed_mean': float(np.mean(true_gsc_seed)),
        'true_gsc_seed_std': float(np.std(true_gsc_seed)),
        'true_gsc_min_mean_bin': float(np.nanmin(np.mean(true_group_coverages, axis=0))),
        'true_group_coverages_mean': np.mean(true_group_coverages, axis=0).tolist(),
        'true_group_coverages_std': np.std(true_group_coverages, axis=0).tolist(),
        'bin_edges': metrics[0].bin_edges.tolist(),
        'num_seeds': len(metrics),
    }
