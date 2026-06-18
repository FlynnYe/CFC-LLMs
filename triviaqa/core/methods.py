from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .data import DocumentRecord
from .features import build_feature_map, difficulty_array

condconf_root = Path(__file__).resolve().parents[2] / 'conditional-conformal'
if str(condconf_root) not in sys.path:
    sys.path.insert(0, str(condconf_root))

from conditionalconformal import CondConf


METHOD_NAMES = {'TOPK', 'ICP', 'LEARNT_CP', 'CFC', 'CFC_FULL', 'CFC_PAC', 'CFC_PAC_FULL'}


@dataclass
class MethodOutput:
    name: str
    selected_masks: list[np.ndarray]
    thresholds: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'thresholds': None if self.thresholds is None else [float(value) for value in self.thresholds],
            'metadata': self.metadata,
        }


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    arr = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    n = arr.size
    if n == 0:
        return 1.0
    rank = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), n)
    return float(arr[rank - 1])


def _identity_score_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    del x
    return np.asarray(y, dtype=float).reshape(-1)


def _scalar_threshold_output(threshold: np.ndarray | float, x: np.ndarray) -> np.ndarray:
    del x
    return np.asarray(threshold, dtype=float).reshape(-1)


def _record_scores(record: DocumentRecord) -> np.ndarray:
    return np.asarray([candidate.score for candidate in record.candidates], dtype=float)


def _topk_mask(record: DocumentRecord, k: int) -> np.ndarray:
    scores = _record_scores(record)
    order = np.argsort(scores)
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[order[: min(int(k), scores.shape[0])]] = True
    return mask


def _threshold_mask(record: DocumentRecord, threshold: float) -> np.ndarray:
    return _record_scores(record) <= float(threshold)


def _truncate_after_best_selected(record: DocumentRecord, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(selected):
        return selected.copy()
    scores = _record_scores(record)
    selected_indices = np.flatnonzero(selected)
    best_index = selected_indices[int(np.argmin(scores[selected_indices]))]
    keep_prefix = np.arange(selected.shape[0], dtype=int) <= int(best_index)
    return selected & keep_prefix


def _truncate_masks(records: list[DocumentRecord], masks: list[np.ndarray]) -> list[np.ndarray]:
    return [_truncate_after_best_selected(record, mask) for record, mask in zip(records, masks)]


def _coverage_with_topk(records: list[DocumentRecord], k: int) -> float:
    if not records:
        return float('nan')
    covered = []
    for record in records:
        mask = _topk_mask(record, k)
        covered.append(any(candidate.correct for candidate, selected in zip(record.candidates, mask) if selected))
    return float(np.mean(covered))


def select_topk_k(
    calib_records: list[DocumentRecord],
    alpha: float,
    max_k: int = 20,
    selection_mode: str = 'closest',
) -> tuple[int, dict[str, Any]]:
    target = 1.0 - float(alpha)
    candidates = list(range(1, max(1, max_k) + 1))
    coverages = {k: _coverage_with_topk(calib_records, k) for k in candidates}

    if selection_mode == 'at_least':
        feasible = [k for k in candidates if coverages[k] >= target]
        chosen = feasible[0] if feasible else candidates[-1]
    else:
        chosen = min(candidates, key=lambda k: (abs(coverages[k] - target), k))

    return chosen, {
        'target_coverage': float(target),
        'selection_mode': selection_mode,
        'candidate_coverages': {str(k): float(v) for k, v in coverages.items()},
    }


def run_topk(
    calib_records: list[DocumentRecord],
    test_records: list[DocumentRecord],
    alpha: float,
    max_k: int = 20,
    selection_mode: str = 'closest',
) -> MethodOutput:
    chosen_k, meta = select_topk_k(calib_records, alpha=alpha, max_k=max_k, selection_mode=selection_mode)
    masks = [_topk_mask(record, chosen_k) for record in test_records]
    return MethodOutput(
        name='TOPK',
        selected_masks=masks,
        thresholds=None,
        metadata={'alpha': float(alpha), 'selected_k': int(chosen_k), **meta},
    )


def run_icp(calib_records: list[DocumentRecord], test_records: list[DocumentRecord], alpha: float) -> MethodOutput:
    success_scores = np.asarray([record.success_score for record in calib_records], dtype=float)
    threshold = split_conformal_quantile(success_scores, alpha=alpha)
    masks = [_threshold_mask(record, threshold) for record in test_records]
    thresholds = np.full(len(test_records), threshold, dtype=float)
    return MethodOutput(
        name='ICP',
        selected_masks=masks,
        thresholds=thresholds,
        metadata={'alpha': float(alpha), 'threshold': float(threshold), 'mode': 'global_split_cp'},
    )


def _build_condconf(
    calib_records: list[DocumentRecord],
    proxy: str,
    seed: int,
    feature_mode: str,
    feature_bins: int,
) -> tuple[CondConf, np.ndarray, np.ndarray, dict[str, Any]]:
    difficulty = difficulty_array(calib_records, proxy=proxy)
    success_scores = np.asarray([record.success_score for record in calib_records], dtype=float)
    phi = build_feature_map(difficulty, proxy=proxy, mode=feature_mode, bins=feature_bins)
    condconf = CondConf(score_fn=_identity_score_fn, Phi_fn=phi, seed=seed)
    condconf.setup_problem(difficulty.reshape(-1, 1), success_scores)
    phi_meta = phi.to_dict() if hasattr(phi, 'to_dict') else {'mode': feature_mode}
    return condconf, difficulty, success_scores, phi_meta


def _predict_thresholds(condconf: CondConf, difficulty: np.ndarray, alpha: float, exact: bool) -> np.ndarray:
    unique_vals, inverse = np.unique(np.asarray(difficulty, dtype=float), return_inverse=True)
    thresholds = np.empty(unique_vals.shape[0], dtype=float)
    for idx, value in enumerate(unique_vals):
        threshold = condconf.predict(
            1.0 - float(alpha),
            np.asarray([[float(value)]], dtype=float),
            score_inv_fn=_scalar_threshold_output,
            S_min=0.0,
            S_max=1.0,
            exact=exact,
        )
        thresholds[idx] = float(np.asarray(threshold).reshape(-1)[0])
    return np.clip(thresholds[inverse], 0.0, 1.0)


def _predict_thresholds_naive(condconf: CondConf, difficulty: np.ndarray, alpha: float) -> np.ndarray:
    thresholds = condconf.predict_naive(
        1.0 - float(alpha),
        np.asarray(difficulty, dtype=float).reshape(-1, 1),
        score_inv_fn=_scalar_threshold_output,
    )
    return np.clip(np.asarray(thresholds, dtype=float).reshape(-1), 0.0, 1.0)


def run_learnt_cp(
    calib_records: list[DocumentRecord],
    test_records: list[DocumentRecord],
    alpha: float,
    proxy: str = 'disagreement',
    seed: int = 0,
    feature_mode: str = 'linear',
    feature_bins: int = 5,
) -> MethodOutput:
    condconf, _, _, phi_meta = _build_condconf(
        calib_records,
        proxy=proxy,
        seed=seed,
        feature_mode=feature_mode,
        feature_bins=feature_bins,
    )
    test_difficulty = difficulty_array(test_records, proxy=proxy)
    thresholds = _predict_thresholds_naive(condconf, test_difficulty, alpha=alpha)
    masks = [_threshold_mask(record, tau) for record, tau in zip(test_records, thresholds)]
    return MethodOutput(
        name='LEARNT_CP',
        selected_masks=masks,
        thresholds=thresholds,
        metadata={
            'alpha': float(alpha),
            'proxy': proxy,
            'feature_mode': feature_mode,
            'feature_bins': int(feature_bins),
            'phi': phi_meta,
            'mode': 'naive_threshold_regression',
        },
    )


def run_cfc_full(
    calib_records: list[DocumentRecord],
    test_records: list[DocumentRecord],
    alpha: float,
    proxy: str = 'disagreement',
    seed: int = 0,
    exact: bool = True,
    feature_mode: str = 'linear',
    feature_bins: int = 5,
) -> MethodOutput:
    condconf, _, _, phi_meta = _build_condconf(
        calib_records,
        proxy=proxy,
        seed=seed,
        feature_mode=feature_mode,
        feature_bins=feature_bins,
    )
    test_difficulty = difficulty_array(test_records, proxy=proxy)
    thresholds = _predict_thresholds(condconf, test_difficulty, alpha=alpha, exact=exact)
    masks = [_threshold_mask(record, tau) for record, tau in zip(test_records, thresholds)]
    return MethodOutput(
        name='CFC_FULL',
        selected_masks=masks,
        thresholds=thresholds,
        metadata={
            'alpha': float(alpha),
            'proxy': proxy,
            'feature_mode': feature_mode,
            'feature_bins': int(feature_bins),
            'phi': phi_meta,
            'exact': bool(exact),
            'postprocess': 'none',
        },
    )


def run_cfc(
    calib_records: list[DocumentRecord],
    test_records: list[DocumentRecord],
    alpha: float,
    proxy: str = 'disagreement',
    seed: int = 0,
    exact: bool = True,
    feature_mode: str = 'linear',
    feature_bins: int = 5,
) -> MethodOutput:
    result = run_cfc_full(
        calib_records=calib_records,
        test_records=test_records,
        alpha=alpha,
        proxy=proxy,
        seed=seed,
        exact=exact,
        feature_mode=feature_mode,
        feature_bins=feature_bins,
    )
    result.name = 'CFC'
    result.selected_masks = _truncate_masks(test_records, result.selected_masks)
    result.metadata.update(
        {
            'postprocess': 'truncate_after_best_selected',
            'base_method': 'CFC_FULL',
        }
    )
    return result


def run_cfc_pac_full(
    calib_records: list[DocumentRecord],
    test_records: list[DocumentRecord],
    alpha: float,
    proxy: str = 'disagreement',
    seed: int = 0,
    exact: bool = True,
    delta: float = 0.90,
    stability_scale: float = 1.0,
    feature_mode: str = 'linear',
    feature_bins: int = 5,
) -> MethodOutput:
    if not (0.0 < delta < 1.0):
        raise ValueError('delta must lie in (0, 1).')
    epsilon = float(stability_scale) * sqrt(log(1.0 / float(delta)) / max(len(calib_records), 1))
    alpha_eff = max(float(alpha) - epsilon, 1e-6)
    result = run_cfc_full(
        calib_records=calib_records,
        test_records=test_records,
        alpha=alpha_eff,
        proxy=proxy,
        seed=seed,
        exact=exact,
        feature_mode=feature_mode,
        feature_bins=feature_bins,
    )
    result.name = 'CFC_PAC_FULL'
    result.metadata.update(
        {
            'alpha_nominal': float(alpha),
            'alpha_effective': float(alpha_eff),
            'pac_delta': float(delta),
            'pac_epsilon': float(epsilon),
            'pac_mode': 'stability',
            'stability_scale': float(stability_scale),
        }
    )
    return result


def run_cfc_pac(
    calib_records: list[DocumentRecord],
    test_records: list[DocumentRecord],
    alpha: float,
    proxy: str = 'disagreement',
    seed: int = 0,
    exact: bool = True,
    delta: float = 0.90,
    stability_scale: float = 1.0,
    feature_mode: str = 'linear',
    feature_bins: int = 5,
) -> MethodOutput:
    result = run_cfc_pac_full(
        calib_records=calib_records,
        test_records=test_records,
        alpha=alpha,
        proxy=proxy,
        seed=seed,
        exact=exact,
        delta=delta,
        stability_scale=stability_scale,
        feature_mode=feature_mode,
        feature_bins=feature_bins,
    )
    result.name = 'CFC_PAC'
    result.selected_masks = _truncate_masks(test_records, result.selected_masks)
    result.metadata.update(
        {
            'postprocess': 'truncate_after_best_selected',
            'base_method': 'CFC_PAC_FULL',
        }
    )
    return result


def run_method(
    method: str,
    calib_records: list[DocumentRecord],
    test_records: list[DocumentRecord],
    alpha: float,
    proxy: str = 'disagreement',
    seed: int = 0,
    exact: bool = True,
    topk_max_k: int = 20,
    topk_selection_mode: str = 'closest',
    pac_delta: float = 0.90,
    pac_stability_scale: float = 1.0,
    feature_mode: str = 'linear',
    feature_bins: int = 5,
) -> MethodOutput:
    method = method.upper()
    if method == 'TOPK':
        return run_topk(calib_records, test_records, alpha=alpha, max_k=topk_max_k, selection_mode=topk_selection_mode)
    if method == 'ICP':
        return run_icp(calib_records, test_records, alpha=alpha)
    if method == 'LEARNT_CP':
        return run_learnt_cp(
            calib_records,
            test_records,
            alpha=alpha,
            proxy=proxy,
            seed=seed,
            feature_mode=feature_mode,
            feature_bins=feature_bins,
        )
    if method == 'CFC':
        return run_cfc(
            calib_records,
            test_records,
            alpha=alpha,
            proxy=proxy,
            seed=seed,
            exact=exact,
            feature_mode=feature_mode,
            feature_bins=feature_bins,
        )
    if method == 'CFC_FULL':
        return run_cfc_full(
            calib_records,
            test_records,
            alpha=alpha,
            proxy=proxy,
            seed=seed,
            exact=exact,
            feature_mode=feature_mode,
            feature_bins=feature_bins,
        )
    if method == 'CFC_PAC':
        return run_cfc_pac(
            calib_records,
            test_records,
            alpha=alpha,
            proxy=proxy,
            seed=seed,
            exact=exact,
            delta=pac_delta,
            stability_scale=pac_stability_scale,
            feature_mode=feature_mode,
            feature_bins=feature_bins,
        )
    if method == 'CFC_PAC_FULL':
        return run_cfc_pac_full(
            calib_records,
            test_records,
            alpha=alpha,
            proxy=proxy,
            seed=seed,
            exact=exact,
            delta=pac_delta,
            stability_scale=pac_stability_scale,
            feature_mode=feature_mode,
            feature_bins=feature_bins,
        )
    raise ValueError(f'Unknown method: {method}')
