from __future__ import annotations

from dataclasses import dataclass, field
from math import lgamma, log, sqrt
from pathlib import Path
from typing import Any
import json
import sys

import numpy as np

from .data import SyntheticDataset
from .features import FeatureMap

condconf_root = Path(__file__).resolve().parents[2] / 'conditional-conformal'
if str(condconf_root) not in sys.path:
    sys.path.insert(0, str(condconf_root))

from conditionalconformal import CondConf


@dataclass
class PredictionResult:
    name: str
    thresholds: np.ndarray
    accept_mask: np.ndarray
    set_size: np.ndarray
    T: np.ndarray
    S: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, out_dir: str | Path) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        arrays = {
            'thresholds': np.asarray(self.thresholds),
            'accept_mask': np.asarray(self.accept_mask),
            'set_size': np.asarray(self.set_size),
            'T': np.asarray(self.T),
            'S': np.asarray(self.S),
        }
        arrays.update({k: np.asarray(v) for k, v in self.extras.items()})
        np.savez_compressed(out_path / 'preds.npz', **arrays)
        with open(out_path / 'meta.json', 'w', encoding='utf-8') as handle:
            json.dump(self.metadata, handle, indent=2, default=_json_default)


@dataclass
class PACResult:
    alpha_eff: float
    epsilon: float
    cert_report: dict[str, Any]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    n = scores.size
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    k = min(max(k, 1), n)
    return float(scores[k - 1])


def _identity_score_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    del x
    return np.asarray(y, dtype=float).reshape(-1)


def _scalar_threshold_output(threshold: np.ndarray | float, x: np.ndarray) -> np.ndarray:
    del x
    return np.asarray(threshold, dtype=float).reshape(-1)


def _build_condconf(calib: SyntheticDataset, feature_map: FeatureMap, seed: int) -> CondConf:
    fitted_map = FeatureMap(mode=feature_map.mode, bins=feature_map.bins)
    fitted_map.fit(calib.T)
    gcc = CondConf(score_fn=_identity_score_fn, Phi_fn=fitted_map, seed=seed)
    gcc.setup_problem(calib.T.reshape(-1, 1), calib.S)
    return gcc


def _predict_thresholds_exact(
    gcc: CondConf,
    alpha: float,
    dataset: SyntheticDataset,
    exact: bool = True,
) -> np.ndarray:
    quantile = 1.0 - float(alpha)
    x_test = dataset.T.reshape(-1, 1)
    phi_test = np.asarray(gcc._evaluate_phi(x_test), dtype=float)
    _, unique_idx, inverse = np.unique(phi_test, axis=0, return_index=True, return_inverse=True)

    unique_thresholds = np.empty(len(unique_idx), dtype=float)
    for j, idx in enumerate(unique_idx):
        t = float(dataset.T[int(idx)])
        threshold = gcc.predict(
            quantile,
            np.asarray([[t]], dtype=float),
            score_inv_fn=_scalar_threshold_output,
            S_min=0.0,
            S_max=1.0,
            exact=exact,
        )
        unique_thresholds[j] = float(np.asarray(threshold).reshape(-1)[0])
    thresholds = unique_thresholds[inverse]
    return np.clip(thresholds, 0.0, 1.0)


def _prediction_from_thresholds(
    name: str,
    dataset: SyntheticDataset,
    thresholds: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> PredictionResult:
    thresholds = np.asarray(thresholds, dtype=float).reshape(-1)
    accept_mask = dataset.V <= thresholds[:, None]
    return PredictionResult(
        name=name,
        thresholds=thresholds.astype(np.float32),
        accept_mask=accept_mask,
        set_size=accept_mask.sum(axis=1).astype(np.int32),
        T=dataset.T.astype(np.float32),
        S=dataset.S.astype(np.float32),
        metadata=metadata or {},
    )


def _group_bin_indices(feature_map: FeatureMap, values: np.ndarray) -> np.ndarray:
    if feature_map.mode != 'group' or feature_map.edges is None:
        raise ValueError('Group bin indices require a fitted group feature map.')
    values = np.asarray(values, dtype=float).reshape(-1)
    edges = np.asarray(feature_map.edges, dtype=float)
    return np.digitize(values, edges[1:-1], right=True).astype(int)


def _group_exact_thresholds(
    calib: SyntheticDataset,
    test: SyntheticDataset,
    alpha: float,
    feature_map: FeatureMap,
) -> tuple[np.ndarray, dict[str, Any]]:
    fitted_map = FeatureMap(mode='group', bins=feature_map.bins)
    fitted_map.fit(calib.T)
    calib_bins = _group_bin_indices(fitted_map, calib.T)
    test_bins = _group_bin_indices(fitted_map, test.T)
    thresholds_by_bin = np.empty(int(fitted_map.bins), dtype=float)
    for b in range(int(fitted_map.bins)):
        idx = calib_bins == b
        if not np.any(idx):
            raise RuntimeError(f'Calibration bin {b} is empty in grouped exact CFC.')
        thresholds_by_bin[b] = split_conformal_quantile(calib.S[idx], alpha)
    thresholds = thresholds_by_bin[test_bins]
    metadata = {
        'feature_map': fitted_map.to_dict(),
        'bin_thresholds': thresholds_by_bin.tolist(),
    }
    return np.clip(thresholds, 0.0, 1.0), metadata


def run_icp(calib: SyntheticDataset, test: SyntheticDataset, alpha: float) -> PredictionResult:
    threshold = split_conformal_quantile(calib.S, alpha)
    thresholds = np.full(test.n, threshold, dtype=float)
    return _prediction_from_thresholds(
        name='ICP',
        dataset=test,
        thresholds=thresholds,
        metadata={'alpha': float(alpha), 'threshold': float(threshold), 'mode': 'global_cp'},
    )


def select_topk(calib: SyntheticDataset, alpha: float, candidates: list[int]) -> int:
    best_k = None
    best_gap = None
    target = 1.0 - float(alpha)
    order = np.argsort(calib.V, axis=1)
    rows = np.arange(calib.n)[:, None]
    for k in sorted(set(int(value) for value in candidates)):
        frontier = calib.V[rows, order[:, :k]].max(axis=1)
        coverage = float(np.mean(calib.S <= frontier))
        gap = abs(coverage - target)
        if best_gap is None or gap < best_gap - 1e-12 or (abs(gap - best_gap) <= 1e-12 and k < best_k):
            best_k = k
            best_gap = gap
    if best_k is None:
        raise ValueError('TopK candidate list is empty.')
    return int(best_k)


def run_topk(
    calib: SyntheticDataset,
    test: SyntheticDataset,
    alpha: float,
    candidates: list[int],
    k: int | None = None,
) -> PredictionResult:
    chosen_k = select_topk(calib, alpha, candidates) if k is None else int(k)
    order = np.argsort(test.V, axis=1)
    rows = np.arange(test.n)[:, None]
    selected = order[:, :chosen_k]
    accept_mask = np.zeros_like(test.V, dtype=bool)
    accept_mask[rows, selected] = True
    accept_v_max = test.V[rows, selected].max(axis=1)
    return PredictionResult(
        name='TOPK',
        thresholds=np.full(test.n, np.nan, dtype=np.float32),
        accept_mask=accept_mask,
        set_size=accept_mask.sum(axis=1).astype(np.int32),
        T=test.T.astype(np.float32),
        S=test.S.astype(np.float32),
        extras={'accept_v_max': accept_v_max.astype(np.float32)},
        metadata={'alpha': float(alpha), 'selected_k': int(chosen_k), 'candidate_grid': list(candidates)},
    )


def run_learnt_cp(
    calib: SyntheticDataset,
    test: SyntheticDataset,
    alpha: float,
    feature_map: FeatureMap,
    seed: int = 0,
) -> PredictionResult:
    gcc = _build_condconf(calib, feature_map=feature_map, seed=seed)
    thresholds = gcc.predict_naive(
        1.0 - float(alpha),
        test.T.reshape(-1, 1),
        score_inv_fn=_scalar_threshold_output,
    )
    return _prediction_from_thresholds(
        name='LEARNT_CP',
        dataset=test,
        thresholds=np.clip(np.asarray(thresholds, dtype=float).reshape(-1), 0.0, 1.0),
        metadata={
            'alpha': float(alpha),
            'feature_map': gcc.Phi_fn.to_dict() if hasattr(gcc.Phi_fn, 'to_dict') else None,
            'mode': 'naive_threshold_regression',
        },
    )


def run_cfc(
    calib: SyntheticDataset,
    test: SyntheticDataset,
    alpha: float,
    feature_map: FeatureMap,
    seed: int = 0,
    exact: bool = True,
) -> PredictionResult:
    if feature_map.mode == 'group':
        thresholds, extra_meta = _group_exact_thresholds(calib=calib, test=test, alpha=alpha, feature_map=feature_map)
        metadata = {
            'alpha': float(alpha),
            'feature_map': extra_meta['feature_map'],
            'exact': True,
            'solver': 'group_split_conformal',
            'bin_thresholds': extra_meta['bin_thresholds'],
        }
        return _prediction_from_thresholds(name='CFC', dataset=test, thresholds=thresholds, metadata=metadata)

    gcc = _build_condconf(calib, feature_map=feature_map, seed=seed)
    thresholds = _predict_thresholds_exact(gcc, alpha=alpha, dataset=test, exact=exact)
    return _prediction_from_thresholds(
        name='CFC',
        dataset=test,
        thresholds=thresholds,
        metadata={
            'alpha': float(alpha),
            'feature_map': gcc.Phi_fn.to_dict() if hasattr(gcc.Phi_fn, 'to_dict') else None,
            'exact': bool(exact),
        },
    )


def _log_binom_pmf(n: int, k: int, p: float) -> float:
    if p <= 0.0:
        return -np.inf if k > 0 else 0.0
    if p >= 1.0:
        return -np.inf if k < n else 0.0
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log(1.0 - p)


def _binom_tail_sf(successes: int, n: int, p: float) -> float:
    if successes <= 0:
        return 1.0
    if successes > n:
        return 0.0
    logs = np.asarray([_log_binom_pmf(n, k, p) for k in range(successes, n + 1)], dtype=float)
    max_log = float(np.max(logs))
    return float(np.exp(max_log) * np.sum(np.exp(logs - max_log)))


def clopper_pearson_lower_bound(successes: int, n: int, delta: float, tol: float = 1e-6) -> float:
    if successes <= 0:
        return 0.0
    if successes >= n:
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lower + upper)
        tail = _binom_tail_sf(successes, n, mid)
        if tail > delta:
            upper = mid
        else:
            lower = mid
        if upper - lower <= tol:
            break
    return float(lower)


def compute_pac_adjustment_stability(alpha: float, delta: float, n_cal: int, scale: float = 1.0) -> PACResult:
    if not (0.0 < delta < 1.0):
        raise ValueError(f'delta must lie in (0, 1); got {delta}.')
    epsilon = float(scale) * sqrt(log(1.0 / float(delta)) / max(1, int(n_cal)))
    alpha_eff = max(0.0, float(alpha) - epsilon)
    return PACResult(
        alpha_eff=alpha_eff,
        epsilon=epsilon,
        cert_report={'mode': 'stability', 'scale': float(scale), 'n_cal': int(n_cal), 'delta': float(delta)},
    )


def choose_alpha_eff_holdout(
    calib: SyntheticDataset,
    alpha: float,
    delta: float,
    feature_map: FeatureMap,
    seed: int = 0,
    cert_size: int = 2000,
    grid_step: float = 0.01,
    max_adjust: float = 0.2,
    exact: bool = True,
) -> tuple[float, dict[str, Any], SyntheticDataset]:
    n = calib.n
    cert_size = min(max(int(cert_size), 1), n - 1)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    cert_idx = permutation[:cert_size]
    train_idx = permutation[cert_size:]

    calib_train = SyntheticDataset(
        X=calib.X[train_idx],
        T=calib.T[train_idx],
        V=calib.V[train_idx],
        A=calib.A[train_idx],
        S=calib.S[train_idx],
    )
    calib_cert = SyntheticDataset(
        X=calib.X[cert_idx],
        T=calib.T[cert_idx],
        V=calib.V[cert_idx],
        A=calib.A[cert_idx],
        S=calib.S[cert_idx],
    )

    lower_alpha = max(0.0, float(alpha) - float(max_adjust))
    grid = np.arange(float(alpha), lower_alpha - 1e-12, -float(grid_step))
    report: dict[str, Any] = {'mode': 'holdout', 'grid': grid.tolist(), 'cert_size': cert_size, 'delta': float(delta)}
    best_alpha = None
    best_lower = None

    for alpha_eff in grid:
        result = run_cfc(calib_train, calib_cert, alpha=alpha_eff, feature_map=feature_map, seed=seed, exact=exact)
        successes = int(np.sum(calib_cert.S <= result.thresholds))
        lower = clopper_pearson_lower_bound(successes, calib_cert.n, delta)
        report[f'{alpha_eff:.6f}'] = {'successes': successes, 'n': calib_cert.n, 'cp_lower': lower}
        if lower >= 1.0 - float(alpha):
            best_alpha = float(alpha_eff)
            best_lower = float(lower)
    if best_alpha is None:
        best_alpha = float(grid.min())
        best_lower = float(report[f'{best_alpha:.6f}']['cp_lower'])
    report['selected_alpha_eff'] = best_alpha
    report['selected_cp_lower'] = best_lower
    return best_alpha, report, calib_train


def run_cfc_pac(
    calib: SyntheticDataset,
    test: SyntheticDataset,
    alpha: float,
    delta: float,
    feature_map: FeatureMap,
    seed: int = 0,
    exact: bool = True,
    mode: str = 'stability',
    cert_size: int = 2000,
    grid_step: float = 0.01,
    max_adjust: float = 0.2,
    stability_scale: float = 1.0,
    fixed_alpha_eff: float | None = None,
) -> PredictionResult:
    if fixed_alpha_eff is not None:
        alpha_eff = float(fixed_alpha_eff)
        if not (0.0 <= alpha_eff <= 1.0):
            raise ValueError(f'fixed_alpha_eff must lie in [0, 1]; got {fixed_alpha_eff}.')
        pac = PACResult(
            alpha_eff=alpha_eff,
            epsilon=float(alpha) - alpha_eff,
            cert_report={'mode': 'fixed', 'fixed_alpha_eff': alpha_eff},
        )
        fit_calib = calib
    elif mode == 'stability':
        pac = compute_pac_adjustment_stability(alpha=alpha, delta=delta, n_cal=calib.n, scale=stability_scale)
        fit_calib = calib
    elif mode == 'holdout':
        alpha_eff, cert_report, fit_calib = choose_alpha_eff_holdout(
            calib=calib,
            alpha=alpha,
            delta=delta,
            feature_map=feature_map,
            seed=seed,
            cert_size=cert_size,
            grid_step=grid_step,
            max_adjust=max_adjust,
            exact=exact,
        )
        pac = PACResult(alpha_eff=alpha_eff, epsilon=float(alpha - alpha_eff), cert_report=cert_report)
    else:
        raise ValueError(f'Unsupported PAC mode: {mode}')

    result = run_cfc(
        calib=fit_calib,
        test=test,
        alpha=pac.alpha_eff,
        feature_map=feature_map,
        seed=seed,
        exact=exact,
    )
    result.name = 'CFC_PAC'
    result.metadata.update(
        {
            'target_alpha': float(alpha),
            'alpha_eff': float(pac.alpha_eff),
            'epsilon': float(pac.epsilon),
            'delta': float(delta),
            'pac_mode': mode,
            'cert_report': pac.cert_report,
        }
    )
    return result
