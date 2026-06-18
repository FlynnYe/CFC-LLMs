from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SyntheticParams:
    eta: int = 2
    gc0: float = 0.2
    gc1: float = 0.3
    gi0: float = 0.6
    gi1: float = 0.3
    p_scale: float = 1.0
    gamma_c_shift: float = 0.0
    gamma_i_shift: float = 0.0
    tail_tau: float | None = None
    tail_p_add: float = 0.0
    tail_gamma_c_delta: float = 0.0
    tail_gamma_i_delta: float = 0.0


@dataclass
class SyntheticDataset:
    X: np.ndarray
    T: np.ndarray
    V: np.ndarray
    A: np.ndarray
    S: np.ndarray

    @property
    def n(self) -> int:
        return int(self.T.shape[0])

    @property
    def M(self) -> int:
        return int(self.V.shape[1])

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            'X': np.asarray(self.X),
            'T': np.asarray(self.T),
            'V': np.asarray(self.V),
            'A': np.asarray(self.A),
            'S': np.asarray(self.S),
        }


@dataclass
class SyntheticSplits:
    calib: SyntheticDataset
    test: SyntheticDataset


def sample_prompt(
    M: int,
    params: SyntheticParams,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    x = float(rng.uniform(0.0, 1.0))
    t = x

    p_t = (1.0 - (t ** params.eta)) * params.p_scale
    if params.tail_tau is not None and t >= params.tail_tau:
        weight = (t - params.tail_tau) / max(1e-12, 1.0 - params.tail_tau)
        p_t = p_t + params.tail_p_add * weight
    p_t = float(np.clip(p_t, 0.0, 1.0))

    scores = np.empty(M, dtype=np.float32)
    labels = np.empty(M, dtype=np.int8)
    for j in range(M):
        is_correct = rng.uniform() < p_t
        gamma = (
            params.gc0 + params.gc1 * t + params.gamma_c_shift
            if is_correct
            else params.gi0 + params.gi1 * t + params.gamma_i_shift
        )
        if params.tail_tau is not None and t >= params.tail_tau:
            weight = (t - params.tail_tau) / max(1e-12, 1.0 - params.tail_tau)
            if is_correct:
                gamma = gamma + params.tail_gamma_c_delta * weight
            else:
                gamma = gamma + params.tail_gamma_i_delta * weight
        gamma = float(np.clip(gamma, 1e-3, 0.99))
        u = float(rng.uniform())
        scores[j] = u ** (1.0 / gamma)
        labels[j] = 1 if is_correct else 0

    correct_scores = scores[labels == 1]
    success_score = float(correct_scores.min()) if correct_scores.size else 1.0
    return {'X': x, 'T': t, 'V': scores, 'A': labels, 'S': success_score}


def _build_split(
    n: int,
    M: int,
    params: SyntheticParams,
    rng: np.random.Generator,
) -> SyntheticDataset:
    X = np.empty(n, dtype=float)
    T = np.empty(n, dtype=float)
    V = np.empty((n, M), dtype=np.float32)
    A = np.empty((n, M), dtype=np.int8)
    S = np.empty(n, dtype=np.float32)
    for i in range(n):
        record = sample_prompt(M=M, params=params, rng=rng)
        X[i] = float(record['X'])
        T[i] = float(record['T'])
        V[i] = np.asarray(record['V'], dtype=np.float32)
        A[i] = np.asarray(record['A'], dtype=np.int8)
        S[i] = float(record['S'])
    return SyntheticDataset(X=X, T=T, V=V, A=A, S=S)


def generate_dataset(
    params: SyntheticParams,
    n_cal: int,
    n_test: int,
    M: int,
    seed: int,
) -> SyntheticSplits:
    rng = np.random.default_rng(seed)
    calib = _build_split(n=n_cal, M=M, params=params, rng=rng)
    test = _build_split(n=n_test, M=M, params=params, rng=rng)
    return SyntheticSplits(calib=calib, test=test)


def save_dataset(splits: SyntheticSplits, out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path / 'calib.npz', **splits.calib.as_dict())
    np.savez_compressed(out_path / 'test.npz', **splits.test.as_dict())


def _load_single(path: Path) -> SyntheticDataset:
    with np.load(path, allow_pickle=False) as data:
        return SyntheticDataset(
            X=np.asarray(data['X'], dtype=float),
            T=np.asarray(data['T'], dtype=float),
            V=np.asarray(data['V'], dtype=np.float32),
            A=np.asarray(data['A'], dtype=np.int8),
            S=np.asarray(data['S'], dtype=np.float32),
        )


def load_dataset(data_dir: str | Path) -> SyntheticSplits:
    data_path = Path(data_dir)
    return SyntheticSplits(
        calib=_load_single(data_path / 'calib.npz'),
        test=_load_single(data_path / 'test.npz'),
    )
