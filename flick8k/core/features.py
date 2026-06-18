from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DocumentRecord


DIFFICULTY_PROXIES = {
    'min_loss',
    'mean_loss',
    'max_loss',
    'spread',
    'num_candidates',
    'avg_caption_length',
    'length_std',
    'hybrid',
}
FEATURE_MODES = {
    'linear',
    'poly2',
    'poly3',
    'group',
}


@dataclass
class LinearDifficultyFeatureMap:
    proxy: str = 'mean_loss'

    def fit(self, x) -> 'LinearDifficultyFeatureMap':
        del x
        return self

    def __call__(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1, 1)
        return np.column_stack([np.ones(arr.shape[0], dtype=float), arr[:, 0]])

    def to_dict(self) -> dict:
        return {'mode': 'linear_difficulty', 'proxy': self.proxy}


@dataclass
class PolynomialDifficultyFeatureMap:
    proxy: str = 'mean_loss'
    degree: int = 2

    def __call__(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        columns = [np.ones(arr.shape[0], dtype=float)]
        for power in range(1, int(self.degree) + 1):
            columns.append(arr ** power)
        return np.column_stack(columns)

    def to_dict(self) -> dict:
        return {
            'mode': 'polynomial_difficulty',
            'proxy': self.proxy,
            'degree': int(self.degree),
        }


@dataclass
class GroupedDifficultyFeatureMap:
    proxy: str = 'mean_loss'
    bin_edges: np.ndarray | None = None

    def __call__(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if self.bin_edges is None:
            raise RuntimeError('GroupedDifficultyFeatureMap requires bin_edges before use.')
        edges = np.asarray(self.bin_edges, dtype=float).reshape(-1)
        if edges.size < 2:
            raise ValueError('bin_edges must contain at least two entries.')

        bins = edges.size - 1
        phi = np.zeros((arr.shape[0], bins), dtype=float)
        for idx, value in enumerate(arr):
            group = np.searchsorted(edges[1:-1], float(value), side='right')
            phi[idx, group] = 1.0
        return phi

    def to_dict(self) -> dict:
        return {
            'mode': 'grouped_difficulty',
            'proxy': self.proxy,
            'bin_edges': None if self.bin_edges is None else [float(value) for value in self.bin_edges],
        }


def compute_difficulty(record: DocumentRecord, proxy: str = 'mean_loss') -> float:
    if proxy == 'min_loss':
        return float(record.min_score)
    if proxy == 'mean_loss':
        return float(record.mean_score)
    if proxy == 'max_loss':
        return float(record.max_score)
    if proxy == 'spread':
        return float(record.score_spread)
    if proxy == 'num_candidates':
        return float(record.num_candidates)
    if proxy == 'avg_caption_length':
        return float(record.avg_caption_length)
    if proxy == 'length_std':
        return float(record.caption_length_std)
    if proxy == 'hybrid':
        return float(0.5 * record.mean_score + 0.5 * record.score_spread)
    raise ValueError(f'Unsupported difficulty proxy: {proxy}')


def difficulty_array(records: list[DocumentRecord], proxy: str = 'mean_loss') -> np.ndarray:
    return np.asarray([compute_difficulty(record, proxy=proxy) for record in records], dtype=float)


def equal_frequency_edges(values: np.ndarray, bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError('values must be non-empty.')
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)
    edges = np.quantile(arr, quantiles)
    edges[0] = min(edges[0], np.min(arr))
    edges[-1] = max(edges[-1], np.max(arr))
    return np.asarray(edges, dtype=float)


def build_feature_map(
    calib_difficulty: np.ndarray,
    proxy: str = 'mean_loss',
    mode: str = 'linear',
    bins: int = 5,
):
    if mode == 'linear':
        return LinearDifficultyFeatureMap(proxy=proxy)
    if mode == 'poly2':
        return PolynomialDifficultyFeatureMap(proxy=proxy, degree=2)
    if mode == 'poly3':
        return PolynomialDifficultyFeatureMap(proxy=proxy, degree=3)
    if mode == 'group':
        edges = equal_frequency_edges(np.asarray(calib_difficulty, dtype=float), bins=bins)
        return GroupedDifficultyFeatureMap(proxy=proxy, bin_edges=edges)
    raise ValueError(f'Unsupported feature mode: {mode}')
