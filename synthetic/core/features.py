from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureMap:
    mode: str = 'linear'
    bins: int = 10
    edges: np.ndarray | None = None

    def fit(self, t_calib: np.ndarray) -> 'FeatureMap':
        t_calib = np.asarray(t_calib, dtype=float).reshape(-1)
        if self.mode == 'group':
            edges = np.quantile(t_calib, np.linspace(0.0, 1.0, int(self.bins) + 1))
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = edges[i - 1] + 1e-12
            self.edges = edges.astype(float)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError(f'Expected x with shape (n, d>=1); got {values.shape}.')
        t = values[:, 0]
        if self.mode == 'intercept':
            return np.ones((len(t), 1), dtype=float)
        if self.mode == 'linear':
            return np.stack([np.ones_like(t), t], axis=1)
        if self.mode == 'group':
            if self.edges is None:
                raise RuntimeError('Call fit() on the feature map before using group mode.')
            idx = np.digitize(t, self.edges[1:-1], right=True)
            one_hot = np.zeros((len(t), len(self.edges) - 1), dtype=float)
            one_hot[np.arange(len(t)), idx] = 1.0
            return one_hot
        raise ValueError(f'Unsupported feature-map mode: {self.mode}')

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {'mode': self.mode}
        if self.mode == 'group':
            payload['bins'] = int(self.bins)
            payload['edges'] = None if self.edges is None else self.edges.tolist()
        return payload
