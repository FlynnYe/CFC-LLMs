from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compare(expected, observed, path: str, tol: float, errors: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            errors.append(f'{path}: expected dict, observed {type(observed).__name__}')
            return
        expected_keys = set(expected.keys())
        observed_keys = set(observed.keys())
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        if missing:
            errors.append(f'{path}: missing keys {missing}')
        if extra:
            errors.append(f'{path}: extra keys {extra}')
        for key in sorted(expected_keys & observed_keys):
            compare(expected[key], observed[key], f'{path}.{key}' if path else key, tol, errors)
        return

    if isinstance(expected, list):
        if not isinstance(observed, list):
            errors.append(f'{path}: expected list, observed {type(observed).__name__}')
            return
        if len(expected) != len(observed):
            errors.append(f'{path}: expected list length {len(expected)}, observed {len(observed)}')
            return
        for idx, (exp_item, obs_item) in enumerate(zip(expected, observed)):
            compare(exp_item, obs_item, f'{path}[{idx}]', tol, errors)
        return

    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        exp = float(expected)
        obs = float(observed)
        if math.isnan(exp) and math.isnan(obs):
            return
        if abs(exp - obs) > tol:
            errors.append(f'{path}: expected {exp}, observed {obs}, diff {abs(exp - obs)}')
        return

    if expected != observed:
        errors.append(f'{path}: expected {expected!r}, observed {observed!r}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Verify reproduced release outputs against the expected normalized snapshot.')
    parser.add_argument('--expected', type=str, required=True)
    parser.add_argument('--observed', type=str, required=True)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    expected = json.loads(Path(args.expected).read_text())
    observed = json.loads(Path(args.observed).read_text())
    errors: list[str] = []
    compare(expected, observed, '', float(args.tolerance), errors)
    if errors:
        for error in errors[:100]:
            print(error)
        raise SystemExit(1)
    print('Verification passed.')


if __name__ == '__main__':
    main()
