from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from core.data import load_or_build_records, parse_int_spec, split_records
from core.features import DIFFICULTY_PROXIES, FEATURE_MODES, difficulty_array
from core.methods import METHOD_NAMES, run_method
from core.metrics import evaluate_predictions


def parse_method_list(raw: str) -> list[str]:
    methods = [token.strip().upper() for token in raw.split(',') if token.strip()]
    for method in methods:
        if method not in METHOD_NAMES:
            raise ValueError(f'Unsupported method: {method}')
    return methods


def parse_float_list(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(',') if token.strip()]


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float('nan'), float('nan')
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def aggregate_metrics(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ['ecr', 'apss', 'gsc', 'selection_rate']
    summary = {}
    for field in fields:
        values = [float(item[field]) for item in per_seed]
        mu, sigma = mean_std(values)
        summary[f'{field}_mean'] = mu
        summary[f'{field}_std'] = sigma
    group_matrix = np.asarray([item['group_coverages'] for item in per_seed], dtype=float)
    summary['group_coverages_mean'] = np.nanmean(group_matrix, axis=0).tolist()
    summary['group_coverages_std'] = np.nanstd(group_matrix, axis=0).tolist()
    summary['per_seed'] = per_seed
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Clean TriviaQA experiment runner.')
    parser.add_argument('--base-dir', type=str, required=True, help='Directory containing seed_{i}/<model>/samples*_scored.jsonl.')
    parser.add_argument('--model-glob', type=str, default='*')
    parser.add_argument('--sample-seeds', type=str, default='0-19', help='Seeds used as the M sampled candidates per prompt.')
    parser.add_argument('--split-seeds', type=str, default='0,1,2,3,4', help='RNG seeds for repeated calibration/test splits.')
    parser.add_argument('--num-docs', type=int, default=10000, help='Number of complete docs to subsample before the 50/50 split.')
    parser.add_argument('--alphas', type=str, default='0.20,0.25,0.30,0.35,0.40,0.45')
    parser.add_argument('--methods', type=str, default='TOPK,ICP,CFC,CFC_PAC')
    parser.add_argument('--difficulty-proxy', type=str, default='disagreement', choices=sorted(DIFFICULTY_PROXIES))
    parser.add_argument('--feature-mode', type=str, default='linear', choices=sorted(FEATURE_MODES))
    parser.add_argument('--feature-bins', type=int, default=5, help='Number of equal-frequency bins for group feature mode.')
    parser.add_argument('--string-agg', type=str, default='mean', choices=['mean', 'min', 'max', 'median', 'q25', 'q75'])
    parser.add_argument('--group-bins', type=int, default=5)
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--run-dir', type=str, required=True)
    parser.add_argument('--force-rebuild-cache', action='store_true')
    parser.add_argument('--no-exact', action='store_true')
    parser.add_argument('--topk-max-k', type=int, default=20)
    parser.add_argument('--topk-selection-mode', type=str, default='closest', choices=['closest', 'at_least'])
    parser.add_argument('--pac-delta', type=float, default=0.90)
    parser.add_argument('--pac-stability-scale', type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    methods = parse_method_list(args.methods)
    alphas = parse_float_list(args.alphas)
    sample_seeds = parse_int_spec(args.sample_seeds)
    split_seeds = parse_int_spec(args.split_seeds)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    records, cache_path = load_or_build_records(
        base_dir=args.base_dir,
        model_glob=args.model_glob,
        sample_seeds=sample_seeds,
        string_agg=args.string_agg,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_cache,
    )

    config = {
        'base_dir': args.base_dir,
        'model_glob': args.model_glob,
        'sample_seeds': sample_seeds,
        'split_seeds': split_seeds,
        'num_docs': int(args.num_docs),
        'alphas': alphas,
        'methods': methods,
        'difficulty_proxy': args.difficulty_proxy,
        'feature_mode': args.feature_mode,
        'feature_bins': int(args.feature_bins),
        'string_agg': args.string_agg,
        'group_bins': int(args.group_bins),
        'cache_path': str(cache_path),
        'exact': not args.no_exact,
        'topk_max_k': int(args.topk_max_k),
        'topk_selection_mode': args.topk_selection_mode,
        'pac_delta': float(args.pac_delta),
        'pac_stability_scale': float(args.pac_stability_scale),
        'docs_available': len(records),
    }
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    aggregated: dict[str, dict[str, Any]] = {}
    for alpha in alphas:
        alpha_key = f'{alpha:.2f}'
        aggregated[alpha_key] = {}
        for method in methods:
            per_seed_results: list[dict[str, Any]] = []
            for split_seed in split_seeds:
                calib_records, test_records = split_records(records, num_docs=args.num_docs, split_seed=split_seed)
                method_output = run_method(
                    method,
                    calib_records=calib_records,
                    test_records=test_records,
                    alpha=alpha,
                    proxy=args.difficulty_proxy,
                    seed=split_seed,
                    exact=not args.no_exact,
                    topk_max_k=args.topk_max_k,
                    topk_selection_mode=args.topk_selection_mode,
                    pac_delta=args.pac_delta,
                    pac_stability_scale=args.pac_stability_scale,
                    feature_mode=args.feature_mode,
                    feature_bins=args.feature_bins,
                )
                difficulty = difficulty_array(test_records, proxy=args.difficulty_proxy)
                metrics = evaluate_predictions(
                    test_records,
                    method_output.selected_masks,
                    difficulty=difficulty,
                    group_bins=args.group_bins,
                )
                per_seed_results.append(
                    {
                        'split_seed': int(split_seed),
                        **metrics.to_dict(),
                        'metadata': method_output.metadata,
                    }
                )
            aggregated[alpha_key][method] = aggregate_metrics(per_seed_results)

    summary = {
        'config': config,
        'results': aggregated,
    }
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
