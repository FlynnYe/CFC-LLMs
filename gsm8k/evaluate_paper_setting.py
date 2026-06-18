from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from core.data import load_or_build_records, parse_int_spec, split_records
from core.features import difficulty_array
from core.methods import run_method
from core.metrics import evaluate_predictions


SWEEP_ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SWEEP_METHODS = ['TOPK', 'ICP', 'LEARNT_CP', 'CFC', 'CFC_FULL', 'CFC_PAC', 'CFC_PAC_FULL']
ABLATION_METHODS = ['TOPK', 'ICP', 'LEARNT_CP', 'CFC', 'CFC_FULL', 'CFC_PAC', 'CFC_PAC_FULL']


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float('nan'), float('nan')
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def aggregate_metrics(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ['ecr', 'apss', 'gsc', 'selection_rate']
    summary: dict[str, Any] = {}
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


def evaluate_setting(
    records,
    split_seeds: list[int],
    alphas: list[float],
    max_samples: int,
    pac_delta: float,
    pac_stability_scale: float,
) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for alpha in alphas:
        alpha_key = f'{alpha:.2f}'
        aggregated[alpha_key] = {}
        for method in SWEEP_METHODS:
            per_seed_results: list[dict[str, Any]] = []
            for split_seed in split_seeds:
                calib_records, test_records = split_records(records, num_docs=0, split_seed=split_seed)
                method_output = run_method(
                    method,
                    calib_records=calib_records,
                    test_records=test_records,
                    alpha=alpha,
                    proxy='mean_loss',
                    seed=split_seed,
                    exact=True,
                    topk_max_k=max_samples,
                    topk_selection_mode='closest',
                    pac_delta=pac_delta,
                    pac_stability_scale=pac_stability_scale,
                    feature_mode='poly2',
                    feature_bins=5,
                )
                difficulty = difficulty_array(test_records, proxy='mean_loss')
                metrics = evaluate_predictions(
                    test_records,
                    method_output.selected_masks,
                    difficulty=difficulty,
                    group_bins=5,
                )
                per_seed_results.append(
                    {
                        'split_seed': int(split_seed),
                        **metrics.to_dict(),
                        'metadata': method_output.metadata,
                    }
                )
            aggregated[alpha_key][method] = aggregate_metrics(per_seed_results)
    return aggregated


def evaluate_budget_ablation(
    records_by_n: dict[int, list],
    split_seeds: list[int],
    alpha: float,
    pac_delta: float,
    pac_stability_scale: float,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for max_samples, records in records_by_n.items():
        per_method: dict[str, list[dict[str, Any]]] = {name: [] for name in ABLATION_METHODS}
        for split_seed in split_seeds:
            calib_records, test_records = split_records(records, num_docs=0, split_seed=split_seed)
            for method in ABLATION_METHODS:
                method_output = run_method(
                    method,
                    calib_records=calib_records,
                    test_records=test_records,
                    alpha=alpha,
                    proxy='mean_loss',
                    seed=split_seed,
                    exact=True,
                    topk_max_k=max_samples,
                    topk_selection_mode='closest',
                    pac_delta=pac_delta,
                    pac_stability_scale=pac_stability_scale,
                    feature_mode='poly2',
                    feature_bins=5,
                )
                difficulty = difficulty_array(test_records, proxy='mean_loss')
                metrics = evaluate_predictions(
                    test_records,
                    method_output.selected_masks,
                    difficulty=difficulty,
                    group_bins=5,
                )
                per_method[method].append(
                    {
                        'split_seed': int(split_seed),
                        **metrics.to_dict(),
                        'metadata': method_output.metadata,
                    }
                )
        out[str(max_samples)] = {
            method_name: aggregate_metrics(entries) for method_name, entries in per_method.items()
        }
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate the GSM8K paper setting and sample-budget ablation.')
    parser.add_argument('--score-dir', type=str, required=True)
    parser.add_argument('--score-glob', type=str, default='gsm8k_with_rm_*.jsonl')
    parser.add_argument('--split-seeds', type=str, default='0,1,2,3,4')
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--force-rebuild-cache', action='store_true')
    parser.add_argument('--pac-delta', type=float, default=0.90)
    parser.add_argument('--pac-stability-scale', type=float, default=1.0)
    parser.add_argument('--run-dir', type=str, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split_seeds = parse_int_spec(args.split_seeds)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    records_5, cache_5 = load_or_build_records(
        score_dir=args.score_dir,
        score_glob=args.score_glob,
        max_samples=5,
        collapse_answers=False,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_cache,
    )
    records_20, cache_20 = load_or_build_records(
        score_dir=args.score_dir,
        score_glob=args.score_glob,
        max_samples=20,
        collapse_answers=False,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_cache,
    )

    summary = {
        'config': {
            'score_dir': args.score_dir,
            'score_glob': args.score_glob,
            'split_seeds': split_seeds,
            'sweep_alphas': SWEEP_ALPHAS,
            'paper_setting': {
                'max_samples': 5,
                'difficulty_proxy': 'mean_loss',
                'feature_mode': 'poly2',
                'group_bins': 5,
            },
            'budget_ablation': {
                'alpha': 0.10,
                'max_samples': [5, 20],
            },
            'cache_paths': {
                '5': str(cache_5),
                '20': str(cache_20),
            },
            'pac_delta': float(args.pac_delta),
            'pac_stability_scale': float(args.pac_stability_scale),
        },
        'results': evaluate_setting(
            records_5,
            split_seeds=split_seeds,
            alphas=SWEEP_ALPHAS,
            max_samples=5,
            pac_delta=args.pac_delta,
            pac_stability_scale=args.pac_stability_scale,
        ),
        'n_ablation_alpha_0.10': evaluate_budget_ablation(
            {5: records_5, 20: records_20},
            split_seeds=split_seeds,
            alpha=0.10,
            pac_delta=args.pac_delta,
            pac_stability_scale=args.pac_stability_scale,
        ),
    }
    summary_path = run_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'summary': str(summary_path)}, indent=2))


if __name__ == '__main__':
    main()
