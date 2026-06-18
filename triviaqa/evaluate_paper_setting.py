from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from core.data import load_or_build_records, parse_int_spec, split_records
from core.features import difficulty_array
from core.methods import run_icp, run_method, run_topk
from core.metrics import evaluate_predictions


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float('nan'), float('nan')
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def aggregate_metrics(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ['ecr', 'apss', 'gsc']
    out = {}
    for field in fields:
        vals = [float(item[field]) for item in per_seed]
        mu, sigma = mean_std(vals)
        out[f'{field}_mean'] = mu
        out[f'{field}_std'] = sigma
    group_matrix = np.asarray([item['group_coverages'] for item in per_seed], dtype=float)
    out['group_coverages_mean'] = np.nanmean(group_matrix, axis=0).tolist()
    out['group_coverages_std'] = np.nanstd(group_matrix, axis=0).tolist()
    out['per_seed'] = per_seed
    return out


def evaluate_masks(
    records,
    selected_masks,
    group_ids: np.ndarray,
    num_groups: int,
) -> dict[str, Any]:
    covered = []
    set_sizes = []
    for record, mask in zip(records, selected_masks):
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        covered.append(any(candidate.correct for candidate, selected in zip(record.candidates, mask) if selected))
        set_sizes.append(int(np.sum(mask)))

    covered_arr = np.asarray(covered, dtype=bool)
    set_sizes_arr = np.asarray(set_sizes, dtype=float)
    gids = np.asarray(group_ids, dtype=int)

    group_coverages = []
    for group_idx in range(num_groups):
        mask = gids == group_idx
        group_coverages.append(float(np.mean(covered_arr[mask])) if np.any(mask) else float('nan'))

    return {
        'ecr': float(np.mean(covered_arr)) if covered_arr.size else float('nan'),
        'apss': float(np.mean(set_sizes_arr)) if set_sizes_arr.size else float('nan'),
        'gsc': float(np.nanmin(group_coverages)) if group_coverages else float('nan'),
        'group_coverages': group_coverages,
    }


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True)
    return ranks


def _chosen_combo_scores(records) -> np.ndarray:
    entropy = difficulty_array(records, proxy='entropy')
    max_loss = difficulty_array(records, proxy='max_loss')
    return np.maximum(_rank_normalize(entropy), _rank_normalize(max_loss))


def _assign_groups_scalar(
    calib_score: np.ndarray,
    test_score: np.ndarray,
    quantiles: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    inner = np.quantile(calib_score, np.asarray(quantiles, dtype=float))
    edges = np.concatenate(([-np.inf], np.asarray(inner, dtype=float), [np.inf]))
    calib_gid = np.searchsorted(edges[1:-1], calib_score, side='right')
    test_gid = np.searchsorted(edges[1:-1], test_score, side='right')
    return calib_gid, test_gid


def _split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    arr = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    n = arr.size
    if n == 0:
        return 1.0
    rank = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), n)
    return float(arr[rank - 1])


def _naive_quantile(scores: np.ndarray, alpha: float) -> float:
    arr = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    n = arr.size
    if n == 0:
        return 1.0
    rank = int(np.ceil(n * (1.0 - float(alpha))))
    rank = min(max(rank, 1), n)
    return float(arr[rank - 1])


def _threshold_masks(records, thresholds: np.ndarray) -> list[np.ndarray]:
    masks = []
    for record, tau in zip(records, thresholds):
        masks.append(np.asarray([candidate.score <= float(tau) for candidate in record.candidates], dtype=bool))
    return masks


def __grouped_thresholds(
    calib_records,
    calib_group_ids: np.ndarray,
    test_group_ids: np.ndarray,
    alpha: float,
    mode: str,
) -> np.ndarray:
    success_scores = np.asarray([record.success_score for record in calib_records], dtype=float)
    if mode == 'split':
        global_tau = _split_conformal_quantile(success_scores, alpha=alpha)
        quantile_fn = _split_conformal_quantile
    elif mode == 'naive':
        global_tau = _naive_quantile(success_scores, alpha=alpha)
        quantile_fn = _naive_quantile
    else:
        raise ValueError(f'Unsupported mode: {mode}')

    num_groups = int(max(np.max(calib_group_ids), np.max(test_group_ids)) + 1)
    taus = np.empty(test_group_ids.shape[0], dtype=float)
    for group_idx in range(num_groups):
        mask = calib_group_ids == group_idx
        tau = quantile_fn(success_scores[mask], alpha=alpha) if np.any(mask) else global_tau
        taus[test_group_ids == group_idx] = tau
    return taus


def _truncate_after_best_selected(record, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(selected):
        return selected.copy()
    scores = np.asarray([candidate.score for candidate in record.candidates], dtype=float)
    selected_indices = np.flatnonzero(selected)
    best_index = selected_indices[int(np.argmin(scores[selected_indices]))]
    keep_prefix = np.arange(selected.shape[0], dtype=int) <= int(best_index)
    return selected & keep_prefix


def _chosen_group_ids(calib_records, test_records) -> tuple[np.ndarray, np.ndarray]:
    calib_score = _chosen_combo_scores(calib_records)
    test_score = _chosen_combo_scores(test_records)
    return _assign_groups_scalar(calib_score, test_score, (0.925,))


def _chosen_method_masks(calib_records, test_records, alpha: float, delta: float) -> dict[str, list[np.ndarray]]:
    calib_gid, test_gid = _chosen_group_ids(calib_records, test_records)

    learnt_taus = _grouped_thresholds(calib_records, calib_gid, test_gid, alpha=alpha, mode='naive')
    learnt_masks = _threshold_masks(test_records, learnt_taus)

    cfc_full_taus = _grouped_thresholds(calib_records, calib_gid, test_gid, alpha=alpha, mode='split')
    cfc_full_masks = _threshold_masks(test_records, cfc_full_taus)
    cfc_masks = [_truncate_after_best_selected(record, mask) for record, mask in zip(test_records, cfc_full_masks)]

    epsilon = np.sqrt(np.log(1.0 / float(delta)) / max(len(calib_records), 1))
    alpha_eff = max(float(alpha) - float(epsilon), 1e-6)
    pac_taus = _grouped_thresholds(calib_records, calib_gid, test_gid, alpha=alpha_eff, mode='split')
    pac_full_masks = _threshold_masks(test_records, pac_taus)
    pac_masks = [_truncate_after_best_selected(record, mask) for record, mask in zip(test_records, pac_full_masks)]

    return {
        'LEARNT_CP': learnt_masks,
        'CFC': cfc_masks,
        'CFC_FULL': cfc_full_masks,
        'CFC_PAC': pac_masks,
        'CFC_PAC_FULL': pac_full_masks,
        '_group_ids': test_gid.tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate the chosen TriviaQA hard-tail setting.')
    parser.add_argument('--base-dir', type=str, required=True)
    parser.add_argument('--model-glob', type=str, required=True)
    parser.add_argument('--sample-seeds', type=str, default='0-19')
    parser.add_argument('--split-seeds', type=str, default='0,1,2,3,4')
    parser.add_argument('--num-docs', type=int, default=10000)
    parser.add_argument('--alphas', type=str, default='0.20,0.25,0.30,0.35,0.40,0.45')
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--force-rebuild-cache', action='store_true')
    parser.add_argument('--topk-max-k', type=int, default=20)
    parser.add_argument('--topk-selection-mode', type=str, default='closest', choices=['closest', 'at_least'])
    parser.add_argument('--pac-delta', type=float, default=0.90)
    parser.add_argument('--run-dir', type=str, required=True)
    return parser


def parse_float_list(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(',') if token.strip()]


def main() -> None:
    args = build_parser().parse_args()
    sample_seeds = parse_int_spec(args.sample_seeds)
    split_seeds = parse_int_spec(args.split_seeds)
    alphas = parse_float_list(args.alphas)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    records, cache_path = load_or_build_records(
        base_dir=args.base_dir,
        model_glob=args.model_glob,
        sample_seeds=sample_seeds,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_cache,
    )

    results: dict[str, dict[str, Any]] = {}
    for alpha in alphas:
        alpha_key = f'{alpha:.2f}'
        per_method: dict[str, list[dict[str, Any]]] = {
            name: []
            for name in ['TOPK', 'ICP', 'LEARNT_CP', 'CFC', 'CFC_FULL', 'CFC_PAC', 'CFC_PAC_FULL']
        }
        for split_seed in split_seeds:
            calib_records, test_records = split_records(records, num_docs=args.num_docs, split_seed=split_seed)
            calib_gid, test_gid = _chosen_group_ids(calib_records, test_records)
            num_groups = int(max(np.max(calib_gid), np.max(test_gid)) + 1)

            topk = run_topk(
                calib_records,
                test_records,
                alpha=alpha,
                max_k=args.topk_max_k,
                selection_mode=args.topk_selection_mode,
            )
            icp = run_icp(calib_records, test_records, alpha=alpha)
            grouped_masks = _chosen_method_masks(
                calib_records,
                test_records,
                alpha=alpha,
                delta=args.pac_delta,
            )

            for method_name, masks in [
                ('TOPK', topk.selected_masks),
                ('ICP', icp.selected_masks),
                ('LEARNT_CP', grouped_masks['LEARNT_CP']),
                ('CFC', grouped_masks['CFC']),
                ('CFC_FULL', grouped_masks['CFC_FULL']),
                ('CFC_PAC', grouped_masks['CFC_PAC']),
                ('CFC_PAC_FULL', grouped_masks['CFC_PAC_FULL']),
            ]:
                metrics = evaluate_masks(test_records, masks, np.asarray(test_gid, dtype=int), num_groups)
                per_method[method_name].append(
                    {
                        'split_seed': int(split_seed),
                        **metrics,
                    }
                )

        results[alpha_key] = {
            method_name: aggregate_metrics(entries)
            for method_name, entries in per_method.items()
        }

    ablation_rows = []
    ablation_settings = [
        ('Entropy-linear $\\Phi$', {'proxy': 'entropy', 'feature_mode': 'linear', 'feature_bins': 5}),
        ('Max-loss-linear $\\Phi$', {'proxy': 'max_loss', 'feature_mode': 'linear', 'feature_bins': 5}),
    ]
    alpha_ablate = 0.30
    for label, cfg in ablation_settings:
        row_metrics = {'CFC': [], 'CFC_PAC_FULL': []}
        for split_seed in split_seeds:
            calib_records, test_records = split_records(records, num_docs=args.num_docs, split_seed=split_seed)
            for method_name in ('CFC', 'CFC_PAC_FULL'):
                output = run_method(
                    method_name,
                    calib_records=calib_records,
                    test_records=test_records,
                    alpha=alpha_ablate,
                    proxy=cfg['proxy'],
                    seed=split_seed,
                    exact=True,
                    topk_max_k=args.topk_max_k,
                    topk_selection_mode=args.topk_selection_mode,
                    pac_delta=args.pac_delta,
                    pac_stability_scale=1.0,
                    feature_mode=cfg['feature_mode'],
                    feature_bins=cfg['feature_bins'],
                )
                difficulty = difficulty_array(test_records, proxy=cfg['proxy'])
                metrics = evaluate_predictions(
                    test_records,
                    output.selected_masks,
                    difficulty=difficulty,
                    group_bins=5,
                )
                row_metrics[method_name].append(metrics.to_dict())
        ablation_rows.append(
            {
                'setting': label,
                'metrics': {name: aggregate_metrics(entries) for name, entries in row_metrics.items()},
            }
        )

    chosen_ablation = {'CFC': [], 'CFC_PAC_FULL': []}
    for split_seed in split_seeds:
        calib_records, test_records = split_records(records, num_docs=args.num_docs, split_seed=split_seed)
        calib_gid, test_gid = _chosen_group_ids(calib_records, test_records)
        num_groups = int(max(np.max(calib_gid), np.max(test_gid)) + 1)
        grouped_masks = _chosen_method_masks(
            calib_records,
            test_records,
            alpha=alpha_ablate,
            delta=args.pac_delta,
        )
        for method_name in ('CFC', 'CFC_PAC_FULL'):
            chosen_ablation[method_name].append(
                evaluate_masks(test_records, grouped_masks[method_name], np.asarray(test_gid, dtype=int), num_groups)
            )
    ablation_rows.append(
        {
            'setting': 'Chosen $\\Phi$',
            'metrics': {name: aggregate_metrics(entries) for name, entries in chosen_ablation.items()},
        }
    )

    summary = {
        'config': {
            'base_dir': args.base_dir,
            'model_glob': args.model_glob,
            'sample_seeds': sample_seeds,
            'split_seeds': split_seeds,
            'num_docs': int(args.num_docs),
            'alphas': alphas,
            'cache_path': str(cache_path),
            'pac_delta': float(args.pac_delta),
            'chosen_split': {
                'kind': 'tail',
                'score': 'max_entropy_max_loss',
                'quantile': 0.925,
            },
        },
        'results': results,
        'ablation_alpha_0.30': ablation_rows,
    }
    summary_path = run_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'summary': str(summary_path)}, indent=2))


if __name__ == '__main__':
    main()
