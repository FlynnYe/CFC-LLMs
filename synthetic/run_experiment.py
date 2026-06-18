from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from core.data import SyntheticParams, generate_dataset, load_dataset, save_dataset
from core.features import FeatureMap
from core.methods import run_cfc, run_cfc_pac, run_icp, run_learnt_cp, run_topk
from core.metrics import aggregate_method_metrics, evaluate_prediction_result


DEFAULT_TOPK_CANDIDATES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
DEFAULT_SEEDS = [1, 2, 4, 7, 10]


def parse_int_list(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(',') if token.strip()]


def parse_method_list(raw: str) -> list[str]:
    return [token.strip().upper() for token in raw.split(',') if token.strip()]


def maybe_generate_data(args: argparse.Namespace, seed: int, data_dir: Path) -> None:
    calib_path = data_dir / 'calib.npz'
    test_path = data_dir / 'test.npz'
    if not args.regenerate_data and calib_path.exists() and test_path.exists():
        return
    params = SyntheticParams(
        eta=args.eta,
        gc0=args.gc0,
        gc1=args.gc1,
        gi0=args.gi0,
        gi1=args.gi1,
        p_scale=args.p_scale,
        gamma_c_shift=args.gamma_c_shift,
        gamma_i_shift=args.gamma_i_shift,
        tail_tau=args.tail_tau,
        tail_p_add=args.tail_p_add,
        tail_gamma_c_delta=args.tail_gamma_c_delta,
        tail_gamma_i_delta=args.tail_gamma_i_delta,
    )
    splits = generate_dataset(params=params, n_cal=args.n_cal, n_test=args.n_test, M=args.M, seed=seed)
    save_dataset(splits=splits, out_dir=data_dir)


def run_methods_for_seed(args: argparse.Namespace, seed: int, run_dir: Path) -> dict[str, dict]:
    data_dir = run_dir / 'data' / f'seed_{seed}'
    maybe_generate_data(args=args, seed=seed, data_dir=data_dir)
    splits = load_dataset(data_dir)

    outputs_dir = run_dir / 'outputs' / f'seed_{seed}'
    outputs_dir.mkdir(parents=True, exist_ok=True)

    method_metrics: dict[str, dict] = {}
    topk_candidates = parse_int_list(args.topk_candidates)

    dispatch: dict[str, Callable[[], object]] = {
        'TOPK': lambda: run_topk(splits.calib, splits.test, alpha=args.alpha, candidates=topk_candidates),
        'ICP': lambda: run_icp(splits.calib, splits.test, alpha=args.alpha),
        'LEARNT_CP': lambda: run_learnt_cp(
            splits.calib,
            splits.test,
            alpha=args.alpha,
            feature_map=FeatureMap(mode=args.learnt_basis_mode, bins=args.learnt_basis_bins),
            seed=seed,
        ),
        'CFC': lambda: run_cfc(
            splits.calib,
            splits.test,
            alpha=args.alpha,
            feature_map=FeatureMap(mode=args.cfc_basis_mode, bins=args.cfc_basis_bins),
            seed=seed,
            exact=args.exact,
        ),
        'CFC_PAC': lambda: run_cfc_pac(
            splits.calib,
            splits.test,
            alpha=args.alpha,
            delta=args.delta,
            feature_map=FeatureMap(mode=args.cfc_basis_mode, bins=args.cfc_basis_bins),
            seed=seed,
            exact=args.exact,
            mode=args.pac_mode,
            cert_size=args.cert_size,
            grid_step=args.grid_step,
            max_adjust=args.max_adjust,
            stability_scale=args.stability_scale,
            fixed_alpha_eff=args.pac_fixed_alpha_eff,
        ),
    }

    methods = parse_method_list(args.methods)
    total_methods = len(methods)
    for index, method in enumerate(methods, start=1):
        method_dir = outputs_dir / method
        metrics_path = method_dir / 'metrics.json'
        preds_path = method_dir / 'preds.npz'
        meta_path = method_dir / 'meta.json'
        print(f'[seed {seed}] {index}/{total_methods} {method}')
        if not args.force and metrics_path.exists() and preds_path.exists() and meta_path.exists():
            with open(metrics_path, 'r', encoding='utf-8') as handle:
                method_metrics[method] = json.load(handle)
            print(f'[seed {seed}] {method} reused existing outputs')
            continue

        result = dispatch[method]()
        result.save(method_dir)
        metrics = evaluate_prediction_result(result, splits.test, bins=args.eval_bins)
        method_metrics[result.name] = metrics.to_dict()
        with open(metrics_path, 'w', encoding='utf-8') as handle:
            json.dump(metrics.to_dict(), handle, indent=2)
        print(f'[seed {seed}] {method} finished')

    return method_metrics


def build_summary(seed_metrics: dict[int, dict[str, dict]]) -> dict[str, dict]:
    by_method: dict[str, list] = {}
    for _, metrics in seed_metrics.items():
        for method_name, payload in metrics.items():
            by_method.setdefault(method_name, []).append(payload)

    summary: dict[str, dict] = {}
    for method_name, payloads in by_method.items():
        from core.metrics import MethodMetrics

        metrics = [
            MethodMetrics(
                method=payload['method'],
                ecr=payload['ecr'],
                apss=payload['apss'],
                gsc=payload['gsc'],
                group_coverages=np.asarray(payload['group_coverages'], dtype=float),
                bin_edges=np.asarray(payload['bin_edges'], dtype=float),
                true_ecr=payload['true_ecr'],
                true_gsc=payload['true_gsc'],
                true_group_coverages=np.asarray(payload['true_group_coverages'], dtype=float),
            )
            for payload in payloads
        ]
        summary[method_name] = aggregate_method_metrics(metrics)
    return summary


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Clean synthetic experiment runner.')
    ap.add_argument('--run-dir', type=str, required=True, help='Directory where data, outputs, and summaries are written.')
    ap.add_argument('--methods', type=str, default='TOPK,ICP,LEARNT_CP,CFC,CFC_PAC')
    ap.add_argument('--seeds', type=str, default=','.join(str(seed) for seed in DEFAULT_SEEDS))
    ap.add_argument('--regenerate-data', action='store_true')
    ap.add_argument('--force', action='store_true', help='Rerun methods even when saved outputs already exist.')

    ap.add_argument('--n-cal', type=int, default=10000)
    ap.add_argument('--n-test', type=int, default=10000)
    ap.add_argument('--M', type=int, default=50)
    ap.add_argument('--alpha', type=float, default=0.10)
    ap.add_argument('--delta', type=float, default=0.90)
    ap.add_argument('--eval-bins', type=int, default=10)

    ap.add_argument('--eta', type=int, default=2)
    ap.add_argument('--gc0', type=float, default=0.2)
    ap.add_argument('--gc1', type=float, default=0.3)
    ap.add_argument('--gi0', type=float, default=0.6)
    ap.add_argument('--gi1', type=float, default=0.3)
    ap.add_argument('--p-scale', type=float, default=0.25)
    ap.add_argument('--gamma-c-shift', type=float, default=0.45)
    ap.add_argument('--gamma-i-shift', type=float, default=0.45)
    ap.add_argument('--tail-tau', type=float, default=0.90)
    ap.add_argument('--tail-p-add', type=float, default=0.05)
    ap.add_argument('--tail-gamma-c-delta', type=float, default=-0.05)
    ap.add_argument('--tail-gamma-i-delta', type=float, default=0.0)

    ap.add_argument('--topk-candidates', type=str, default=','.join(str(value) for value in DEFAULT_TOPK_CANDIDATES))

    ap.add_argument('--learnt-basis-mode', type=str, default='group', choices=['intercept', 'linear', 'group'])
    ap.add_argument('--learnt-basis-bins', type=int, default=4)
    ap.add_argument('--cfc-basis-mode', type=str, default='group', choices=['intercept', 'linear', 'group'])
    ap.add_argument('--cfc-basis-bins', type=int, default=10)
    ap.add_argument('--no-exact', action='store_true', help='Use binary-search CondConf prediction instead of the exact finite-dimensional path.')

    ap.add_argument('--pac-mode', type=str, default='stability', choices=['stability', 'holdout'])
    ap.add_argument('--stability-scale', type=float, default=1.0)
    ap.add_argument('--cert-size', type=int, default=2000)
    ap.add_argument('--grid-step', type=float, default=0.01)
    ap.add_argument('--max-adjust', type=float, default=0.20)
    ap.add_argument('--pac-fixed-alpha-eff', type=float, default=None, help='Override PAC selection and use a fixed alpha_eff.')
    return ap


if __name__ == '__main__':
    args = parser().parse_args()
    args.exact = not args.no_exact
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    with open(run_dir / 'config.json', 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2)

    seed_metrics: dict[int, dict[str, dict]] = {}
    for seed in parse_int_list(args.seeds):
        seed_metrics[seed] = run_methods_for_seed(args=args, seed=seed, run_dir=run_dir)

    summary = build_summary(seed_metrics)
    with open(run_dir / 'summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
