from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import run_experiment as ce
from core.data import load_dataset
from core.features import FeatureMap
from core.methods import run_cfc, split_conformal_quantile


SEEDS = [1, 2, 4, 7, 10]
ALPHAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
N_CALS = [2000, 5000, 10000]
MS = [50, 100, 150]
ABLATION_BINS = [10]

DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / 'outputs' / 'synthetic'
BASE_DIR = DEFAULT_BASE_DIR
RUNS_DIR = BASE_DIR / 'runs'
TABLES_DIR = BASE_DIR / 'tables'
FIGS_DIR = BASE_DIR / 'figs'
DATA_DIR = BASE_DIR / 'data'

METHOD_COLORS = {
    'TOPK': '#4C72B0',
    'ICP': '#DD8452',
    'LEARNT_CP': '#55A868',
    'CFC': '#C44E52',
    'CFC_PAC': '#8172B3',
}

DISPLAY_NAMES = {
    'TOPK': 'TopK',
    'ICP': 'ICP',
    'LEARNT_CP': 'Learnt CP',
    'CFC': 'CFC',
    'CFC_PAC': 'CFC-PAC',
}


def fmt_percent(mean: float, std: float) -> str:
    return f'{100.0 * mean:.1f} $\\pm$ {100.0 * std:.1f}'


def fmt_float(mean: float, std: float) -> str:
    return f'{mean:.2f} $\\pm$ {std:.2f}'


def set_output_root(base_dir: Path) -> None:
    global BASE_DIR, RUNS_DIR, TABLES_DIR, FIGS_DIR, DATA_DIR
    BASE_DIR = Path(base_dir)
    RUNS_DIR = BASE_DIR / 'runs'
    TABLES_DIR = BASE_DIR / 'tables'
    FIGS_DIR = BASE_DIR / 'figs'
    DATA_DIR = BASE_DIR / 'data'


def ensure_dirs() -> None:
    for path in [BASE_DIR, RUNS_DIR, TABLES_DIR, FIGS_DIR, DATA_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def run_experiment(run_dir: Path, extra_args: list[str]) -> dict[str, dict]:
    parser = ce.parser()
    args = parser.parse_args(['--run-dir', str(run_dir)] + extra_args)
    args.exact = not args.no_exact
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    write_json(run_dir / 'config.json', config)

    seed_metrics: dict[int, dict[str, dict]] = {}
    for seed in ce.parse_int_list(args.seeds):
        seed_metrics[seed] = ce.run_methods_for_seed(args=args, seed=seed, run_dir=run_dir)
    summary = ce.build_summary(seed_metrics)
    write_json(run_dir / 'summary.json', summary)
    return summary


def load_seed_metrics(run_dir: Path, method: str) -> list[dict[str, Any]]:
    payloads = []
    for seed in SEEDS:
        with open(run_dir / 'outputs' / f'seed_{seed}' / method / 'metrics.json', 'r', encoding='utf-8') as handle:
            payloads.append(json.load(handle))
    return payloads


def compute_group_coverages(success: np.ndarray, t: np.ndarray, bins: int) -> np.ndarray:
    edges = np.quantile(t, np.linspace(0.0, 1.0, int(bins) + 1))
    coverages = np.empty(int(bins), dtype=float)
    for b in range(int(bins)):
        lo = edges[b]
        hi = edges[b + 1] + 1e-12
        idx = (t >= lo) & (t <= hi)
        coverages[b] = float(np.mean(success[idx])) if np.any(idx) else np.nan
    return coverages


def recompute_true_group_coverages(run_dir: Path, method: str, bins: int) -> list[list[float]]:
    all_coverages: list[list[float]] = []
    for seed in SEEDS:
        splits = load_dataset(run_dir / 'data' / f'seed_{seed}')
        with np.load(run_dir / 'outputs' / f'seed_{seed}' / method / 'preds.npz', allow_pickle=False) as preds:
            accept_mask = np.asarray(preds['accept_mask'], dtype=bool)
        true_success = np.any(accept_mask & splits.test.A.astype(bool), axis=1)
        group_coverages = compute_group_coverages(true_success, splits.test.T, bins=bins)
        all_coverages.append(group_coverages.tolist())
    return all_coverages


def build_main_compact_table(summary: dict[str, dict]) -> str:
    order = ['TOPK', 'ICP', 'LEARNT_CP', 'CFC', 'CFC_PAC']
    lines = [
        '\\begin{tabular}{lccc}',
        '\\toprule',
        'Method & ECR & APSS$\\downarrow$ & GSC$\\uparrow$ \\\\',
        '\\midrule',
    ]
    for method in order:
        payload = summary[method]
        name = DISPLAY_NAMES[method]
        if method in {'CFC', 'CFC_PAC'}:
            name = f'\\textbf{{{name} (ours)}}'
        lines.append(
            f'{name} & '
            f'{fmt_percent(payload["true_ecr_mean"], payload["true_ecr_std"])} & '
            f'{fmt_float(payload["apss_mean"], payload["apss_std"])} & '
            f'{fmt_percent(payload["true_gsc_seed_mean"], payload["true_gsc_seed_std"])} \\\\'
        )
    lines.extend(['\\bottomrule', '\\end{tabular}'])
    return '\n'.join(lines) + '\n'


def build_alpha_sweep_table(summaries: dict[float, dict[str, dict]]) -> str:
    blocks = [(0.10, 0.15), (0.20, 0.25), (0.30, 0.35), (0.40, 0.45)]
    order = ['TOPK', 'ICP', 'LEARNT_CP', 'CFC', 'CFC_PAC']
    lines = ['\\begin{tabular}{l ccc ccc}', '\\toprule']
    for block_index, (alpha_left, alpha_right) in enumerate(blocks):
        lines.append(
            'Methods & '
            f'\\multicolumn{{3}}{{c}}{{$\\alpha = {alpha_left:.2f}$}} & '
            f'\\multicolumn{{3}}{{c}}{{$\\alpha = {alpha_right:.2f}$}} \\\\'
        )
        lines.append('\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}')
        lines.append(
            '& ECR & GSC$\\uparrow$ & APSS$\\downarrow$'
            '& ECR & GSC$\\uparrow$ & APSS$\\downarrow$ \\\\'
        )
        lines.append('\\midrule')
        for method in order:
            left = summaries[alpha_left][method]
            right = summaries[alpha_right][method]
            name = DISPLAY_NAMES[method]
            if method in {'CFC', 'CFC_PAC'}:
                name = f'\\textbf{{{name} (ours)}}'
            lines.append(
                f'{name} & '
                f'{fmt_percent(left["true_ecr_mean"], left["true_ecr_std"])} & '
                f'{fmt_percent(left["true_gsc_seed_mean"], left["true_gsc_seed_std"])} & '
                f'{fmt_float(left["apss_mean"], left["apss_std"])} & '
                f'{fmt_percent(right["true_ecr_mean"], right["true_ecr_std"])} & '
                f'{fmt_percent(right["true_gsc_seed_mean"], right["true_gsc_seed_std"])} & '
                f'{fmt_float(right["apss_mean"], right["apss_std"])} \\\\'
            )
        lines.append('\\midrule' if block_index < len(blocks) - 1 else '\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines) + '\n'


def build_ablation_table(aggregated: dict[tuple[int, int], dict[str, dict]], bins: int) -> str:
    lines = [
        '\\begin{tabular}{cc cccc}',
        '\\toprule',
        ' & & \\multicolumn{2}{c}{CFC} & \\multicolumn{2}{c}{CFC-PAC} \\\\',
        '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}',
        '$N_{\\text{cal}}$ & $M$ & True cov. & Mean set & True cov. & Mean set \\\\',
        '\\midrule',
    ]
    for i, n_cal in enumerate(N_CALS):
        for m in MS:
            payload = aggregated[(n_cal, m)]
            cfc = payload['CFC']
            pac = payload['CFC_PAC']
            lines.append(
                f'{n_cal} & {m} & '
                f'{cfc["true_cov"]:.3f} $\\pm$ {cfc["true_cov_std"]:.3f} & '
                f'{cfc["set_mean"]:.2f} $\\pm$ {cfc["set_std"]:.2f} & '
                f'{pac["true_cov"]:.3f} $\\pm$ {pac["true_cov_std"]:.3f} & '
                f'{pac["set_mean"]:.2f} $\\pm$ {pac["set_std"]:.2f} \\\\'
            )
        lines.append('\\midrule' if i < len(N_CALS) - 1 else '\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines) + '\n'


def plot_synfg1(summary: dict[str, dict], alpha: float, out_path: Path) -> None:
    methods = ['TOPK', 'ICP', 'LEARNT_CP', 'CFC', 'CFC_PAC']
    colors = {
        'TOPK': '#4C72B0',
        'ICP': '#DD8452',
        'LEARNT_CP': '#8C8C8C',
        'CFC': '#55A868',
        'CFC_PAC': '#B07AA1',
    }
    x = np.arange(1, 11)
    width = 0.15

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for idx, method in enumerate(methods):
        payload = summary[method]
        miscoverage = 1.0 - np.asarray(payload['true_group_coverages_mean'], dtype=float)
        std = np.asarray(payload['true_group_coverages_std'], dtype=float)
        upper_only_std = np.vstack([np.zeros_like(std), std])
        offset = (idx - (len(methods) - 1) / 2.0) * width
        centers = x + offset
        ax.bar(
            centers,
            miscoverage,
            width=width,
            color=colors[method],
            edgecolor='black',
            linewidth=0.7,
            alpha=0.96,
            yerr=upper_only_std,
            error_kw={
                'ecolor': '#333333',
                'elinewidth': 1.0,
                'capsize': 3,
                'capthick': 1.0,
            },
            zorder=3,
            label=DISPLAY_NAMES[method],
        )

    ax.axhline(float(alpha), color='black', linestyle='--', linewidth=1.8, label='target')
    ax.set_xlabel('Group', fontsize=22)
    ax.set_ylabel('Miscoverage', fontsize=22)
    ax.set_xticks(x)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_ylim(0.0, max(float(alpha) + 0.02, 0.5))
    ax.grid(axis='y', alpha=0.22, linewidth=0.7)
    ax.legend(
        loc='upper left',
        ncol=2,
        frameon=False,
        fontsize=16,
        handlelength=1.8,
        columnspacing=1.0,
        borderaxespad=0.3,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_threshold_adaptation(run_dir: Path, alpha: float, bins: int, out_path: Path) -> dict[str, Any]:
    seed = SEEDS[0]
    splits = load_dataset(run_dir / 'data' / f'seed_{seed}')
    feature_map = FeatureMap(mode='group', bins=bins)
    result = run_cfc(splits.calib, splits.test, alpha=alpha, feature_map=feature_map, seed=seed, exact=True)
    fitted = FeatureMap(mode='group', bins=bins)
    fitted.fit(splits.calib.T)
    edges = np.asarray(fitted.edges, dtype=float)
    thresholds = []
    for b in range(bins):
        lo = edges[b]
        hi = edges[b + 1] + 1e-12
        idx = (splits.test.T >= lo) & (splits.test.T <= hi)
        thresholds.append(float(np.median(result.thresholds[idx])))
    thresholds = np.asarray(thresholds, dtype=float)
    global_threshold = split_conformal_quantile(splits.calib.S, alpha)

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(splits.calib.n, size=min(1500, splits.calib.n), replace=False)
    t_scatter = splits.calib.T[sample_idx]
    s_scatter = splits.calib.S[sample_idx]

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(t_scatter, s_scatter, s=28, alpha=0.18, color='gray', label='Calibration scores')
    cmap = plt.get_cmap('viridis')
    for b, threshold in enumerate(thresholds):
        color = cmap(b / max(1, bins - 1))
        label = 'Group 1 (Easy)' if b == 0 else ('Group 5 (Hard)' if b == bins - 1 and bins == 5 else f'Group {b + 1}')
        ax.hlines(threshold, edges[b], edges[b + 1], colors=[color], linewidth=4.0, label=label)
        ax.scatter([(edges[b] + edges[b + 1]) / 2.0], [threshold], s=150, color=color, edgecolors='black', linewidths=1.7, zorder=3)
    ax.axhline(global_threshold, color='red', linestyle='--', linewidth=2.6, label=f'ICP threshold ({global_threshold:.2f})')
    ax.set_xlabel('Prompt difficulty $T$', fontsize=21)
    ax.set_ylabel('Threshold $\\hat\\lambda$', fontsize=21)
    ax.tick_params(axis='both', labelsize=17)
    ax.set_ylim(0.0, min(1.05, max(float(np.max(s_scatter)), float(np.max(thresholds))) + 0.05))
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(
        loc='upper left',
        frameon=False,
        fontsize=14,
        handlelength=1.8,
        columnspacing=0.9,
        borderaxespad=0.3,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    payload = {
        'seed': seed,
        'alpha': alpha,
        'bins': bins,
        'bin_edges': edges.tolist(),
        'group_thresholds': thresholds.tolist(),
        'icp_threshold': float(global_threshold),
    }
    write_json(out_path.with_suffix('.json'), payload)
    return payload


def plot_ablation_group_miscoverage(bin_payloads: list[dict[str, Any]], bins: int, out_path: Path) -> dict[str, Any]:
    methods = ['CFC', 'CFC_PAC']
    pooled = {method: [] for method in methods}
    for payload in bin_payloads:
        for method in pooled:
            pooled[method].extend(payload[method]['true_group_coverages'])

    x = np.arange(1, bins + 1)
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    summary: dict[str, dict[str, list[float]]] = {}
    offsets = {
        'CFC': -width / 2.0,
        'CFC_PAC': width / 2.0,
    }
    for method in methods:
        matrix = np.asarray(pooled[method], dtype=float)
        miscoverage = 1.0 - matrix
        mean = np.mean(miscoverage, axis=0)
        std = np.std(miscoverage, axis=0)
        summary[method] = {'mean_miscoverage': mean.tolist(), 'std_miscoverage': std.tolist()}
        ax.bar(
            x + offsets[method],
            mean,
            width=width,
            yerr=std,
            capsize=4,
            color=METHOD_COLORS[method],
            edgecolor='black',
            linewidth=0.7,
            alpha=0.9,
            label=DISPLAY_NAMES[method],
        )
    ax.axhline(0.20, color='black', linestyle='--', linewidth=1.8, label='target')
    ax.set_xlabel('Group', fontsize=18)
    ax.set_ylabel('Miscoverage', fontsize=18)
    ax.set_xticks(x)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_title(f'Group miscoverage, bins={bins}', fontsize=18)
    ax.grid(axis='y', alpha=0.2, linewidth=0.7)
    ax.legend(frameon=False, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    write_json(out_path.with_suffix('.json'), summary)
    return summary


def run_main_table() -> dict[str, dict]:
    run_dir = RUNS_DIR / 'synthetic_alpha_010_main'
    summary = run_experiment(
        run_dir,
        [
            '--seeds', ','.join(str(seed) for seed in SEEDS),
            '--methods', 'TOPK,ICP,LEARNT_CP,CFC,CFC_PAC',
            '--alpha', '0.10',
            '--n-cal', '10000',
            '--n-test', '10000',
            '--M', '50',
            '--delta', '0.90',
            '--learnt-basis-mode', 'group',
            '--learnt-basis-bins', '4',
            '--cfc-basis-mode', 'group',
            '--cfc-basis-bins', '10',
        ],
    )
    write_json(TABLES_DIR / 'synthetic_compact_summary.json', summary)
    compact_tex = build_main_compact_table(summary)
    (TABLES_DIR / 'synthetic_compact_main.tex').write_text(compact_tex, encoding='utf-8')
    (TABLES_DIR / 'synthetic_compact_appendix.tex').write_text(compact_tex, encoding='utf-8')
    plot_synfg1(summary, alpha=0.10, out_path=FIGS_DIR / 'synfg1.pdf')
    return summary


def run_alpha_sweep() -> dict[float, dict[str, dict]]:
    summaries: dict[float, dict[str, dict]] = {}
    for alpha in ALPHAS:
        run_dir = RUNS_DIR / f'synthetic_alpha_{int(round(alpha * 100)):02d}'
        summaries[alpha] = run_experiment(
            run_dir,
            [
                '--seeds', ','.join(str(seed) for seed in SEEDS),
                '--methods', 'TOPK,ICP,LEARNT_CP,CFC,CFC_PAC',
                '--alpha', f'{alpha:.2f}',
                '--n-cal', '10000',
                '--n-test', '10000',
                '--M', '50',
                '--delta', '0.90',
                '--learnt-basis-mode', 'group',
                '--learnt-basis-bins', '4',
                '--cfc-basis-mode', 'group',
                '--cfc-basis-bins', '10',
            ],
        )
    write_json(TABLES_DIR / 'synthetic_alpha_sweep_summary.json', {f'{alpha:.2f}': payload for alpha, payload in summaries.items()})
    (TABLES_DIR / 'synthetic_alpha_sweep.tex').write_text(build_alpha_sweep_table(summaries), encoding='utf-8')
    return summaries


def run_threshold_figure() -> dict[str, Any]:
    run_dir = RUNS_DIR / 'synthetic_threshold_bins5'
    run_experiment(
        run_dir,
        [
            '--force',
            '--seeds', str(SEEDS[0]),
            '--methods', 'CFC,ICP',
            '--alpha', '0.10',
            '--n-cal', '10000',
            '--n-test', '10000',
            '--M', '50',
            '--cfc-basis-mode', 'group',
            '--cfc-basis-bins', '5',
        ],
    )
    return plot_threshold_adaptation(run_dir, alpha=0.10, bins=5, out_path=FIGS_DIR / 'threshold_vs_difficulty.png')


def run_ablations() -> dict[int, dict[tuple[int, int], dict[str, dict]]]:
    all_tables: dict[int, dict[tuple[int, int], dict[str, dict]]] = {}
    for bins in ABLATION_BINS:
        table_payload: dict[tuple[int, int], dict[str, dict]] = {}
        figure_payloads: list[dict[str, Any]] = []
        for n_cal in N_CALS:
            for m in MS:
                run_dir = RUNS_DIR / 'ablation' / f'bins_{bins}' / f'ncal_{n_cal}_M_{m}'
                summary = run_experiment(
                    run_dir,
                    [
                        '--seeds', ','.join(str(seed) for seed in SEEDS),
                        '--methods', 'CFC,CFC_PAC',
                        '--alpha', '0.20',
                        '--n-cal', str(n_cal),
                        '--n-test', '10000',
                        '--M', str(m),
                        '--delta', '0.90',
                        '--pac-mode', 'stability',
                        '--cfc-basis-mode', 'group',
                        '--cfc-basis-bins', str(bins),
                        '--eval-bins', str(bins),
                    ],
                )

                table_payload[(n_cal, m)] = {
                    'CFC': {
                        'true_cov': float(summary['CFC']['true_ecr_mean']),
                        'true_cov_std': float(summary['CFC']['true_ecr_std']),
                        'set_mean': float(summary['CFC']['apss_mean']),
                        'set_std': float(summary['CFC']['apss_std']),
                    },
                    'CFC_PAC': {
                        'true_cov': float(summary['CFC_PAC']['true_ecr_mean']),
                        'true_cov_std': float(summary['CFC_PAC']['true_ecr_std']),
                        'set_mean': float(summary['CFC_PAC']['apss_mean']),
                        'set_std': float(summary['CFC_PAC']['apss_std']),
                    },
                }
                figure_payloads.append(
                    {
                        'n_cal': n_cal,
                        'M': m,
                        'CFC': {
                            'true_group_coverages': recompute_true_group_coverages(run_dir, 'CFC', bins=bins),
                        },
                        'CFC_PAC': {
                            'true_group_coverages': recompute_true_group_coverages(run_dir, 'CFC_PAC', bins=bins),
                        },
                    }
                )

        all_tables[bins] = table_payload
        write_json(
            TABLES_DIR / f'ablation_bins{bins}.json',
            {f'{n_cal}_{m}': payload for (n_cal, m), payload in table_payload.items()},
        )
        (TABLES_DIR / f'ablation_bins{bins}.tex').write_text(build_ablation_table(table_payload, bins), encoding='utf-8')
        plot_ablation_group_miscoverage(figure_payloads, bins=bins, out_path=FIGS_DIR / f'ablation_group_bins{bins}.png')
    return all_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Reproduce the synthetic results and figures used in the paper.')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_BASE_DIR),
        help='Directory where synthetic runs, tables, figures, and the manifest are written.',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_output_root(Path(args.output_dir))
    ensure_dirs()
    compact_summary = run_main_table()
    alpha_sweep = run_alpha_sweep()
    threshold_info = run_threshold_figure()
    ablations = run_ablations()

    manifest = {
        'base_dir': str(BASE_DIR),
        'runs_dir': str(RUNS_DIR),
        'tables_dir': str(TABLES_DIR),
        'figs_dir': str(FIGS_DIR),
        'artifacts': {
            'compact_table_main': str(TABLES_DIR / 'synthetic_compact_main.tex'),
            'compact_table_appendix': str(TABLES_DIR / 'synthetic_compact_appendix.tex'),
            'alpha_sweep_table': str(TABLES_DIR / 'synthetic_alpha_sweep.tex'),
            'main_figure': str(FIGS_DIR / 'synfg1.pdf'),
            'threshold_figure': str(FIGS_DIR / 'threshold_vs_difficulty.png'),
            'ablation_figures': [str(FIGS_DIR / f'ablation_group_bins{bins}.png') for bins in ABLATION_BINS],
            'ablation_tables': [str(TABLES_DIR / f'ablation_bins{bins}.tex') for bins in ABLATION_BINS],
        },
        'summaries': {
            'compact': compact_summary,
            'alpha_sweep': {f'{alpha:.2f}': payload for alpha, payload in alpha_sweep.items()},
            'threshold': threshold_info,
            'ablation': {str(bins): {f'{n_cal}_{m}': payload for (n_cal, m), payload in table.items()} for bins, table in ablations.items()},
        },
    }
    write_json(BASE_DIR / 'manifest.json', manifest)
    print(json.dumps(manifest['artifacts'], indent=2))


if __name__ == '__main__':
    main()
