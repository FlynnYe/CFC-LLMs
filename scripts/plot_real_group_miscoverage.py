from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


DISPLAY_NAMES = {
    'TOPK': 'TopK',
    'ICP': 'ICP',
    'LEARNT_CP': 'Learnt CP',
    'CFC': 'CFC',
    'CFC_PAC_FULL': 'CFC-P-F',
}

COLORS = {
    'TOPK': '#4C72B0',
    'ICP': '#DD8452',
    'LEARNT_CP': '#8C8C8C',
    'CFC': '#55A868',
    'CFC_PAC_FULL': '#B07AA1',
}

METHODS = ['TOPK', 'ICP', 'LEARNT_CP', 'CFC', 'CFC_PAC_FULL']


def _trivia_panel(summary_path: Path, alpha_key: str) -> dict:
    data = json.loads(summary_path.read_text())
    return {
        'title': 'TriviaQA',
        'alpha': float(alpha_key),
        'group_labels': ['Easy', 'Hard'],
        'results': {
            name: {'group_coverages_per_seed': [entry['group_coverages'] for entry in data['results'][alpha_key][name]['per_seed']]}
            for name in METHODS
        },
    }


def _generic_panel(summary_path: Path, alpha_key: str, title: str) -> dict:
    data = json.loads(summary_path.read_text())
    return {
        'title': title,
        'alpha': float(alpha_key),
        'group_labels': ['1', '2', '3', '4', '5'],
        'results': {
            name: {'group_coverages_per_seed': [entry['group_coverages'] for entry in data['results'][alpha_key][name]['per_seed']]}
            for name in METHODS
        },
    }


def plot_panels(panels: dict[str, dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.2))
    width = 0.13

    for ax, panel_key in zip(axes, ['triviaqa', 'gsm8k', 'flickr8k']):
        panel = panels[panel_key]
        alpha = float(panel['alpha'])
        group_labels = panel['group_labels']
        groups = np.arange(1, len(group_labels) + 1)
        all_miscoverage = []
        for idx, method in enumerate(METHODS):
            payload = panel['results'][method]
            matrix = 100.0 * (1.0 - np.asarray(payload['group_coverages_per_seed'], dtype=float))
            all_miscoverage.append(matrix)
            mean_vals = np.nanmean(matrix, axis=0)
            std_vals = np.nanstd(matrix, axis=0)
            upper_only_std = np.vstack([np.zeros_like(std_vals), std_vals])
            offset = (idx - (len(METHODS) - 1) / 2.0) * width
            ax.bar(
                groups + offset,
                mean_vals,
                width=width,
                color=COLORS[method],
                edgecolor='black',
                linewidth=0.65,
                alpha=0.95,
                yerr=upper_only_std,
                error_kw={
                    'ecolor': '#333333',
                    'elinewidth': 0.9,
                    'capsize': 2.5,
                    'capthick': 0.9,
                },
                zorder=3,
            )

        ax.axhline(alpha * 100.0, color='black', linestyle='--', linewidth=1.5)
        ax.set_title(f"{panel['title']} ($\\alpha={alpha:.2f}$)", fontsize=14)
        ax.set_xlabel('Group', fontsize=13)
        ax.set_xticks(groups)
        ax.set_xticklabels(group_labels)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(axis='y', alpha=0.22, linewidth=0.7)
        ymax = max(alpha * 100.0 + 3.0, float(np.max(np.concatenate(all_miscoverage, axis=None))) + 3.0)
        ax.set_ylim(0.0, ymax)

    axes[0].set_ylabel('Miscoverage (%)', fontsize=13)
    handles = [Patch(facecolor=COLORS[name], edgecolor='black', label=DISPLAY_NAMES[name]) for name in METHODS]
    fig.legend(
        handles=handles,
        loc='upper center',
        ncol=5,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.04),
        columnspacing=1.2,
        handlelength=1.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate the real-data miscoverage figure from paper-setting summaries.')
    parser.add_argument('--triviaqa-summary', type=str, required=True)
    parser.add_argument('--gsm8k-summary', type=str, required=True)
    parser.add_argument('--flickr8k-summary', type=str, required=True)
    parser.add_argument('--triviaqa-alpha', type=str, default='0.25')
    parser.add_argument('--gsm8k-alpha', type=str, default='0.10')
    parser.add_argument('--flickr8k-alpha', type=str, default='0.03')
    parser.add_argument('--output-figure', type=str, required=True)
    parser.add_argument('--output-json', type=str, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    panels = {
        'triviaqa': _trivia_panel(Path(args.triviaqa_summary), args.triviaqa_alpha),
        'gsm8k': _generic_panel(Path(args.gsm8k_summary), args.gsm8k_alpha, 'GSM8K'),
        'flickr8k': _generic_panel(Path(args.flickr8k_summary), args.flickr8k_alpha, 'Flickr8k'),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(panels, indent=2), encoding='utf-8')
    plot_panels(panels, Path(args.output_figure))
    print(json.dumps({'figure': args.output_figure, 'data': args.output_json}, indent=2))


if __name__ == '__main__':
    main()
