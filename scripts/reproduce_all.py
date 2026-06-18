from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRIVIAQA_FIGURE_ALPHA = '0.25'
GSM8K_FIGURE_ALPHA = '0.10'
FLICKR8K_FIGURE_ALPHA = '0.03'


def run(cmd: list[str]) -> None:
    print('$', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    default_data_root = ROOT / 'data' / 'raw'
    parser = argparse.ArgumentParser(description='Reproduce the paper results from the public release code.')
    parser.add_argument('--output-root', type=str, default=str(ROOT / 'outputs'))
    parser.add_argument('--triviaqa-base-dir', type=str, default=str(default_data_root / 'triviaqa'))
    parser.add_argument('--triviaqa-model-glob', type=str, default='meta-llama__Llama-2-13b-hf')
    parser.add_argument('--gsm8k-score-dir', type=str, default=str(default_data_root / 'gsm8k'))
    parser.add_argument('--gsm8k-score-glob', type=str, default='gsm8k_with_rm_*.jsonl')
    parser.add_argument('--flickr-coverage-cache', type=str, default=str(default_data_root / 'flickr8k' / 'cached_coverage.json'))
    parser.add_argument('--flickr-candidate-cache', type=str, default=str(default_data_root / 'flickr8k' / 'cached_candidates.json'))
    parser.add_argument('--skip-synthetic', action='store_true')
    parser.add_argument('--skip-triviaqa', action='store_true')
    parser.add_argument('--skip-gsm8k', action='store_true')
    parser.add_argument('--skip-flickr8k', action='store_true')
    parser.add_argument('--skip-figure4', action='store_true')
    parser.add_argument('--verify', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    synthetic_dir = output_root / 'synthetic'
    trivia_dir = output_root / 'triviaqa' / 'paper_setting'
    gsm_dir = output_root / 'gsm8k' / 'paper_setting'
    flickr_dir = output_root / 'flick8k' / 'paper_setting'
    paper_dir = output_root / 'paper'

    if not args.skip_synthetic:
        run(
            [
                sys.executable,
                str(ROOT / 'synthetic' / 'reproduce_paper_synthetic.py'),
                '--output-dir',
                str(synthetic_dir),
            ]
        )

    if not args.skip_triviaqa:
        run(
            [
                sys.executable,
                str(ROOT / 'triviaqa' / 'evaluate_paper_setting.py'),
                '--base-dir',
                args.triviaqa_base_dir,
                '--model-glob',
                args.triviaqa_model_glob,
                '--run-dir',
                str(trivia_dir),
                '--cache-dir',
                str(output_root / 'triviaqa' / 'cache'),
            ]
        )

    if not args.skip_gsm8k:
        run(
            [
                sys.executable,
                str(ROOT / 'gsm8k' / 'evaluate_paper_setting.py'),
                '--score-dir',
                args.gsm8k_score_dir,
                '--score-glob',
                args.gsm8k_score_glob,
                '--run-dir',
                str(gsm_dir),
                '--cache-dir',
                str(output_root / 'gsm8k' / 'cache'),
            ]
        )

    if not args.skip_flickr8k:
        run(
            [
                sys.executable,
                str(ROOT / 'flick8k' / 'evaluate_paper_setting.py'),
                '--coverage-cache',
                args.flickr_coverage_cache,
                '--candidate-cache',
                args.flickr_candidate_cache,
                '--run-dir',
                str(flickr_dir),
                '--cache-dir',
                str(output_root / 'flick8k' / 'cache'),
            ]
        )

    if not args.skip_figure4:
        run(
            [
                sys.executable,
                str(ROOT / 'scripts' / 'plot_real_group_miscoverage.py'),
                '--triviaqa-summary',
                str(trivia_dir / 'summary.json'),
                '--gsm8k-summary',
                str(gsm_dir / 'summary.json'),
                '--flickr8k-summary',
                str(flickr_dir / 'summary.json'),
                '--triviaqa-alpha',
                TRIVIAQA_FIGURE_ALPHA,
                '--gsm8k-alpha',
                GSM8K_FIGURE_ALPHA,
                '--flickr8k-alpha',
                FLICKR8K_FIGURE_ALPHA,
                '--output-figure',
                str(paper_dir / 'real_group_miscoverage.pdf'),
                '--output-json',
                str(paper_dir / 'real_group_miscoverage.json'),
            ]
        )

    run(
        [
            sys.executable,
            str(ROOT / 'scripts' / 'build_release_results.py'),
            '--synthetic-manifest',
            str(synthetic_dir / 'manifest.json'),
            '--triviaqa-summary',
            str(trivia_dir / 'summary.json'),
            '--gsm8k-summary',
            str(gsm_dir / 'summary.json'),
            '--flickr8k-summary',
            str(flickr_dir / 'summary.json'),
            '--figure4-json',
            str(paper_dir / 'real_group_miscoverage.json'),
            '--output',
            str(output_root / 'paper_results.json'),
        ]
    )

    if args.verify:
        run(
            [
                sys.executable,
                str(ROOT / 'scripts' / 'verify_release.py'),
                '--expected',
                str(ROOT / 'expected' / 'paper_results.json'),
                '--observed',
                str(output_root / 'paper_results.json'),
            ]
        )


if __name__ == '__main__':
    main()
