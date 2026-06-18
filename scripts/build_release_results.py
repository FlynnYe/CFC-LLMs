from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any



def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _normalize(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_normalize(value) for value in obj]
    if isinstance(obj, str) and obj.startswith('/'):
        p = Path(obj)
        parts = p.parts
        if len(parts) >= 2:
            return '/'.join(parts[-2:])
        return p.name
    return obj

def build_payload(
    synthetic_manifest: Path,
    trivia_summary: Path,
    gsm8k_summary: Path,
    flickr8k_summary: Path,
    figure4_json: Path,
) -> dict:
    synthetic = json.loads(synthetic_manifest.read_text())
    triviaqa = json.loads(trivia_summary.read_text())
    gsm8k = json.loads(gsm8k_summary.read_text())
    flickr8k = json.loads(flickr8k_summary.read_text())
    figure4 = json.loads(figure4_json.read_text())
    return _normalize({
        'synthetic': synthetic['summaries'],
        'triviaqa': triviaqa,
        'gsm8k': {
            'results': gsm8k['results'],
            'n_ablation_alpha_0.10': gsm8k['n_ablation_alpha_0.10'],
        },
        'flickr8k': {
            'results': flickr8k['results'],
            'ablation_alpha_0.03': flickr8k['ablation_alpha_0.03'],
        },
        'figure4': figure4,
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build a normalized, machine-independent snapshot of all release results.')
    parser.add_argument('--synthetic-manifest', type=str, required=True)
    parser.add_argument('--triviaqa-summary', type=str, required=True)
    parser.add_argument('--gsm8k-summary', type=str, required=True)
    parser.add_argument('--flickr8k-summary', type=str, required=True)
    parser.add_argument('--figure4-json', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_payload(
        synthetic_manifest=Path(args.synthetic_manifest),
        trivia_summary=Path(args.triviaqa_summary),
        gsm8k_summary=Path(args.gsm8k_summary),
        flickr8k_summary=Path(args.flickr8k_summary),
        figure4_json=Path(args.figure4_json),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out_path)}, indent=2))


if __name__ == '__main__':
    main()
