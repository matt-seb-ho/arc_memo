#!/usr/bin/env python3
"""
Analyze token usage and costs across all AIME experiments

Usage:
    cd arc_memo/experiments
    python analyze_costs.py
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

# Pricing (as of Nov 2024, adjust as needed)
# Format: {model_name: (input_price_per_1M, output_price_per_1M)}
MODEL_PRICING = {
    "o4-mini-2025-04-16": (1.10, 4.40),  # $1.10 input, $4.40 output per 1M tokens
    "gpt-4.1-2025-04-14": (5.00, 15.00),
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "gpt-4o-2025-05-13": (2.50, 10.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
}


def find_token_usage_files(base_dir: Path) -> List[Tuple[str, Path]]:
    """Find all token_usage.json files in experiment folders"""
    files = []
    for exp_dir in base_dir.iterdir():
        if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
            # Search recursively for token_usage.json
            for token_file in exp_dir.rglob('token_usage.json'):
                rel_path = token_file.relative_to(base_dir)
                files.append((str(rel_path.parent), token_file))
    return sorted(files)


def analyze_token_usage(token_file: Path) -> Dict:
    """Analyze a single token_usage.json file"""
    with open(token_file) as f:
        data = json.load(f)
    
    # Get 'after' stats (ignore 'before')
    after = data.get('after', {})
    
    results = {}
    for model_name, stats in after.items():
        input_tokens = stats.get('input_tokens', 0)
        output_tokens = stats.get('output_tokens', 0)
        reasoning_tokens = stats.get('reasoning_tokens', 0)
        requests = stats.get('requests', 0)
        completions = stats.get('completions', 0)
        
        # Calculate cost
        if model_name in MODEL_PRICING:
            input_price, output_price = MODEL_PRICING[model_name]
            input_cost = (input_tokens / 1_000_000) * input_price
            output_cost = (output_tokens / 1_000_000) * output_price
            total_cost = input_cost + output_cost
        else:
            input_cost = output_cost = total_cost = 0.0
        
        # Visible output tokens (excluding reasoning)
        visible_tokens = output_tokens - reasoning_tokens
        
        results[model_name] = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'reasoning_tokens': reasoning_tokens,
            'visible_tokens': visible_tokens,
            'requests': requests,
            'completions': completions,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
        }
    
    return results


def print_experiment_summary(exp_name: str, analysis: Dict):
    """Print summary for one experiment"""
    print(f"\n{exp_name}")
    print("  " + "=" * 70)
    
    total_cost = 0.0
    for model_name, stats in analysis.items():
        print(f"  Model: {model_name}")
        print(f"    Tokens: {stats['input_tokens']:,} in | {stats['output_tokens']:,} out")
        if stats['reasoning_tokens'] > 0:
            print(f"            ({stats['reasoning_tokens']:,} reasoning + {stats['visible_tokens']:,} visible)")
        print(f"    Requests: {stats['requests']} | Completions: {stats['completions']}")
        if stats['total_cost'] > 0:
            print(f"    Cost: ${stats['input_cost']:.3f} + ${stats['output_cost']:.3f} = ${stats['total_cost']:.3f}")
        else:
            print(f"    Cost: Unknown pricing")
        total_cost += stats['total_cost']
    
    if len(analysis) > 1:
        print(f"  TOTAL COST: ${total_cost:.3f}")


def main():
    experiments_dir = Path(__file__).parent
    
    print("=" * 80)
    print("AIME Experiment Cost Analysis")
    print("=" * 80)
    print(f"\nScanning: {experiments_dir}")
    
    # Find all token usage files
    token_files = find_token_usage_files(experiments_dir)
    
    if not token_files:
        print("\nNo token_usage.json files found!")
        return
    
    print(f"Found {len(token_files)} experiments with token usage data\n")
    
    # Analyze each
    all_analyses = {}
    grand_total_cost = 0.0
    
    for exp_name, token_file in token_files:
        try:
            analysis = analyze_token_usage(token_file)
            all_analyses[exp_name] = analysis
            print_experiment_summary(exp_name, analysis)
            
            # Add to grand total
            for model_stats in analysis.values():
                grand_total_cost += model_stats['total_cost']
        except Exception as e:
            print(f"\n{exp_name}")
            print(f"  ERROR: {e}")
    
    # Grand summary
    print("\n" + "=" * 80)
    print("GRAND TOTAL")
    print("=" * 80)
    
    # Aggregate by model
    model_totals = {}
    for exp_name, analysis in all_analyses.items():
        for model_name, stats in analysis.items():
            if model_name not in model_totals:
                model_totals[model_name] = {
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'reasoning_tokens': 0,
                    'requests': 0,
                    'completions': 0,
                    'total_cost': 0.0,
                }
            model_totals[model_name]['input_tokens'] += stats['input_tokens']
            model_totals[model_name]['output_tokens'] += stats['output_tokens']
            model_totals[model_name]['reasoning_tokens'] += stats['reasoning_tokens']
            model_totals[model_name]['requests'] += stats['requests']
            model_totals[model_name]['completions'] += stats['completions']
            model_totals[model_name]['total_cost'] += stats['total_cost']
    
    for model_name, totals in model_totals.items():
        print(f"\n{model_name}:")
        print(f"  Total tokens: {totals['input_tokens']:,} in | {totals['output_tokens']:,} out")
        if totals['reasoning_tokens'] > 0:
            print(f"                ({totals['reasoning_tokens']:,} reasoning)")
        print(f"  Total requests: {totals['requests']}")
        print(f"  Total cost: ${totals['total_cost']:.2f}")
    
    print(f"\n{'='*80}")
    print(f"GRAND TOTAL COST: ${grand_total_cost:.2f}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

