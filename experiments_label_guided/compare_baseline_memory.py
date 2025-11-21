#!/usr/bin/env env python3
"""
Compare baseline vs with-memory performance for AIME validation runs.

Analyzes:
- Problems that flipped from wrong to right (memory helped)
- Problems that flipped from right to wrong (memory hurt)
- Problems that stayed correct
- Problems that stayed incorrect

Outputs detailed comparison with problem text, lessons used, and answers.

Usage:
  # Auto-detect and compare all baseline/memory pairs in a directory
  python compare_baseline_memory.py experiments_label_guided/aime_val
  
  # Compare specific runs
  python compare_baseline_memory.py baseline_dir memory_dir [output_file]
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_solutions(solutions_path: Path) -> Dict[str, str]:
    """Load solutions.json file."""
    with open(solutions_path) as f:
        return json.load(f)


def load_ground_truth(gt_path: Path) -> Dict[str, str]:
    """Load ground truth answers from validation.json."""
    with open(gt_path) as f:
        data = json.load(f)
    return {p['id']: str(p['answer']).strip() for p in data['problems']}


def load_problem_texts(gt_path: Path) -> Dict[str, str]:
    """Load problem text from validation.json."""
    with open(gt_path) as f:
        data = json.load(f)
    return {p['id']: p['problem'] for p in data['problems']}


def load_prompt_info(memory_dir: Path) -> Dict[str, Dict]:
    """
    Load prompt_info.json with retrieved lessons.
    
    Checks:
    1. memory_dir/prompt_info.json (if selection was done in same dir)
    2. memory_dir/.hydra/config.yaml -> prompt.problem_data path
    """
    # Try direct path first
    direct_path = memory_dir / 'prompt_info.json'
    if direct_path.exists():
        with open(direct_path) as f:
            return json.load(f)
    
    # Try loading from config
    config_path = memory_dir / '.hydra/config.yaml'
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        problem_data_path = cfg.get('prompt', {}).get('problem_data')
        if problem_data_path:
            # Make path absolute relative to memory_dir's parent
            if not Path(problem_data_path).is_absolute():
                # Try relative to repo root
                for root_candidate in [memory_dir.parent.parent.parent, Path.cwd()]:
                    candidate = root_candidate / problem_data_path
                    if candidate.exists():
                        with open(candidate) as f:
                            return json.load(f)
    
    return {}


def find_run_pairs(parent_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Auto-detect baseline/memory pairs in a directory.
    
    Strategy:
    1. Try exact timestamp match first
    2. If no exact matches, pair latest baseline with latest memory
    """
    baseline_dirs = sorted(parent_dir.glob('baseline_*'), key=lambda p: p.name)
    memory_dirs = sorted(parent_dir.glob('with_memory_*'), key=lambda p: p.name)
    
    if not baseline_dirs or not memory_dirs:
        return []
    
    pairs = []
    
    # Try to match by exact timestamp
    for baseline_dir in baseline_dirs:
        # Extract timestamp from baseline_YYYYMMDD-HHMMSS
        match = re.search(r'(\d{8}-\d{6})', baseline_dir.name)
        if not match:
            continue
        timestamp = match.group(1)
        
        # Find matching memory run
        matching_memory = None
        for memory_dir in memory_dirs:
            if timestamp in memory_dir.name:
                matching_memory = memory_dir
                break
        
        if matching_memory:
            pairs.append((baseline_dir, matching_memory))
    
    # If no exact matches, pair latest baseline with latest memory
    if not pairs and baseline_dirs and memory_dirs:
        latest_baseline = baseline_dirs[-1]
        latest_memory = memory_dirs[-1]
        pairs.append((latest_baseline, latest_memory))
    
    return pairs


def is_correct(solution: str, ground_truth: str) -> bool:
    """Check if solution matches ground truth."""
    return solution.strip() == ground_truth.strip() if solution else False


def extract_answer(response: str) -> str:
    """Extract answer from response using same logic as solver."""
    if not response or not response.strip():
        return ''
    
    response = response.strip()
    
    # Prefer boxed answer
    boxed = re.findall(r'\\boxed\{(\d{1,3})\}', response)
    if boxed:
        return boxed[-1]
    
    # Fallback patterns
    match = re.search(r'[Ff]inal\s+[Aa]nswer\s*:\s*(\d+)', response)
    if match:
        return match.group(1)
    
    match = re.search(r'[Aa]nswer\s*:\s*(\d+)', response)
    if match:
        return match.group(1)
    
    match = re.search(r'[Tt]he\s+answer\s+is\s+(\d+)', response)
    if match:
        return match.group(1)
    
    # Last resort: find 1-3 digit number in last 500 chars
    numbers = re.findall(r'\b(\d{1,3})\b', response[-500:])
    for num_str in reversed(numbers):
        num = int(num_str)
        if 0 <= num <= 999:
            return num_str
    
    return ''


def check_pass_at_k(responses: List[str], ground_truth: str, k: int) -> bool:
    """Check if any of the first k responses is correct."""
    for resp in responses[:k]:
        ans = extract_answer(resp)
        if ans and ans.strip() == ground_truth.strip():
            return True
    return False


def compare_runs(
    baseline_dir: Path,
    memory_dir: Path,
    gt_path: Path,
    output_path: Path
) -> None:
    """Compare baseline and with-memory runs."""
    
    # Load data
    baseline_sols = load_solutions(baseline_dir / 'solutions.json')
    memory_sols = load_solutions(memory_dir / 'solutions.json')
    ground_truth = load_ground_truth(gt_path)
    problem_texts = load_problem_texts(gt_path)
    
    # Load full responses for pass@2
    baseline_responses_path = baseline_dir / 'full_responses.json'
    memory_responses_path = memory_dir / 'full_responses.json'
    
    baseline_responses = {}
    memory_responses = {}
    if baseline_responses_path.exists():
        with open(baseline_responses_path) as f:
            baseline_responses = json.load(f)
    if memory_responses_path.exists():
        with open(memory_responses_path) as f:
            memory_responses = json.load(f)
    
    # Load lessons used (from prompt_info.json, following config if needed)
    prompt_info = load_prompt_info(memory_dir)
    
    # Categorize problems (pass@1)
    flipped_right = []  # wrong -> right
    flipped_wrong = []  # right -> wrong
    stayed_right = []
    stayed_wrong = []
    
    # Categorize problems (pass@2)
    flipped_right_p2 = []
    flipped_wrong_p2 = []
    stayed_right_p2 = []
    stayed_wrong_p2 = []
    
    for pid in ground_truth:
        gt = ground_truth[pid]
        baseline_correct = is_correct(baseline_sols.get(pid, ''), gt)
        memory_correct = is_correct(memory_sols.get(pid, ''), gt)
        
        # Check pass@2 if responses available
        baseline_correct_p2 = baseline_correct
        memory_correct_p2 = memory_correct
        if pid in baseline_responses and baseline_responses[pid]:
            baseline_correct_p2 = check_pass_at_k(baseline_responses[pid], gt, 2)
        if pid in memory_responses and memory_responses[pid]:
            memory_correct_p2 = check_pass_at_k(memory_responses[pid], gt, 2)
        
        entry = {
            'problem_id': pid,
            'baseline_answer': baseline_sols.get(pid, ''),
            'memory_answer': memory_sols.get(pid, ''),
            'ground_truth': gt,
            'problem_text': problem_texts.get(pid, ''),
        }
        
        # Extract lessons used
        if pid in prompt_info:
            # prompt_info structure: {pid: {variant_key: {hint: ..., description: ...}}}
            variant_data = prompt_info[pid]
            if isinstance(variant_data, dict):
                # Get first variant (usually 'aime_lessons' or similar)
                first_variant = next(iter(variant_data.values())) if variant_data else {}
                if isinstance(first_variant, dict):
                    entry['lessons_used'] = first_variant.get('hint', '') or first_variant.get('lessons', '')
                else:
                    entry['lessons_used'] = str(first_variant)
            else:
                entry['lessons_used'] = str(variant_data)
        else:
            entry['lessons_used'] = '(no lessons retrieved)'
        
        # Categorize for pass@1
        if not baseline_correct and memory_correct:
            flipped_right.append(entry)
        elif baseline_correct and not memory_correct:
            flipped_wrong.append(entry)
        elif baseline_correct and memory_correct:
            stayed_right.append(entry)
        else:
            stayed_wrong.append(entry)
        
        # Categorize for pass@2
        if not baseline_correct_p2 and memory_correct_p2:
            flipped_right_p2.append(entry)
        elif baseline_correct_p2 and not memory_correct_p2:
            flipped_wrong_p2.append(entry)
        elif baseline_correct_p2 and memory_correct_p2:
            stayed_right_p2.append(entry)
        else:
            stayed_wrong_p2.append(entry)
    
    # Compute summary statistics
    total = len(ground_truth)
    baseline_correct_p1 = sum(1 for pid in ground_truth 
                              if is_correct(baseline_sols.get(pid, ''), ground_truth[pid]))
    memory_correct_p1 = sum(1 for pid in ground_truth 
                            if is_correct(memory_sols.get(pid, ''), ground_truth[pid]))
    
    baseline_correct_p2 = len(stayed_right_p2) + len(flipped_wrong_p2)
    memory_correct_p2 = len(stayed_right_p2) + len(flipped_right_p2)
    
    summary = {
        'total_problems': total,
        'pass@1': {
            'baseline_correct': baseline_correct_p1,
            'memory_correct': memory_correct_p1,
            'flipped_right_count': len(flipped_right),
            'flipped_wrong_count': len(flipped_wrong),
            'stayed_right_count': len(stayed_right),
            'stayed_wrong_count': len(stayed_wrong),
            'net_improvement': len(flipped_right) - len(flipped_wrong),
        },
        'pass@2': {
            'baseline_correct': baseline_correct_p2,
            'memory_correct': memory_correct_p2,
            'flipped_right_count': len(flipped_right_p2),
            'flipped_wrong_count': len(flipped_wrong_p2),
            'stayed_right_count': len(stayed_right_p2),
            'stayed_wrong_count': len(stayed_wrong_p2),
            'net_improvement': len(flipped_right_p2) - len(flipped_wrong_p2),
        }
    }
    
    # Build detailed output
    output = {
        'summary': summary,
        'pass@1': {
            'flipped_right': flipped_right,
            'flipped_wrong': flipped_wrong,
            'stayed_right_ids': [e['problem_id'] for e in stayed_right],
            'stayed_wrong_ids': [e['problem_id'] for e in stayed_wrong],
        },
        'pass@2': {
            'flipped_right': flipped_right_p2,
            'flipped_wrong': flipped_wrong_p2,
            'stayed_right_ids': [e['problem_id'] for e in stayed_right_p2],
            'stayed_wrong_ids': [e['problem_id'] for e in stayed_wrong_p2],
        }
    }
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("BASELINE vs WITH-MEMORY COMPARISON")
    print(f"{'='*70}")
    print(f"Total problems: {total}")
    
    print(f"\n--- pass@1 ---")
    p1 = summary['pass@1']
    print(f"Baseline correct: {p1['baseline_correct']}/{total} ({p1['baseline_correct']/total*100:.1f}%)")
    print(f"With memory correct: {p1['memory_correct']}/{total} ({p1['memory_correct']/total*100:.1f}%)")
    print(f"Flipped RIGHT (wrong → right): {p1['flipped_right_count']}")
    print(f"Flipped WRONG (right → wrong): {p1['flipped_wrong_count']}")
    print(f"Stayed right: {p1['stayed_right_count']}")
    print(f"Stayed wrong: {p1['stayed_wrong_count']}")
    print(f"Net improvement: {p1['net_improvement']:+d}")
    
    print(f"\n--- pass@2 ---")
    p2 = summary['pass@2']
    print(f"Baseline correct: {p2['baseline_correct']}/{total} ({p2['baseline_correct']/total*100:.1f}%)")
    print(f"With memory correct: {p2['memory_correct']}/{total} ({p2['memory_correct']/total*100:.1f}%)")
    print(f"Flipped RIGHT (wrong → right): {p2['flipped_right_count']}")
    print(f"Flipped WRONG (right → wrong): {p2['flipped_wrong_count']}")
    print(f"Stayed right: {p2['stayed_right_count']}")
    print(f"Stayed wrong: {p2['stayed_wrong_count']}")
    print(f"Net improvement: {p2['net_improvement']:+d}")
    
    print(f"\n✓ Detailed comparison saved to: {output_path}")
    print(f"{'='*70}\n")


def main():
    # Default ground truth path (relative to script location)
    script_dir = Path(__file__).parent
    gt_path = script_dir.parent / 'data/aime/validation.json'
    if not gt_path.exists():
        # Try from repo root
        gt_path = Path('arc_memo/data/aime/validation.json')
    if not gt_path.exists():
        gt_path = Path('data/aime/validation.json')
    
    if not gt_path.exists():
        print(f"Error: Ground truth file not found: {gt_path}")
        sys.exit(1)
    
    # Mode 0: No args - auto-detect in current directory's aime_val
    if len(sys.argv) == 1:
        # Assume script is in experiments_label_guided/ or experiments_self_reflective/
        parent_dir = script_dir / 'aime_val'
        if not parent_dir.exists():
            print(f"Error: {parent_dir} not found")
            print("\nUsage:")
            print("  # Auto-detect pairs in current directory:")
            print("  python compare_baseline_memory.py")
            print("\n  # Auto-detect pairs in specified directory:")
            print("  python compare_baseline_memory.py <parent_dir>")
            print("\n  # Compare specific runs:")
            print("  python compare_baseline_memory.py <baseline_dir> <memory_dir> [output_file]")
            sys.exit(1)
        
        pairs = find_run_pairs(parent_dir)
        if not pairs:
            print(f"No baseline/memory pairs found in {parent_dir}")
            print("Looking for: baseline_* and with_memory_* directories with matching timestamps")
            sys.exit(1)
        
        print(f"Found {len(pairs)} baseline/memory pair(s) in {parent_dir}:\n")
        for i, (baseline_dir, memory_dir) in enumerate(pairs, 1):
            print(f"Pair {i}:")
            print(f"  Baseline: {baseline_dir.name}")
            print(f"  Memory:   {memory_dir.name}")
            timestamp = re.search(r'(\d{8}-\d{6})', baseline_dir.name)
            ts_str = timestamp.group(1) if timestamp else f'pair{i}'
            output_file = script_dir / f'comparison_{ts_str}.json'
            compare_runs(baseline_dir, memory_dir, gt_path, output_file)
        return
    
    arg1 = Path(sys.argv[1])
    
    # Mode 1: Auto-detect pairs in specified parent directory
    if len(sys.argv) == 2 and arg1.is_dir():
        pairs = find_run_pairs(arg1)
        if not pairs:
            print(f"No baseline/memory pairs found in {arg1}")
            print("Looking for: baseline_* and with_memory_* directories with matching timestamps")
            sys.exit(1)
        
        print(f"Found {len(pairs)} baseline/memory pair(s):\n")
        for i, (baseline_dir, memory_dir) in enumerate(pairs, 1):
            print(f"Pair {i}:")
            print(f"  Baseline: {baseline_dir.name}")
            print(f"  Memory:   {memory_dir.name}")
            timestamp = re.search(r'(\d{8}-\d{6})', baseline_dir.name)
            ts_str = timestamp.group(1) if timestamp else f'pair{i}'
            output_file = arg1 / f'comparison_{ts_str}.json'
            compare_runs(baseline_dir, memory_dir, gt_path, output_file)
        return
    
    # Mode 2: Compare specific runs
    if len(sys.argv) < 3:
        print("Error: Need both baseline_dir and memory_dir")
        sys.exit(1)
    
    baseline_dir = Path(sys.argv[1])
    memory_dir = Path(sys.argv[2])
    output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('comparison.json')
    
    if not baseline_dir.exists():
        print(f"Error: Baseline directory not found: {baseline_dir}")
        sys.exit(1)
    
    if not memory_dir.exists():
        print(f"Error: Memory directory not found: {memory_dir}")
        sys.exit(1)
    
    compare_runs(baseline_dir, memory_dir, gt_path, output_file)


if __name__ == '__main__':
    main()

