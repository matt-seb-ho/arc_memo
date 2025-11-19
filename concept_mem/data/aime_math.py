"""AIME Math dataset loader for ArcMemo"""

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional


@dataclass
class MathProblem:
    """Mathematical problem structure"""
    problem_id: str
    problem_text: str
    solution: str
    answer: str
    difficulty: Optional[int] = None
    year: Optional[int] = None


def load_aime_data(
    split: str = "train",
    data_dir: Optional[Path] = None,
    num_problems: Optional[int] = None
) -> dict[str, MathProblem]:
    """
    Load AIME mathematical problems
    
    Expected format:
    {
        "problems": [
            {
                "id": "2020_I_1",
                "year": 2020,
                "problem_number": 1,
                "problem": "Problem text...",
                "answer": "42",
                "part": "I"
            },
            ...
        ]
    }
    """
    if data_dir is None:
        # Default location
        data_dir = Path(__file__).parent.parent.parent / "data" / "aime"
    
    data_path = data_dir / f"{split}.json"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"AIME data not found at {data_path}. "
            "Please run: cd data/aime && python download_and_prepare.py"
        )
    
    with open(data_path) as f:
        data = json.load(f)
    
    problems_list = data.get("problems", data) if isinstance(data, dict) else data
    
    problems = {}
    for i, p in enumerate(problems_list):
        if num_problems and i >= num_problems:
            break
        
        pid = p.get("id", f"aime_{i:04d}")
        problems[pid] = MathProblem(
            problem_id=pid,
            problem_text=p.get("problem", ""),
            solution=p.get("solution", ""),
            answer=str(p.get("answer", "")),
            difficulty=p.get("problem_number"),  # Use problem number as difficulty
            year=p.get("year")
        )
    
    return problems

