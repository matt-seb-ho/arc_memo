# concept_mem/concept.py
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


@dataclass
class ParameterSpec:
    """One parameter accepted by a routine/structure."""

    name: str
    typing: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Concept:
    """
    Canonical representation stored in memory.

    ┌──────────┬─────────────────────────────────────────────┐
    │ Field     │ Meaning                                    │
    ├──────────┼─────────────────────────────────────────────┤
    │ name      │ Unique identifier (was `concept`)          │
    │ kind      │ 'structure' or 'routine'                   │
    │ description│ Natural-language explanation              │
    │ output    │ Typing of the value this concept returns   │
    │ parameters│ List[ParameterSpec]                        │
    │ implementation│ Notes / impl hints / pseudocode lines  │
    │ used_in   │ List of puzzle IDs where it occurred       │
    └──────────┴─────────────────────────────────────────────┘
    """

    name: str
    kind: str  # 'structure' | 'routine'
    description: Optional[str] = None
    output: Optional[str] = None
    parameters: List[ParameterSpec] = field(default_factory=list)
    implementation: List[str] = field(default_factory=list)
    used_in: List[str] = field(default_factory=list)

    # --------------------------------------------------------------------- #
    #  Update utilities                                                     #
    # --------------------------------------------------------------------- #

    def update(self, problem_id: str, annotation: Dict) -> None:
        """Merge new information that came from another puzzle."""
        if problem_id not in self.used_in:
            self.used_in.append(problem_id)

        # never overwrite an existing, richer description with a poorer one
        self.description = self.description or annotation.get("description")
        self.output = self.output or annotation.get("output")

        # merge parameters – keep the most recent version of each param name
        if "parameters" in annotation and annotation["parameters"]:
            merged = {p.name: p for p in self.parameters}
            for raw in annotation["parameters"]:
                merged[raw["name"]] = ParameterSpec(
                    name=raw["name"],
                    typing=raw.get("typing"),
                    description=raw.get("description"),
                )
            self.parameters = list(merged.values())

        # accumulate implementation notes (deduplicated, preserve order)
        if "implementation" in annotation and annotation["implementation"]:
            self.implementation = list(
                dict.fromkeys(
                    itertools.chain(self.implementation, annotation["implementation"])
                )
            )

    # --------------------------------------------------------------------- #
    #  Serialization helpers                                                #
    # --------------------------------------------------------------------- #

    def to_string(
        self, *, include_description: bool = True, indentation: int = 0
    ) -> str:
        """Pretty YAML-ish rendering – nice for prompts / debugging."""
        ind = " " * indentation
        out: list[str] = [
            f"{ind}- concept: {self.name}",
            f"{ind}  kind: {self.kind}",
        ]
        if include_description and self.description:
            out.append(f"{ind}  description: {self.description}")
        if self.output:
            out.append(f"{ind}  output: {self.output}")

        if self.parameters:
            out.append(f"{ind}  parameters:")
            for p in self.parameters:
                line = f"{ind}    - name: {p.name}"
                if p.typing:
                    line += f" | {p.typing}"
                if p.description:
                    line += f" - {p.description}"
                out.append(line)

        if self.implementation:
            out.append(f"{ind}  implementation:")
            for note in self.implementation:
                out.append(f"{ind}    - {note}")

        return "\n".join(out)

    # Convenience when persisting with `orjson`
    def asdict(self) -> Dict:
        return asdict(self)
