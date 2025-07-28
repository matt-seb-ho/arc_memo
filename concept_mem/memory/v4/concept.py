# concept_mem/concept.py
import itertools
import logging
import re
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# --------------------------- Utilities --------------------------------- #
_TYPE_DEF_RE = re.compile(r"^\s*([^:=\s]+)\s*:=\s*(.+)$")


def maybe_parse_typedef(s: str | None) -> tuple[str, str] | None:
    """
    If `s` matches the pattern 'Name := python_like_annotation', return (Name, annotation).
    Otherwise return None.
    """
    if not s:
        return None
    m = _TYPE_DEF_RE.match(s)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


# ------------------------ Data structures ------------------------------ #
@dataclass
class ParameterSpec:
    name: str
    typing: str | None = None
    description: str | None = None


@dataclass
class Concept:
    """
    Schema
    ------
    - name: str
    - kind: 'structure' | 'routine'
    - routine_subtype: str | None         # only for kind == 'routine'
    - output_typing: str | None
    - parameters: List[ParameterSpec]
    - description: str | None
    - cues: List[str]                      # how to detect this concept is relevant
    - implementation: List[str]            # coding hints / pseudocode lines
    - used_in: List[str]
    """

    name: str
    kind: str
    routine_subtype: str | None = None
    output_typing: str | None = None
    parameters: list[ParameterSpec] = field(default_factory=list)
    description: str | None = None
    cues: list[str] = field(default_factory=list)
    implementation: list[str] = field(default_factory=list)
    used_in: list[str] = field(default_factory=list)

    # ----------------------- Init / validation ------------------------- #
    def __post_init__(self):
        # loading from file leads to self.parameters being a list[dict]
        assert isinstance(self.parameters, list)
        fixed_params: list[ParameterSpec] = []
        for p in self.parameters:
            if isinstance(p, dict):
                p = ParameterSpec(**p)
            elif not isinstance(p, ParameterSpec):
                raise TypeError(f"Expected ParameterSpec or dict, got {type(p)}: {p}")
            fixed_params.append(p)
        self.parameters = fixed_params

    # ------------------------ Merge logic ------------------------------ #
    def update(self, problem_id: str, annotation: dict) -> None:
        if problem_id not in self.used_in:
            self.used_in.append(problem_id)

        self.description = self.description or annotation.get("description")
        self.output_typing = self.output_typing or annotation.get("output_typing")

        # routine_subtype only applies for routines; keep first non-null
        if self.kind == "routine":
            self.routine_subtype = self.routine_subtype or annotation.get(
                "routine_subtype"
            )

        # Merge parameters
        if "parameters" in annotation and annotation["parameters"]:
            merged = {p.name: p for p in self.parameters}
            for raw in annotation["parameters"]:
                if not isinstance(raw, dict):
                    raw = {"name": str(raw)}
                merged[raw["name"]] = ParameterSpec(
                    name=raw["name"],
                    typing=raw.get("typing"),
                    description=raw.get("description"),
                )
            self.parameters = list(merged.values())

        if "cues" in annotation and annotation["cues"]:
            self.cues = self._merge_lines(self.cues, annotation["cues"])
        if "implementation" in annotation and annotation["implementation"]:
            self.implementation = self._merge_lines(
                self.implementation, annotation["implementation"]
            )

    @staticmethod
    def _merge_lines(curr: list[str], new_lines: list[str]) -> list[str]:
        # merge cues & implementation (dedupe, keep order)
        cleaned_new_lines = []
        for line in new_lines:
            if isinstance(line, dict):
                if len(line) == 1:
                    k, v = next(iter(line.items()))
                    line = f"{k}: {v}"
                else:
                    logger.info(
                        f"merge list[str] expects a string but received a dict with multiple keys: {line}"
                    )
                    line = str(line)
            if isinstance(line, str):
                cleaned_new_lines.append(line.strip())
            else:
                logger.info(f"merge list[str] expects a string but received: {line}")

        return list(dict.fromkeys(itertools.chain(curr, cleaned_new_lines)))

    # --------------------- Rendering helpers --------------------------- #
    def to_string(
        self,
        *,
        include_description: bool = True,
        indentation: int = 0,
        # unified skip-* controls (all default False ⇒ show everything)
        skip_kind: bool = False,
        skip_routine_subtype: bool = False,
        skip_parameters: bool = False,
        skip_parameter_description: bool = False,
        skip_cues: bool = False,
        skip_implementation: bool = False,
    ) -> str:
        """
        Pretty-print this concept as a YAML-ish block.

        All *skip_* flags default to False → everything rendered unless skipped.
        """
        ind = " " * indentation
        lines: list[str] = [f"{ind}- concept: {self.name}"]

        if not skip_kind:
            lines.append(f"{ind}  kind: {self.kind}")

        if self.kind == "routine" and not skip_routine_subtype and self.routine_subtype:
            lines.append(f"{ind}  routine_subtype: {self.routine_subtype}")

        if include_description and self.description:
            lines.append(f"{ind}  description: {self.description}")

        if self.output_typing:
            lines.append(f"{ind}  output_typing: {self.output_typing}")

        # Parameters ----------------------------------------------------
        if self.parameters and not skip_parameters:
            lines.append(f"{ind}  parameters:")
            for p in self.parameters:
                line = f"{ind}    - {p.name}"
                if p.typing:
                    line += f" (type: {p.typing})"
                if p.description and not skip_parameter_description:
                    line += f": {p.description}"
                lines.append(line)

        # Cues & implementation ----------------------------------------
        if self.cues and not skip_cues:
            lines.append(f"{ind}  cues:")
            for c in self.cues:
                lines.append(f"{ind}    - {c}")

        if self.implementation and not skip_implementation:
            lines.append(f"{ind}  implementation:")
            for note in self.implementation:
                lines.append(f"{ind}    - {note}")

        return "\n".join(lines)

    # -------------------------- Misc ----------------------------------- #
    def asdict(self) -> dict:
        return asdict(self)
