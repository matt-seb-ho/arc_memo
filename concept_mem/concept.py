import logging
import textwrap
from dataclasses import dataclass, field
from typing import ClassVar

logger = logging.getLogger(__name__)


# new format:
@dataclass
class Concept:
    # main identifier and most concise representation
    name: str
    # optional elaboration
    description: str | None = None

    # list of parent concept names
    parents: list[str] = field(default_factory=list)
    # map of associated concept name -> relationship description
    associated_concepts: dict[str, str] = field(default_factory=dict)

    # cues to look out for to identify the concept in problems
    relevance_cues: list[str] = field(default_factory=list)

    # miscellaneous annotations
    notes: list[str] = field(default_factory=dict)

    # tracking problem instances (IDs) where this concept is used
    usage: list[str] = field(default_factory=list)

    LIST_FIELDS: ClassVar[list[str]] = [
        "relevance_cues",
        "notes",
        "parents",
    ]
    DICT_FIELDS: ClassVar[list[str]] = [
        "associated_concepts",
    ]
    NON_EDITABLE_FIELDS: ClassVar[list[str]] = [
        "name",
        "description",
    ]

    # TODO:
    # python helper functions that operationalise the concept
    # helper_routines: list[str] = field(default_factory=list)

    def __post_init__(self):
        for field_name in self.LIST_FIELDS:
            seen_set_field_name = f"_seen_set_{field_name}"
            if not hasattr(self, seen_set_field_name):
                setattr(self, seen_set_field_name, set())

    def update(self, problem_id: str, annotation: dict) -> None:
        """
        Update the concept with new information from an annotation.
        """
        # update usage
        self.usage.append(problem_id)

        # can add a description iff it is not already set
        if "description" in annotation:
            if self.description is None:
                self.description = annotation["description"].strip()
            else:
                logger.info(
                    f"Description already set for {self.name}, skipping update."
                )

        for field_name in self.LIST_FIELDS:
            if field_name not in annotation:
                continue
            self._update_list_field(field_name, annotation[field_name])
        for field_name in self.DICT_FIELDS:
            if field_name not in annotation:
                continue
            self._update_dict_field(field_name, annotation[field_name])

    def to_string(
        self,
        include_description: bool = True,
        include_parents: bool = True,
        include_associates: bool = True,
        include_cues: bool = True,
        include_notes: bool = True,
        problem_usage_info: dict[str, str] | None = None,
        indentation: int = 0,
    ) -> str:
        components: list[str] = [f"- concept: {self.name}"]
        if include_description and self.description:
            components.append(f"  description: {self.description}")
        if include_parents and self.parents:
            components.append(f"  parents: {self.parents}")
        if include_associates and self.associated_concepts:
            components.append("  associated concepts:")
            for concept, description in self.associated_concepts.items():
                components.append(f"  - {concept}: {description}")
        if include_cues:
            components.append(self._format_list_field("relevance_cues"))
        if include_notes:
            components.append(self._format_list_field("notes"))
        if problem_usage_info:
            usage_components = []
            for problem_id in self.usage:
                if problem_id not in problem_usage_info:
                    continue
                usage_components.append(
                    f"  {problem_id}: {problem_usage_info[problem_id]}"
                )
            if usage_components:
                usage_list = "\n".join(usage_components)
                components.append(f"  usage:\n{usage_list}")
        result = "\n".join([line for line in components if line is not None])
        if indentation:
            result = textwrap.indent(result, " " * indentation)
        return result

    def _format_list_field(self, field_name: str) -> str | None:
        """
        Format a list field as a YAML block.
        """
        items = getattr(self, field_name)
        if not items:
            return None
        formatted_list = "\n".join(f"  - {item}" for item in items)
        return f"{field_name}:\n{formatted_list}"

    def _update_list_field(
        self,
        field_name: str,
        new_items: list[str],
    ) -> None:
        if not isinstance(new_items, list):
            logger.error(f"Expected list for {field_name}, got {type(new_items)}")
            return
        for item in new_items:
            if not isinstance(item, str):
                logger.info(f"Expected string for {field_name}, got {type(item)}")
                try:
                    item = str(item)
                except Exception as e:
                    logger.error(f"Error converting item to string: {e}")
                    continue
            item = item.strip()
            seen_set = getattr(self, f"_seen_set_{field_name}")
            current_items = getattr(self, field_name)
            if item in seen_set:
                continue
            current_items.append(item)
            seen_set.add(item)

    def _update_dict_field(
        self,
        field_name: str,
        new_items: dict[str, str],
    ) -> None:
        if not isinstance(new_items, dict):
            logger.error(f"Expected dict for {field_name}, got {type(new_items)}")
            return
        current_items = getattr(self, field_name)
        for key, value in new_items.items():
            if not isinstance(key, str) or not isinstance(value, str):
                logger.error(
                    f"Expected string for {field_name} items, got {type(key)}, {type(value)}"
                )
                continue
            current_items[key.strip()] = value.strip()
