import logging
import re

logger = logging.getLogger(__name__)


def parse_angle_tag(text: str, tag: str) -> list[str]:
    """
    Parse the text to extract the content of the specified angle tag.
    The angle tag format is <tag>...</tag>.
    """
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    matches = pattern.findall(text)
    return [match.strip() for match in matches]


def parse_first_angle_tag(text: str, tag: str) -> str | None:
    all_tags = parse_angle_tag(text, tag)
    if len(all_tags) == 0:
        return None
    return all_tags[0]


def parse_obs_spec_output(text: str) -> dict[str, str]:
    res = {
        "text": text,
        "description": parse_first_angle_tag(text, "description"),
        "observations": parse_first_angle_tag(text, "observations"),
        "summary": parse_first_angle_tag(text, "summary"),
        "speculation": parse_first_angle_tag(text, "speculation"),
        "speculation_details": parse_first_angle_tag(text, "details"),
    }
    return res


OBS_SPEC_DESC_TEMPLATE = """\
Observation Summary:
{summary}

Speculation:
{speculation}"""


def reformat_description(
    parsed: dict[str, str | None], observation_only: bool = False
) -> str:
    """
    Target Format for Description:

    observation summary:
    {summary}
    speculation:
    {speculation (but without the details portion)}
    """

    summary = parsed.get("summary", "") or ""
    if summary == "":
        logger.info(
            "No summary found in the parsed output. Falling back to full observation section"
        )
        summary = parsed.get("observations", "") or ""

    if observation_only:
        return summary.strip()

    full_speculation = parsed.get("speculation", "") or ""
    if full_speculation == "":
        logger.info("No speculation found in the parsed output.")
    # replace <details>...</details> with empty string
    speculation = re.sub(
        r"<details>.*?</details>", "", full_speculation, flags=re.DOTALL
    ).strip()
    description = OBS_SPEC_DESC_TEMPLATE.format(
        summary=summary,
        speculation=speculation,
    )
    return description
