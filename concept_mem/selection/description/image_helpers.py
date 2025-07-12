import base64
from pathlib import Path

from concept_mem.constants import REPO_ROOT
from concept_mem.data.arc_agi import Problem
from concept_mem.utils import visualize_problem

DEFAULT_IMAGE_DIR = "data/vlm_inputs"


def create_barc_seed_img(
    seed: Problem, output_directory: Path = (REPO_ROOT / DEFAULT_IMAGE_DIR)
) -> str:
    output_directory.mkdir(parents=True, exist_ok=True)
    drawing = visualize_problem(seed)
    target_file = str(output_directory / f"{seed.uid}.png")
    drawing.save_png(target_file)
    return target_file


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
