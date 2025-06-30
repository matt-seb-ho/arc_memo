from pathlib import Path

# paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DOTENV_PATH = REPO_ROOT / ".env"
COMMON_LIB_PATH = REPO_ROOT / "data/common_lib.txt"
HYRDA_CONFIG_PATH = str(REPO_ROOT / "configs")
BARC_SEED_UIDS_PATH = REPO_ROOT / "data/barc_seed_uids.json"
BARC_SEEDS_PATH = REPO_ROOT / "data/barc_seeds"

# misc
BARC_DATASET_ID = "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems"
DEFAULT_CODE = "# No code"
EXCLUDED_CONCEPTS = {"color change"}
NO_CODE_BLOCK_MESSAGE = "No code block found in the response."

# max line width:
# max dimension is 30 so that's
# [space][open brace][30 * ([digit][comma][space]))][close brace][comma]
# that's 4 chars for outer braces, and 30 * 3 = 90 chars for pixels
# to be safe, we'll use 120 as the max line width
MAX_LINE_WIDTH = 120

O4M_ID = "o4-mini-2025-04-16"
