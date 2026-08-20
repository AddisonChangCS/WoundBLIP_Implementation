from pathlib import Path

#Wound_Caption dir
ROOT_DIR=Path(__file__).resolve().parent.parent

PROMPT_DIR=ROOT_DIR / "prompts"
DATA_DIR=ROOT_DIR / "data"
RESULT_DIR=ROOT_DIR / "result"
CONFIG_DIR=Path(__file__).parent

ENV_PATH=ROOT_DIR / "private" / ".env"
GT_PATH=CONFIG_DIR / "gt_models.yaml"