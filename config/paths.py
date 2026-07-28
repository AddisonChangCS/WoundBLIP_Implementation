from pathlib import Path

#Wound_Caption dir
ROOT_DIR=Path(__file__).resolve().parent.parent

ENV_PATH=ROOT_DIR / "private" / ".env"
PROMPT_DIR=ROOT_DIR / "prompts"
DATA_DIR=ROOT_DIR / "data"
RESULT_DIR=ROOT_DIR / "result"