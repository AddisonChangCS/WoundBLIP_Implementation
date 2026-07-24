from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).parent

def _load_yaml(filename: str):
    config_path = CONFIG_DIR / filename
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

GT_MODELS_CONFIG = _load_yaml("gt_models.yaml")