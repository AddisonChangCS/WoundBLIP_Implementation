"""Ground Truth generating stage
"""
import base64
import json
from pathlib import Path
 
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, AuthenticationError
 
from config import paths, GT_MODELS_CONFIG

PAD_WIDTH = 3
#----------------utils---------------------#
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def read_prompt(prompt_name: str):
    prompt_path = paths.PROMPT_DIR / prompt_name
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
    
def render_prompt(prompt_template: str, image_base64: str, label: str):
    rendered = prompt_template.replace("{{image_base64}}", image_base64)
    rendered = rendered.replace("{{label}}", label)

    return json.loads(rendered)

def get_model_config(model_key: str):
    return GT_MODELS_CONFIG["models"][model_key]

def build_client(model_cfg: dict) -> OpenAI:
    import os
    api_key = os.getenv(model_cfg["api_key_env"])
    if not api_key:
        raise ValueError(f"can't find api_key name:{model_cfg['api_key_env']} in .env file")
    return OpenAI(api_key=api_key, base_url=model_cfg["base_url"])

def generate_caption(client: OpenAI, model_cfg: dict, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=model_cfg["model_name"],
        messages=messages,
        max_tokens=model_cfg["max_token"],
        temperature=model_cfg["temperature"],
    )
    return response.choices[0].message.content

def get_csv_path(label: str) -> Path:
    return paths.RESULT_DIR / f"{label}_gt.csv"

def sort_csv_by_id(csv_path: Path) -> None:
    df = pd.read_csv(csv_path, dtype={"ID": str})
    df["_sort_key"] = df["ID"]
    df = df.sort_values("_sort_key").drop(columns="_sort_key")
    df.to_csv(csv_path, index=False)

def append_row_to_csv(row: dict, csv_path: Path) -> None:
    df = pd.DataFrame([row])
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)
#--------------cores-------------------#
def generate_ground_truth(
    label: str,
    image_path: Path,
    client: OpenAI,
    model_cfg: dict,
    prompt_template: str,
):
    img_id = image_path.stem.zfill(PAD_WIDTH)
    img_base64 = encode_image_to_base64(image_path)
 
    messages = render_prompt(prompt_template, img_base64, label)
    caption = generate_caption(client, model_cfg, messages)
 
    return {"ID": img_id, "Label": label, "Caption": caption}


def load_processed_ids(csv_path: Path) -> set:
    """get processed result

    Returns:
        set: processed ids in previous csv 
    """
    try:
        result_df = pd.read_csv(csv_path, dtype={"ID": str})
        return set(result_df["ID"].astype(str))
    except(FileNotFoundError, pd.errors.EmptyDataError):
        return set()

def list_images(label: str) -> list[Path]:
    """list all .jpg in label.dir
    """
    label_dir = paths.DATA_DIR / label
    if not label_dir.exists():
        raise FileNotFoundError(f"Can't find dir: {label_dir}")
    return sorted(label_dir.glob("*.jpg"), key=lambda p: p.stem.zfill(PAD_WIDTH))

#--------------------main----------------------------#

def run_stage(
    label: str,
    model_key: str = "gemini-3.1-lite",
    prompt_name: str = "gt_prompt_v0.json"
) -> Path:
    """inference all Unprocessed Image under specified label

    Args:
        label: specified label of image wants to process
        model_key: model key in gt_models config
        prompt_name: prompt.json using in prompts/
    Returns:
        Path: path of result
    """

    model_cfg=get_model_config(model_key)
    prompt_template = read_prompt(prompt_name)
    csv_path=get_csv_path(label)
    processed_ids = load_processed_ids(csv_path)

    n_failed=0
    n_success=0

    image_paths = list_images(label)

    load_dotenv(paths.ENV_PATH)
    client = build_client(model_cfg)

    for img_path in image_paths:
        img_id = img_path.stem.zfill(PAD_WIDTH)
        if img_id not in processed_ids:
            try:
                ground_truth=generate_ground_truth(label=label, image_path=img_path, client=client, model_cfg=model_cfg, prompt_template=prompt_template)
                append_row_to_csv(row=ground_truth,csv_path=csv_path)
                n_success += 1
                print(f"[OK] finish {img_path.name}")
            except AuthenticationError as e:
                print(f"[FATAL] API key verification failed, aborting:{e}")
                break
            except RateLimitError as e:
                print(f"[FAILED] {img_path.name} Insufficient quota or rate limit exceeded:{e}")
                n_failed += 1
            except APIError as e:
                print(f"[FAILED] {img_path.name} API error:{e}")
                n_failed += 1

    if csv_path.exists():
        sort_csv_by_id(csv_path)

    print(f"[Finish] Ground Truth generation task on label:{label} completed. Success:{n_success}, Failed:{n_failed}")

    return csv_path