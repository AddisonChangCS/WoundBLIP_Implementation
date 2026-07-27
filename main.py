"""Pipeline Entry
 
Usage:
    python main.py gt-generate --label Ulcer
    python main.py gt-generate --label Ulcer --model gemini-3.1-lite --prompt gt_prompt_v0.json
"""
import argparse
 
from src.gt_generate import run_stage as run_gt_generate
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Wound care pipeline entry point")
    subparsers = parser.add_subparsers(dest="stage", required=True)
 
    gt_parser = subparsers.add_parser(
        "gt-generate", help="Generate ground truth for dataset under specific label"
    )
    gt_parser.add_argument("--label", required=True, help="Label of dictionary")
    gt_parser.add_argument(
        "--model", default="gemini-3.1-lite", help="model key you want to use in config/gt_models.yaml"
    )
    gt_parser.add_argument(
        "--prompt", default="gt_prompt_v0.json", help="prompt name you want to use under prompts/"
    )
 
    args = parser.parse_args()
 
    if args.stage == "gt-generate":
        csv_path = run_gt_generate(
            label=args.label, model_key=args.model, prompt_name=args.prompt
        )
        print(f"Result saved to: {csv_path}")
 
 
if __name__ == "__main__":
    main()