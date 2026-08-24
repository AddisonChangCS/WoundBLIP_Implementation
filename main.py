"""Pipeline Entry
 
Usage:
    python main.py gt-generate --label Ulcer
    python main.py gt-generate --label Ulcer --model gemini-3.1-lite --prompt gt_prompt_v0.json
"""
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

import src.data as data
from config import paths
from config.manager import ConfigManager
from src.gt_generate import run_stage as run_gt_generate
from src.model import load_model_and_processor
from src.dataset import ImageCaptionDataset
from src.collator import ImageCaptionCollator
from src.trainer import WoundBLIPTrainer
from src.inference import evaluate_test_set


def _build_dataloader(df, processor, cfg, shuffle: bool, drop_last: bool=False,  persistent_workers: bool=False) -> DataLoader:
    return DataLoader(
        ImageCaptionDataset(
            df,
            img_dir=paths.DATA_DIR,
            image_col=cfg.data.image_col,
            text_col=cfg.data.text_col,
        ),
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        collate_fn=ImageCaptionCollator(
            processor=processor,
            max_length=cfg.data.max_length,
        ),
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wound care pipeline entry point")
    subparsers = parser.add_subparsers(dest="stage", required=True)
 
    gt_parser = subparsers.add_parser("gt-generate", help="Generate ground truth for dataset under specific label")
    gt_parser.add_argument("--label",  required=True)
    gt_parser.add_argument("--model",  default="gemini-3.1-lite")
    gt_parser.add_argument("--prompt", default="gt_prompt_v0.json")
 
    train_parser = subparsers.add_parser("train", help="Fine-tune BLIP")
    train_parser.add_argument("--config", default="config/train_config.yaml")
 
    infer_parser = subparsers.add_parser("infer", help="Run inference on saved test split")
    infer_parser.add_argument("--config", default="config/train_config.yaml")
 
    return parser.parse_args()

def main() -> None:
    args = parse_args()
 
    if args.stage == "gt-generate":
        csv_path = run_gt_generate(
            label=args.label, model_key=args.model, prompt_name=args.prompt
        )
        print(f"Result saved to: {csv_path}")
 
    elif args.stage == "train":
        # ConfigManager loads YAML and auto-resolves device
        manager = ConfigManager(args.config)
        cfg = manager.cfg
        run_dir=Path(cfg.run_dir)
 
        WoundBLIPTrainer.seed_everything(cfg.data.random_state)
 
        # data
        df=data.read_csv(paths.DATA_DIR)
        train_df, val_df, test_df = data.split_data(
            df=df,
            val_size=cfg.data.val_size,
            test_size=cfg.data.test_size,
            random_state=cfg.data.random_state
        )
        print(f"Split: train={len(train_df)}, vali={len(val_df)}, test={len(test_df)}")
        train_df.to_csv(run_dir / "train_split.csv", index=False)
        val_df.to_csv(run_dir / "val_split.csv", index=False)
        test_df.to_csv(run_dir / "test_split.csv", index=False)

        # model
        model, processor= load_model_and_processor(
            model_name=cfg.model.name,
            freeze_vision_encoder=cfg.model.freeze_vision_encoder,
        )

        #dataloaders
        train_loader=_build_dataloader(
            df=train_df,
            processor=processor,
            cfg=cfg,
            shuffle=True,
            drop_last= True,
            persistent_workers=True
        )
        val_loader=_build_dataloader(
            df=val_df,
            processor=processor,
            cfg=cfg,
            shuffle=False,
            drop_last=False,
            persistent_workers=True
        )        

        # train: will save best model during training
        trainer = WoundBLIPTrainer(
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
        )
        best_model_path = trainer.fit()
        processor.save_pretrained(best_model_path)

        manager.save(run_dir / "config_resolved.yaml")
 
        print(f"Training complete. Best model: {best_model_path}")
    elif args.stage == "infer":
        #config
        manager = ConfigManager(args.config)
        cfg = manager.cfg
        run_dir=Path(cfg.run_dir)

        #model
        model_path=run_dir / "model"
        if not model_path.exists():
            raise FileNotFoundError(
            f"model not found in {run_dir}. "
            "Run 'python main.py train' first."
        )
        model, processor = load_model_and_processor(
            model_name=model_path,
            device=cfg.device
        )

        #data
        test_split_path = run_dir / "test_split.csv"
        if not test_split_path.exists():
            raise FileNotFoundError(
                f"test_split.csv not found in {run_dir}. "
                "Run 'python main.py train' first."
            )
        test_df=pd.read_csv(test_split_path)
        print(f"Loaded test split: {len(test_df)} rows from {test_split_path}")

        #dataloader
        test_loader = _build_dataloader(
            df=test_df,
            processor=processor,
            cfg=cfg,
            shuffle=False,
            drop_last=False,
            persistent_workers=False
        )

        #inference
        test_df['prediction']=evaluate_test_set(
            dataloader=test_loader,
            model=model,
            device=cfg.device,
            processor=processor
        )
        test_df.sort_values(by=["Label", "ID"], inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        #save
        test_df.to_csv(
        run_dir / (cfg.run_name + '.csv'),
        index=False,
        )

 
 
if __name__ == "__main__":
    main()