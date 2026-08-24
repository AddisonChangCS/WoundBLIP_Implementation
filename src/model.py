"""BLIP model loading and freezing"""
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

def _check_blip_trainable_params(model):
    """Print a breakdown of trainable vs total parameters."""
    vision_trainable = sum(p.numel() for p in model.vision_model.parameters() if p.requires_grad)
    text_trainable = sum(p.numel() for p in model.text_decoder.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model device name: |{model.device}|")
    print(f"Vision model trainable params: {vision_trainable:,}")
    print(f"Text model trainable params: {text_trainable:,}")
    print(f"Total trainable params: {total_trainable:,} / {total_params:,} ({100 * total_trainable / total_params:.2f}%)")

def load_model_and_processor(
    model_name: str,
    device: torch.device,
    freeze_vision_encoder: bool = True,
) -> tuple[BlipForConditionalGeneration, BlipProcessor]:
    """Load BLIP model and processor, optionally freeze vision encoder.
 
    Args:
        model_name: HuggingFace model identifier
        freeze_vision_encoder: if True, vision encoder weights are frozen
            so only the text decoder is trainable
 
    Returns:
        (model, processor)
    """
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
 
    if freeze_vision_encoder:
        for param in model.vision_model.parameters():
            param.requires_grad = False
 
    model.to(device)
    _check_blip_trainable_params(model)

    return model, processor