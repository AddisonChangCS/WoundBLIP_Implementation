import torch
from tqdm.auto import tqdm

def evaluate_test_set(dataloader, model, device, processor, max_tokens=150):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="test"):
            pixel_values = batch["pixel_values"].to(device)
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=max_tokens
            )

            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

            all_preds.extend(preds)
    return all_preds