class ImageCaptionCollator:
    """
    Generic Callabel class including Hugging Face Processor for Dataloader 
    """
    def __init__(self, processor, max_length=64):
       self.processor=processor
       self.max_length=max_length

    def __call__(self, batch):
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]

        tensors = self.processor(
            images=images,
            text=texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = tensors["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        tensors["labels"] = labels

        return tensors