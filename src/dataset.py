import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class ImageCaptionDataset(Dataset):
    """Generic Image Caption Dataset, Support Relative Image Path to Image Root Dir"""
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        img_dir: str = "",            
        image_col: str = "image_path",
        text_col: str = "caption",      
        transform = None                
    ):
        """
        Args:
            df(pd.DataFrame): DataFrame which stores related image path and Captions
            img_dir(str): Root directory of Image Datas
            img_col(str): Name of image column in DF
            text_col(str): Name of text column in DF
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.image_col = image_col
        self.text_col = text_col
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
    
        base_dir = self.img_dir if self.img_dir else Path(".")
        img_path = base_dir / str(row[self.image_col])
    
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Can't open this image, please check image path: {img_path}") from e

    
        if self.transform is not None:
            image = self.transform(image)
        
        text = str(row[self.text_col]) 
    
        return {
            "image": image,
            "text": text
        }