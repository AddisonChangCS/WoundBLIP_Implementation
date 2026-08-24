"""Data reading and splitting stage (Preprocess).
 
Reads all *_gt.csv from RESULT_DIR, builds a unified DataFrame,
and splits into train / validation / test sets.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def read_csv(csv_dir: Path) -> pd.DataFrame:
    """Read All gt_csv in csv_dir
    
    Also builds the 'Image' column as '<Label>/<ID>.jpg'
    so ImageCaptionDataset can locate the file directly.

    Returns:
        pd.DataFrame with columns: ID, Label, Image, Caption
    """
    df_list=[]
    for csv in csv_dir.glob("*_gt.csv"):
        df=pd.read_csv(csv)
        df_list.append(df)
    if not df_list:
        raise FileNotFoundError(f"No *_gt.csv found in {csv_dir}")

    result_df=pd.concat(df_list, ignore_index=True)
    result_df['Image'] = result_df['Label'] + '/' + result_df['ID'].astype(str) + '.jpg'
    result_df=result_df[['ID', 'Label', 'Image', 'Caption']]

    return result_df

def split_data(df: pd.DataFrame,
               val_size: float=0.1, 
               test_size: float=0.1, 
               random_state: int=42
)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Stratified train / vali / test split.
 
    Args:
        df: full DataFrame
        test_size: fraction of test dataset out of total
        vali_size: fraction of validation dataset out of total
        random_state: reproducibility seed
 
    Returns:
        (train_df, vali_df, test_df)
    """
    temp_ratio=val_size+test_size
    test_ratio=test_size/temp_ratio

    train_df, temp_df=train_test_split(
        df, 
        test_size=temp_ratio, 
        stratify=df['Label'], 
        random_state=random_state
    )
    vali_df, test_df=train_test_split(
        temp_df, 
        test_size=test_ratio, 
        stratify=temp_df['Label'], 
        random_state=random_state
    )

    return (train_df,vali_df,test_df)
