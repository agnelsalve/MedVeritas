"""
Data processing utilities for MedVeritas project.
Handles data loading, cleaning, and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load data from Excel file.
    
    Args:
        file_path: Path to the Excel file
        sample_size: Optional sample size for testing (None = load all)
    
    Returns:
        DataFrame with loaded data
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"Sampled {len(df)} rows for testing.")
    
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing invalid entries.
    
    Args:
        df: Raw DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    print("Cleaning data...")
    initial_len = len(df)
    
    df = df.dropna(subset=['review', 'rating', 'drugName'])
    
    if 'condition' in df.columns:
        df['condition'] = df['condition'].astype(str).str.strip().str.replace("'", "", regex=False)
        df['condition'] = df['condition'].str.replace('"', '', regex=False)
        df['condition'] = df['condition'].str.replace('?', '', regex=False).str.strip()
    
    if 'review' in df.columns:
        df['review'] = df['review'].astype(str).str.strip()
        df['review'] = df['review'].str.replace("'", "", regex=False)
        df['review'] = df['review'].str.replace('"', '', regex=False)
    
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[(df['rating'] >= 1) & (df['rating'] <= 10)]
    df = df[df['review'].str.len() >= 10]
    df = df.drop_duplicates(subset=['review'], keep='first')
    
    final_len = len(df)
    print(f"Cleaned data: {initial_len} -> {final_len} rows ({initial_len - final_len} removed)")
    
    return df.reset_index(drop=True)


def create_effectiveness_label(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    """
    Create binary effectiveness label (effective >= threshold, not effective < threshold).
    
    Args:
        df: DataFrame with rating column
        threshold: Rating threshold for effectiveness (default: 7)
    
    Returns:
        DataFrame with 'is_effective' column added
    """
    df['is_effective'] = (df['rating'] >= threshold).astype(int)
    print(f"Created effectiveness labels (threshold={threshold}):")
    print(f"  Effective (>= {threshold}): {df['is_effective'].sum()} ({df['is_effective'].sum()/len(df)*100:.1f}%)")
    print(f"  Not Effective (< {threshold}): {(~df['is_effective'].astype(bool)).sum()} ({(~df['is_effective'].astype(bool)).sum()/len(df)*100:.1f}%)")
    return df


def extract_temporal_features(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    """
    Extract temporal features from date column if available.
    
    Args:
        df: DataFrame
        date_col: Name of date column (if exists)
    
    Returns:
        DataFrame with temporal features added
    """
    if date_col and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            df['year_month'] = df[date_col].dt.to_period('M')
            print("Extracted temporal features: year, month, year_month")
        except:
            print("Could not extract temporal features from date column")
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the dataset.
    
    Args:
        df: DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_reviews': len(df),
        'unique_drugs': df['drugName'].nunique() if 'drugName' in df.columns else 0,
        'unique_conditions': df['condition'].nunique() if 'condition' in df.columns else 0,
        'avg_rating': df['rating'].mean() if 'rating' in df.columns else 0,
        'median_rating': df['rating'].median() if 'rating' in df.columns else 0,
        'avg_review_length': df['review'].str.len().mean() if 'review' in df.columns else 0,
    }
    return summary


def prepare_train_test_split(df: pd.DataFrame, test_size: float = 0.2, 
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Args:
        df: DataFrame
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['is_effective'] if 'is_effective' in df.columns else None
    )
    
    print(f"Train set: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, test_df

