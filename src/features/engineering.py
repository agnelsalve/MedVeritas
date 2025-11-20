"""
Feature engineering utilities for MedVeritas project.
Creates TF-IDF features, categorical encodings, and derived features.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


def create_tfidf_features(df: pd.DataFrame, text_col: str = 'review_processed',
                          max_features: int = 500, ngram_range: Tuple[int, int] = (1, 1),
                          return_sparse: bool = True) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Create TF-IDF features from text.
    
    Args:
        df: DataFrame with preprocessed text
        text_col: Name of preprocessed text column
        max_features: Maximum number of features (reduced to avoid memory issues)
        ngram_range: Range of n-grams to extract
        return_sparse: Whether to return sparse matrix (saves memory)
    
    Returns:
        Tuple of (feature_df, vectorizer)
    """
    print(f"Creating TF-IDF features (max_features={max_features}, ngram_range={ngram_range})...")
    
    texts = df[text_col].fillna('').astype(str)
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
    
    if return_sparse:
        tfidf_df = pd.DataFrame.sparse.from_spmatrix(
            tfidf_matrix,
            columns=feature_names,
            index=df.index
        )
    else:
        print("Warning: Converting sparse matrix to dense array. This may use significant memory.")
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=df.index
        )
    
    print(f"Created {len(feature_names)} TF-IDF features (sparse={return_sparse}).")
    return tfidf_df, vectorizer


def encode_categorical_features(df: pd.DataFrame, 
                                categorical_cols: List[str] = ['drugName', 'condition']) -> pd.DataFrame:
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df: DataFrame
        categorical_cols: List of categorical column names
    
    Returns:
        DataFrame with encoded features
    """
    print("Encoding categorical features...")
    df_encoded = df.copy()
    
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"  Encoded {col}: {df[col].nunique()} unique values")
    
    return df_encoded, encoders


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from existing columns.
    
    Args:
        df: DataFrame
    
    Returns:
        DataFrame with derived features
    """
    print("Creating derived features...")
    df_derived = df.copy()
    
    if 'drugName' in df.columns:
        drug_counts = df['drugName'].value_counts()
        df_derived['drug_popularity'] = df['drugName'].map(drug_counts)
        print(f"  Created drug_popularity feature")
    
    if 'condition' in df.columns:
        condition_counts = df['condition'].value_counts()
        df_derived['condition_popularity'] = df['condition'].map(condition_counts)
        print(f"  Created condition_popularity feature")
    
    if 'drugName' in df.columns and 'rating' in df.columns:
        drug_avg_rating = df.groupby('drugName')['rating'].mean()
        df_derived['drug_avg_rating'] = df['drugName'].map(drug_avg_rating)
        print(f"  Created drug_avg_rating feature")
    
    if 'condition' in df.columns and 'rating' in df.columns:
        condition_avg_rating = df.groupby('condition')['rating'].mean()
        df_derived['condition_avg_rating'] = df['condition'].map(condition_avg_rating)
        print(f"  Created condition_avg_rating feature")
    
    if 'review_length' in df.columns:
        df_derived['review_length_category'] = pd.cut(
            df['review_length'],
            bins=[0, 100, 500, 1000, float('inf')],
            labels=['short', 'medium', 'long', 'very_long']
        )
        print(f"  Created review_length_category feature")
    
    print("Derived features created successfully.")
    return df_derived


def prepare_features(df: pd.DataFrame, 
                     text_col: str = 'review_processed',
                     include_tfidf: bool = True,
                     max_tfidf_features: int = 500,
                     categorical_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Prepare all features for modeling.
    
    Args:
        df: DataFrame with preprocessed data
        text_col: Name of preprocessed text column
        include_tfidf: Whether to include TF-IDF features
        max_tfidf_features: Maximum TF-IDF features
        categorical_cols: List of categorical columns to encode
    
    Returns:
        Tuple of (feature_df, feature_info_dict)
    """
    print("Preparing features for modeling...")
    
    if categorical_cols is None:
        categorical_cols = ['drugName', 'condition']
    
    df_features = create_derived_features(df)
    df_features, encoders = encode_categorical_features(df_features, categorical_cols)
    
    feature_cols = []
    
    text_feature_cols = [col for col in df_features.columns if any(
        x in col.lower() for x in ['vader', 'textblob', 'review_length', 'word_count', 
                                   'char_count', 'avg_word_length', 'exclamation', 
                                   'question', 'uppercase']
    )]
    feature_cols.extend(text_feature_cols)
    
    encoded_cols = [col for col in df_features.columns if col.endswith('_encoded')]
    feature_cols.extend(encoded_cols)
    
    derived_cols = [col for col in df_features.columns if any(
        x in col for x in ['popularity', 'avg_rating']
    )]
    feature_cols.extend(derived_cols)
    
    temporal_cols = [col for col in df_features.columns if col in ['year', 'month']]
    feature_cols.extend(temporal_cols)
    
    base_features = df_features[feature_cols].copy()
    
    feature_info = {
        'base_features': feature_cols,
        'encoders': encoders,
        'vectorizer': None
    }
    
    if include_tfidf and text_col in df_features.columns:
        tfidf_df, vectorizer = create_tfidf_features(
            df_features, 
            text_col=text_col,
            max_features=max_tfidf_features,
            return_sparse=True
        )
        base_features = pd.concat([base_features, tfidf_df], axis=1, sort=False)
        feature_info['vectorizer'] = vectorizer
        feature_info['tfidf_features'] = list(tfidf_df.columns)
    
    print(f"Prepared {len(base_features.columns)} features for modeling.")
    
    return base_features, feature_info

