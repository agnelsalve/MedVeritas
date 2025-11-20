"""
NLP utilities for text preprocessing and sentiment analysis.
"""

import re
import string
import numpy as np
import pandas as pd
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words.update(['drug', 'medication', 'medicine', 'pill', 'pills', 'mg', 'ml'])
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        return tokens
    
    def preprocess(self, text: str, remove_stopwords: bool = True, 
                   lemmatize: bool = True) -> str:
        """
        Full text preprocessing pipeline.
        
        Args:
            text: Raw text
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize tokens
        
        Returns:
            Preprocessed text
        """
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        tokens = [t for t in tokens if t not in string.punctuation and len(t) > 2]
        
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)


class SentimentAnalyzer:
    """Sentiment analysis using multiple methods."""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def get_vader_sentiment(self, text: str) -> dict:
        """Get VADER sentiment scores."""
        scores = self.vader.polarity_scores(str(text))
        return {
            'vader_compound': scores['compound'],
            'vader_pos': scores['pos'],
            'vader_neu': scores['neu'],
            'vader_neg': scores['neg']
        }
    
    def get_textblob_sentiment(self, text: str) -> dict:
        """Get TextBlob sentiment scores."""
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        return {
            'textblob_polarity': polarity,
            'textblob_subjectivity': subjectivity
        }
    
    def get_all_sentiment(self, text: str) -> dict:
        """Get all sentiment scores."""
        vader_scores = self.get_vader_sentiment(text)
        textblob_scores = self.get_textblob_sentiment(text)
        return {**vader_scores, **textblob_scores}


def extract_text_features(df: pd.DataFrame, text_col: str = 'review', 
                          use_sentiment: bool = True, batch_size: int = 10000) -> pd.DataFrame:
    """
    Extract text-based features from reviews.
    
    Args:
        df: DataFrame with review column
        text_col: Name of text column
        use_sentiment: Whether to compute sentiment (can be slow for large datasets)
        batch_size: Batch size for sentiment analysis
    
    Returns:
        DataFrame with additional text features
    """
    print("Extracting text features...")
    
    df['review_length'] = df[text_col].astype(str).str.len()
    df['word_count'] = df[text_col].astype(str).str.split().str.len()
    df['char_count'] = df[text_col].astype(str).str.len()
    df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1)
    
    df['exclamation_count'] = df[text_col].astype(str).str.count('!')
    df['question_count'] = df[text_col].astype(str).str.count('\\?')
    df['uppercase_ratio'] = df[text_col].astype(str).str.findall(r'[A-Z]').str.len() / df['char_count'].replace(0, 1)
    if use_sentiment:
        print(f"Computing sentiment scores for {len(df)} reviews (this may take a while)...")
        sentiment_analyzer = SentimentAnalyzer()
        
        sentiment_results = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            batch_num = (i // batch_size) + 1
            print(f"  Processing batch {batch_num}/{total_batches} (rows {i+1}-{batch_end})...")
            
            batch_scores = df[text_col].iloc[i:batch_end].apply(sentiment_analyzer.get_all_sentiment)
            sentiment_results.extend(batch_scores.tolist())
        
        sentiment_df = pd.DataFrame(sentiment_results, index=df.index)
        df = pd.concat([df, sentiment_df], axis=1)
        print("  Sentiment scores computed.")
    else:
        print("  Skipping sentiment analysis (set use_sentiment=True to enable)")
    
    print("Text features extracted successfully.")
    return df


def preprocess_reviews(df: pd.DataFrame, text_col: str = 'review', 
                      preprocessed_col: str = 'review_processed',
                      batch_size: int = 10000) -> pd.DataFrame:
    """
    Preprocess reviews using the TextPreprocessor.
    
    Args:
        df: DataFrame with review column
        text_col: Name of text column
        preprocessed_col: Name for preprocessed column
        batch_size: Batch size for processing (shows progress)
    
    Returns:
        DataFrame with preprocessed reviews
    """
    print(f"Preprocessing {len(df)} reviews (this may take a while)...")
    preprocessor = TextPreprocessor()
    
    preprocessed_results = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in range(0, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))
        batch_num = (i // batch_size) + 1
        print(f"  Processing batch {batch_num}/{total_batches} (rows {i+1}-{batch_end})...")
        
        batch_preprocessed = df[text_col].iloc[i:batch_end].apply(
            lambda x: preprocessor.preprocess(x, remove_stopwords=True, lemmatize=True)
        )
        preprocessed_results.extend(batch_preprocessed.tolist())
    
    df[preprocessed_col] = preprocessed_results
    print("Reviews preprocessed successfully.")
    return df

