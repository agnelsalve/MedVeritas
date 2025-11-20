"""
Visualization utilities for MedVeritas project.
Creates statistical, NLP, and insight visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from typing import Optional, List, Tuple
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def ensure_dir(directory: str):
    """Ensure directory exists."""
    directory = os.path.normpath(directory)
    os.makedirs(directory, exist_ok=True)


def save_plot(fig, filename: str, output_dir: str = 'results/figures'):
    """Save plot to file."""
    output_dir = os.path.normpath(output_dir)
    ensure_dir(output_dir)
    filepath = os.path.normpath(os.path.join(output_dir, filename))
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filepath}")
    plt.close(fig)


# Statistical Visualizations

def plot_rating_distribution_by_category(df: pd.DataFrame, category_col: str = 'condition',
                                        output_dir: str = 'results/figures'):
    """
    Plot distribution of ratings by category (histogram/box plots).
    
    Args:
        df: DataFrame with rating and category columns
        category_col: Name of category column
        output_dir: Output directory for plots
    """
    print(f"Creating rating distribution plots by {category_col}...")
    
    # Get top categories by count
    top_categories = df[category_col].value_counts().head(10).index
    
    # Box plot
    fig, ax = plt.subplots(figsize=(14, 8))
    df_top = df[df[category_col].isin(top_categories)]
    sns.boxplot(data=df_top, x=category_col, y='rating', ax=ax)
    ax.set_title(f'Rating Distribution by {category_col.title()} (Top 10)', fontsize=14, fontweight='bold')
    ax.set_xlabel(category_col.title(), fontsize=12)
    ax.set_ylabel('Rating', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    save_plot(fig, f'rating_distribution_boxplot_{category_col}.png', output_dir)
    
    # Histogram
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, category in enumerate(top_categories):
        category_data = df[df[category_col] == category]['rating']
        axes[i].hist(category_data, bins=10, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'{category[:30]}', fontsize=10)
        axes[i].set_xlabel('Rating')
        axes[i].set_ylabel('Frequency')
        axes[i].set_xlim(0, 11)
    
    plt.suptitle(f'Rating Distribution by {category_col.title()} (Top 10)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_plot(fig, f'rating_distribution_histogram_{category_col}.png', output_dir)


def plot_wordclouds(df: pd.DataFrame, text_col: str = 'review_processed',
                    rating_col: str = 'rating', threshold: int = 7,
                    output_dir: str = 'results/figures'):
    """
    Create word clouds for positive vs negative reviews.
    
    Args:
        df: DataFrame with text and rating columns
        text_col: Name of text column
        rating_col: Name of rating column
        threshold: Rating threshold for positive/negative
        output_dir: Output directory for plots
    """
    print("Creating word clouds...")
    
    # Positive reviews
    positive_reviews = ' '.join(df[df[rating_col] >= threshold][text_col].fillna('').astype(str))
    
    # Negative reviews
    negative_reviews = ' '.join(df[df[rating_col] < threshold][text_col].fillna('').astype(str))
    
    # Create word clouds
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    if positive_reviews:
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white',
                                 max_words=100, colormap='Greens').generate(positive_reviews)
        axes[0].imshow(wordcloud_pos, interpolation='bilinear')
        axes[0].set_title('Positive Reviews (Rating >= 7)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
    
    if negative_reviews:
        wordcloud_neg = WordCloud(width=800, height=400, background_color='white',
                                 max_words=100, colormap='Reds').generate(negative_reviews)
        axes[1].imshow(wordcloud_neg, interpolation='bilinear')
        axes[1].set_title('Negative Reviews (Rating < 7)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
    
    plt.tight_layout()
    save_plot(fig, 'wordclouds_positive_vs_negative.png', output_dir)


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: List[str],
                             target_col: str = 'rating',
                             output_dir: str = 'results/figures'):
    """
    Plot correlation heatmap between features and target.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Name of target column
        output_dir: Output directory for plots
    """
    print("Creating correlation heatmap...")
    
    # Select numeric features only
    numeric_cols = [col for col in feature_cols if col in df.columns and 
                    df[col].dtype in [np.int64, np.float64]]
    
    if not numeric_cols:
        print("No numeric features found for correlation heatmap.")
        return
    
    # Limit to top features by correlation
    correlations = df[numeric_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
    top_features = correlations.head(20).index.tolist()
    top_features = [f for f in top_features if f != target_col]
    
    # Create correlation matrix
    corr_matrix = df[top_features + [target_col]].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Heatmap: Features vs Effectiveness', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'correlation_heatmap.png', output_dir)


def plot_time_series_ratings(df: pd.DataFrame, date_col: str = 'date',
                            drug_col: str = 'drugName', rating_col: str = 'rating',
                            top_n_drugs: int = 10,
                            output_dir: str = 'results/figures'):
    """
    Plot rating trends over time for popular drugs.
    
    Args:
        df: DataFrame with date, drug, and rating columns
        date_col: Name of date column
        drug_col: Name of drug column
        rating_col: Name of rating column
        top_n_drugs: Number of top drugs to plot
        output_dir: Output directory for plots
    """
    if date_col not in df.columns:
        print(f"Date column '{date_col}' not found. Skipping time series plot.")
        return
    
    print("Creating time series plot...")
    
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        
        # Get top drugs by review count
        top_drugs = df[drug_col].value_counts().head(top_n_drugs).index
        
        # Aggregate by month
        df['year_month'] = df[date_col].dt.to_period('M')
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        for drug in top_drugs:
            drug_data = df[df[drug_col] == drug]
            monthly_avg = drug_data.groupby('year_month')[rating_col].mean()
            ax.plot(monthly_avg.index.astype(str), monthly_avg.values, 
                   marker='o', label=drug[:30], linewidth=2, markersize=4)
        
        ax.set_title(f'Rating Trends Over Time (Top {top_n_drugs} Drugs)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Year-Month', fontsize=12)
        ax.set_ylabel('Average Rating', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        save_plot(fig, 'time_series_ratings.png', output_dir)
    except Exception as e:
        print(f"Error creating time series plot: {e}")


def plot_feature_importance(feature_importance: dict, top_n: int = 15,
                           output_dir: str = 'results/figures'):
    """
    Plot feature importance bar chart.
    
    Args:
        feature_importance: Dictionary of {feature_name: importance_score}
        top_n: Number of top features to show
        output_dir: Output directory for plots
    """
    print("Creating feature importance plot...")
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]
    
    features, importances = zip(*top_features)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['green' if imp > 0 else 'red' for imp in importances]
    ax.barh(range(len(features)), importances, color=colors, alpha=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    save_plot(fig, 'feature_importance.png', output_dir)


# NLP Visualizations

def plot_topic_modeling_results(lda_model, dictionary=None, top_n_words: int = 10,
                               output_dir: str = 'results/figures'):
    """
    Visualize topic modeling results from gensim LDA model.
    
    Args:
        lda_model: Trained gensim LDA model
        dictionary: Gensim dictionary (optional, for word lookup)
        top_n_words: Number of top words per topic
        output_dir: Output directory for plots
    """
    print("Creating topic modeling visualization...")
    
    n_topics = lda_model.num_topics
    
    # Calculate number of rows and columns for subplots
    n_cols = 5
    n_rows = (n_topics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_topics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for topic_idx in range(n_topics):
        # Get top words for this topic
        topic_words = lda_model.show_topic(topic_idx, topn=top_n_words)
        words, scores = zip(*topic_words)
        
        ax = axes[topic_idx]
        ax.barh(range(len(words)), scores, alpha=0.7, color='steelblue')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=9)
        ax.set_title(f'Topic {topic_idx + 1}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Weight', fontsize=9)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    
    # Hide extra subplots
    for idx in range(n_topics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Topic Modeling Results (LDA)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'topic_modeling_results.png', output_dir)


def plot_sentiment_distribution_by_condition(df: pd.DataFrame,
                                            sentiment_col: str = 'vader_compound',
                                            condition_col: str = 'condition',
                                            output_dir: str = 'results/figures'):
    """
    Plot sentiment distribution across different medical conditions.
    
    Args:
        df: DataFrame with sentiment and condition columns
        sentiment_col: Name of sentiment column
        condition_col: Name of condition column
        output_dir: Output directory for plots
    """
    print("Creating sentiment distribution plot...")
    
    if sentiment_col not in df.columns:
        print(f"Sentiment column '{sentiment_col}' not found.")
        return
    
    # Get top conditions
    top_conditions = df[condition_col].value_counts().head(15).index
    
    fig, ax = plt.subplots(figsize=(14, 8))
    df_top = df[df[condition_col].isin(top_conditions)]
    
    sns.boxplot(data=df_top, x=condition_col, y=sentiment_col, ax=ax)
    ax.set_title('Sentiment Distribution by Medical Condition (Top 15)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Sentiment Score (VADER Compound)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Neutral')
    ax.legend()
    plt.tight_layout()
    save_plot(fig, 'sentiment_distribution_by_condition.png', output_dir)


def plot_confusion_matrix(y_true, y_pred, model_name: str = 'Model',
                         output_dir: str = 'results/figures'):
    """
    Plot confusion matrix for classification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        output_dir: Output directory for plots
    """
    from sklearn.metrics import confusion_matrix
    
    print(f"Creating confusion matrix for {model_name}...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Effective', 'Effective'],
                yticklabels=['Not Effective', 'Effective'])
    ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    save_plot(fig, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', output_dir)


def plot_roc_curves(roc_data: List[Tuple[str, np.ndarray, np.ndarray, float]],
                   output_dir: str = 'results/figures'):
    """
    Plot ROC curves comparing different models.
    
    Args:
        roc_data: List of tuples (model_name, fpr, tpr, auc_score)
        output_dir: Output directory for plots
    """
    from sklearn.metrics import roc_curve, auc
    
    print("Creating ROC curves...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, fpr, tpr, auc_score in roc_data:
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, 'roc_curves_comparison.png', output_dir)


# Insight Visualizations

def plot_top_effective_drugs_by_condition(df: pd.DataFrame, top_n: int = 10,
                                          condition_col: str = 'condition',
                                          drug_col: str = 'drugName',
                                          rating_col: str = 'rating',
                                          output_dir: str = 'results/figures'):
    """
    Plot top N most effective drugs by condition.
    
    Args:
        df: DataFrame with condition, drug, and rating columns
        top_n: Number of top drugs to show per condition
        condition_col: Name of condition column
        drug_col: Name of drug column
        rating_col: Name of rating column
        output_dir: Output directory for plots
    """
    print("Creating top effective drugs plot...")
    
    # Get top conditions by review count
    top_conditions = df[condition_col].value_counts().head(10).index
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, condition in enumerate(top_conditions):
        condition_data = df[df[condition_col] == condition]
        drug_ratings = condition_data.groupby(drug_col)[rating_col].mean().sort_values(ascending=False)
        top_drugs = drug_ratings.head(top_n)
        
        ax = axes[i]
        ax.barh(range(len(top_drugs)), top_drugs.values, alpha=0.7)
        ax.set_yticks(range(len(top_drugs)))
        ax.set_yticklabels([d[:20] for d in top_drugs.index], fontsize=8)
        ax.set_title(f'{condition[:25]}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Avg Rating', fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 11)
    
    plt.suptitle(f'Top {top_n} Most Effective Drugs by Condition', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_plot(fig, 'top_effective_drugs_by_condition.png', output_dir)


def plot_side_effects_by_category(df: pd.DataFrame, text_col: str = 'review',
                                  category_col: str = 'drugName',
                                  output_dir: str = 'results/figures'):
    """
    Plot most frequently mentioned side effects by category.
    Note: This is a simplified version. Full side effect extraction would require NLP.
    
    Args:
        df: DataFrame with text and category columns
        text_col: Name of text column
        category_col: Name of category column
        output_dir: Output directory for plots
    """
    print("Creating side effects analysis plot...")
    
    # Common side effect keywords
    side_effect_keywords = [
        'nausea', 'headache', 'dizziness', 'fatigue', 'drowsiness',
        'insomnia', 'diarrhea', 'constipation', 'dry mouth', 'rash',
        'anxiety', 'depression', 'weight gain', 'weight loss', 'pain'
    ]
    
    # Get top categories
    top_categories = df[category_col].value_counts().head(10).index
    
    side_effect_counts = {}
    for category in top_categories:
        category_data = df[df[category_col] == category]
        text = ' '.join(category_data[text_col].fillna('').astype(str).str.lower())
        counts = {keyword: text.count(keyword) for keyword in side_effect_keywords}
        side_effect_counts[category] = counts
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    categories = list(side_effect_counts.keys())
    keywords = side_effect_keywords
    
    data_matrix = np.array([[side_effect_counts[cat][kw] for kw in keywords] for cat in categories])
    
    sns.heatmap(data_matrix, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=keywords, yticklabels=[c[:30] for c in categories],
                ax=ax, cbar_kws={'label': 'Mention Count'})
    ax.set_title('Side Effect Mentions by Category (Top 10)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Side Effect Keywords', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    plt.tight_layout()
    save_plot(fig, 'side_effects_by_category.png', output_dir)


def plot_review_length_vs_rating(df: pd.DataFrame,
                                 length_col: str = 'review_length',
                                 rating_col: str = 'rating',
                                 output_dir: str = 'results/figures'):
    """
    Plot review length vs rating scatter plot.
    
    Args:
        df: DataFrame with length and rating columns
        length_col: Name of length column
        rating_col: Name of rating column
        output_dir: Output directory for plots
    """
    print("Creating review length vs rating plot...")
    
    if length_col not in df.columns:
        print(f"Length column '{length_col}' not found.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sample data if too large
    plot_df = df.sample(n=min(10000, len(df)), random_state=42) if len(df) > 10000 else df
    
    scatter = ax.scatter(plot_df[length_col], plot_df[rating_col], 
                        alpha=0.5, s=20, c=plot_df[rating_col], 
                        cmap='viridis', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Review Length (characters)', fontsize=12)
    ax.set_ylabel('Rating', fontsize=12)
    ax.set_title('Review Length vs Rating', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(plot_df[length_col], plot_df[rating_col], 1)
    p = np.poly1d(z)
    ax.plot(plot_df[length_col].sort_values(), p(plot_df[length_col].sort_values()), 
           "r--", alpha=0.8, linewidth=2, label='Trend Line')
    ax.legend()
    
    plt.colorbar(scatter, ax=ax, label='Rating')
    plt.tight_layout()
    save_plot(fig, 'review_length_vs_rating.png', output_dir)

