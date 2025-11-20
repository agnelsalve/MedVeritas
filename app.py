"""
Streamlit Dashboard for MedVeritas Drug Effectiveness Prediction

Interactive dashboard for:
- Drug comparison tool
- Effectiveness predictor (input text review → predicted rating)
- Data exploration and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.processing import load_data, clean_data, create_effectiveness_label
from src.nlp.utils import extract_text_features, preprocess_reviews, TextPreprocessor, SentimentAnalyzer
from src.features.engineering import prepare_features

st.set_page_config(
    page_title="MedVeritas - Drug Effectiveness Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

@st.cache_data
def load_processed_data():
    """Load processed data."""
    data_path = project_root / 'data' / 'processed' / 'medVe_data_final_version.xlsx'
    if not data_path.exists():
        return None
    
    df = load_data(str(data_path))
    df = clean_data(df)
    df = create_effectiveness_label(df, threshold=7)
    return df

@st.cache_resource
def load_models():
    """Load trained models."""
    models = {}
    model_dir = project_root / 'models'
    
    classification_dir = model_dir / 'classification'
    if classification_dir.exists():
        for model_file in classification_dir.glob('*.pkl'):
            model_name = model_file.stem
            try:
                with open(model_file, 'rb') as f:
                    models[f'classification_{model_name}'] = pickle.load(f)
            except Exception as e:
                st.warning(f"Could not load {model_name}: {e}")
    
    regression_dir = model_dir / 'regression'
    if regression_dir.exists():
        for model_file in regression_dir.glob('*.pkl'):
            model_name = model_file.stem
            try:
                with open(model_file, 'rb') as f:
                    models[f'regression_{model_name}'] = pickle.load(f)
            except Exception as e:
                st.warning(f"Could not load {model_name}: {e}")
    
    return models

def predict_effectiveness(review_text, models, feature_vectorizer=None):
    """Predict effectiveness from review text."""
    if not review_text or len(review_text.strip()) < 10:
        return None, "Review text too short"
    
    try:
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.preprocess(review_text)
        
        temp_df = pd.DataFrame({'review': [review_text], 'review_processed': [processed_text]})
        temp_df = extract_text_features(temp_df, text_col='review')
        
        sentiment_scores = {
            'vader_compound': temp_df['vader_compound'].iloc[0],
            'vader_pos': temp_df['vader_pos'].iloc[0],
            'vader_neg': temp_df['vader_neg'].iloc[0],
            'textblob_polarity': temp_df['textblob_polarity'].iloc[0],
            'review_length': temp_df['review_length'].iloc[0]
        }
        
        if sentiment_scores['vader_compound'] > 0.1:
            predicted_effective = True
            confidence = min(0.9, 0.5 + abs(sentiment_scores['vader_compound']) * 0.4)
        else:
            predicted_effective = False
            confidence = min(0.9, 0.5 + abs(sentiment_scores['vader_compound']) * 0.4)
        
        estimated_rating = 5 + (sentiment_scores['vader_compound'] * 5)
        estimated_rating = max(1, min(10, estimated_rating))
        
        return {
            'effective': predicted_effective,
            'confidence': confidence,
            'estimated_rating': estimated_rating,
            'sentiment_scores': sentiment_scores
        }, None
    except Exception as e:
        return None, str(e)

# Main app
def main():
    st.markdown('<div class="main-header">💊 MedVeritas: Drug Effectiveness Predictor</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a page:",
            ["🏠 Home", "🔍 Drug Comparison", "📝 Effectiveness Predictor", "📊 Data Explorer"]
        )
        
        st.markdown("---")
        st.header("About")
        st.info("""
        MedVeritas uses NLP and Machine Learning to predict drug effectiveness from patient reviews.
        
        **Features:**
        - Compare drugs side-by-side
        - Predict effectiveness from review text
        - Explore dataset insights
        """)
    
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            df = load_processed_data()
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
            else:
                st.error("Could not load data. Please ensure data file exists.")
                return
    
    df = st.session_state.df
    
    if not st.session_state.models_loaded:
        with st.spinner("Loading models..."):
            models = load_models()
            st.session_state.models = models
            st.session_state.models_loaded = True
    
    models = st.session_state.models
    
    if page == "🏠 Home":
        show_home(df)
    elif page == "🔍 Drug Comparison":
        show_drug_comparison(df)
    elif page == "📝 Effectiveness Predictor":
        show_predictor(models)
    elif page == "📊 Data Explorer":
        show_data_explorer(df)

def show_home(df):
    """Home page with overview statistics."""
    st.header("Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    with col2:
        st.metric("Unique Drugs", f"{df['drugName'].nunique():,}")
    with col3:
        st.metric("Unique Conditions", f"{df['condition'].nunique():,}")
    with col4:
        st.metric("Avg Rating", f"{df['rating'].mean():.2f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Effectiveness Distribution")
        effective_count = df['is_effective'].sum()
        not_effective_count = len(df) - effective_count
        st.bar_chart({
            'Effective (≥7)': effective_count,
            'Not Effective (<7)': not_effective_count
        })
    
    with col2:
        st.subheader("Rating Distribution")
        rating_counts = df['rating'].value_counts().sort_index()
        st.bar_chart(rating_counts)
    
    st.markdown("---")
    st.subheader("Top 10 Most Reviewed Drugs")
    top_drugs = df['drugName'].value_counts().head(10)
    st.dataframe(pd.DataFrame({
        'Drug': top_drugs.index,
        'Review Count': top_drugs.values
    }), use_container_width=True)

def show_drug_comparison(df):
    """Drug comparison tool."""
    st.header("🔍 Drug Comparison Tool")
    st.markdown("Compare effectiveness and side effects across different drugs.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        drug1 = st.selectbox(
            "Select Drug 1:",
            options=sorted(df['drugName'].unique()),
            index=0
        )
    
    with col2:
        drug2 = st.selectbox(
            "Select Drug 2:",
            options=sorted(df['drugName'].unique()),
            index=min(1, len(df['drugName'].unique()) - 1)
        )
    
    if drug1 == drug2:
        st.warning("Please select two different drugs for comparison.")
        return
    
    drug1_data = df[df['drugName'] == drug1]
    drug2_data = df[df['drugName'] == drug2]
    
    st.subheader("Comparison Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Drug 1 Avg Rating", f"{drug1_data['rating'].mean():.2f}", 
                 delta=f"{drug1_data['rating'].mean() - drug2_data['rating'].mean():.2f}")
    with col2:
        st.metric("Drug 2 Avg Rating", f"{drug2_data['rating'].mean():.2f}")
    with col3:
        st.metric("Drug 1 Effectiveness", f"{drug1_data['is_effective'].mean()*100:.1f}%",
                 delta=f"{(drug1_data['is_effective'].mean() - drug2_data['is_effective'].mean())*100:.1f}%")
    with col4:
        st.metric("Drug 2 Effectiveness", f"{drug2_data['is_effective'].mean()*100:.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{drug1} Details")
        st.write(f"**Total Reviews:** {len(drug1_data)}")
        st.write(f"**Average Rating:** {drug1_data['rating'].mean():.2f}")
        st.write(f"**Median Rating:** {drug1_data['rating'].median():.2f}")
        st.write(f"**Effectiveness Rate:** {drug1_data['is_effective'].mean()*100:.1f}%")
        
        st.write("**Top Conditions:**")
        top_conditions_1 = drug1_data['condition'].value_counts().head(5)
        for condition, count in top_conditions_1.items():
            st.write(f"- {condition}: {count} reviews")
    
    with col2:
        st.subheader(f"{drug2} Details")
        st.write(f"**Total Reviews:** {len(drug2_data)}")
        st.write(f"**Average Rating:** {drug2_data['rating'].mean():.2f}")
        st.write(f"**Median Rating:** {drug2_data['rating'].median():.2f}")
        st.write(f"**Effectiveness Rate:** {drug2_data['is_effective'].mean()*100:.1f}%")
        
        st.write("**Top Conditions:**")
        top_conditions_2 = drug2_data['condition'].value_counts().head(5)
        for condition, count in top_conditions_2.items():
            st.write(f"- {condition}: {count} reviews")
    
    st.subheader("Rating Distribution Comparison")
    comparison_df = pd.DataFrame({
        drug1: drug1_data['rating'].value_counts().sort_index(),
        drug2: drug2_data['rating'].value_counts().sort_index()
    }).fillna(0)
    st.bar_chart(comparison_df)

def show_predictor(models):
    """Effectiveness predictor from review text."""
    st.header("📝 Effectiveness Predictor")
    st.markdown("Enter a patient review to predict drug effectiveness and estimated rating.")
    
    review_text = st.text_area(
        "Enter patient review:",
        height=200,
        placeholder="Example: I've been taking this medication for 3 months and it has significantly improved my condition. The side effects were minimal and manageable..."
    )
    
    if st.button("Predict Effectiveness", type="primary"):
        if review_text:
            with st.spinner("Analyzing review..."):
                result, error = predict_effectiveness(review_text, models)
                
                if error:
                    st.error(f"Error: {error}")
                elif result:
                    st.success("Prediction Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        effectiveness_status = "✅ Effective" if result['effective'] else "❌ Not Effective"
                        st.metric("Prediction", effectiveness_status)
                    
                    with col2:
                        st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    
                    with col3:
                        st.metric("Estimated Rating", f"{result['estimated_rating']:.1f}/10")
                    
                    st.markdown("---")
                    st.subheader("Sentiment Analysis")
                    
                    sentiment_scores = result['sentiment_scores']
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("VADER Compound", f"{sentiment_scores['vader_compound']:.3f}")
                    with col2:
                        st.metric("VADER Positive", f"{sentiment_scores['vader_pos']:.3f}")
                    with col3:
                        st.metric("VADER Negative", f"{sentiment_scores['vader_neg']:.3f}")
                    with col4:
                        st.metric("TextBlob Polarity", f"{sentiment_scores['textblob_polarity']:.3f}")
        else:
            st.warning("Please enter a review text to predict.")

def show_data_explorer(df):
    """Data exploration page."""
    st.header("📊 Data Explorer")
    st.markdown("Explore the dataset with interactive filters.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_drugs = st.multiselect(
            "Filter by Drug:",
            options=sorted(df['drugName'].unique())[:100],
            default=[]
        )
    
    with col2:
        selected_conditions = st.multiselect(
            "Filter by Condition:",
            options=sorted(df['condition'].unique())[:100],
            default=[]
        )
    
    with col3:
        min_rating = st.slider("Minimum Rating:", 1, 10, 1)
        max_rating = st.slider("Maximum Rating:", 1, 10, 10)
    
    filtered_df = df.copy()
    
    if selected_drugs:
        filtered_df = filtered_df[filtered_df['drugName'].isin(selected_drugs)]
    if selected_conditions:
        filtered_df = filtered_df[filtered_df['condition'].isin(selected_conditions)]
    
    filtered_df = filtered_df[
        (filtered_df['rating'] >= min_rating) & 
        (filtered_df['rating'] <= max_rating)
    ]
    
    st.write(f"**Filtered Results:** {len(filtered_df):,} reviews")
    
    if len(filtered_df) > 0:
        st.subheader("Sample Reviews")
        display_cols = ['drugName', 'condition', 'rating', 'review']
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        st.dataframe(
            filtered_df[available_cols].head(100),
            use_container_width=True,
            height=400
        )
    else:
        st.info("No data matches the selected filters. Please adjust your filters.")

if __name__ == "__main__":
    main()

