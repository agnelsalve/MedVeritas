# MedVeritas: Drug Effectiveness Prediction from Patient Reviews

Using NLP and Machine Learning to Analyze Patient-Reported Outcomes

## Overview

MedVeritas is a data science project that leverages Natural Language Processing (NLP) and Machine Learning to predict drug effectiveness from patient-written reviews. With rising healthcare costs and medication non-adherence affecting 50% of patients, understanding what makes a drug "effective" from the patient perspective is crucial. This project analyzes over 215,000 patient reviews to extract actionable insights that could help patients make informed medication decisions.

## Problem Statement

Can we predict drug effectiveness from patient-written reviews, and what features (textual sentiment, drug characteristics, patient demographics) are most predictive of treatment success?

### Key Questions

- What words and phrases most strongly correlate with effective vs. ineffective drugs?
- Can sentiment analysis of reviews predict numerical effectiveness ratings?
- What are the most common side effects mentioned for different drug categories?
- How do patient ratings vary across different medical conditions?

## Dataset

**Primary Source**: [UCI Drug Review Dataset](https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com) / [Kaggle Alternative](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018)

**Dataset Characteristics**:
- 215,063 patient reviews
- 3,436 unique drugs
- 885 different medical conditions
- Time period: 2008-2017
- Average review length: ~100 words

**Key Features**:
- `drugName`: Name of the medication
- `condition`: Medical condition being treated
- `review`: Patient-written text review
- `rating`: 10-star patient satisfaction rating
- `date`: Review submission date
- `usefulCount`: Community validation metric

## Methodology

### Phase 1: Data Preprocessing & Exploration
- Data cleaning and validation
- Exploratory data analysis (EDA)
- Text preprocessing: tokenization, stopword removal, lemmatization
- Initial sentiment analysis
- Train-test split (80-20)

### Phase 2: Feature Engineering
**Text Features**:
- TF-IDF vectorization (unigrams and bigrams)
- Sentiment scores (VADER/TextBlob)
- Review length and readability metrics
- Keyword presence indicators

**Categorical Features**:
- Drug name and condition encoding
- Temporal features (year, month)

**Derived Features**:
- Social validation metrics (useful count)
- Drug popularity indicators

### Phase 3: Modeling

**Classification Task**: Predict effective (rating ≥ 7) vs not effective (rating < 7)
- Logistic Regression (baseline)
- Random Forest Classifier
- XGBoost Classifier

**Regression Task**: Predict exact rating (1-10)
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting models

**Evaluation Metrics**:
- Classification: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Regression: RMSE, MAE, R² score

### Phase 4: NLP Deep Dive
- Topic Modeling using Latent Dirichlet Allocation (LDA)
- Word importance analysis for high vs low ratings
- Side effect pattern extraction
- Feature importance analysis using SHAP values

### Phase 5: Insights & Interpretation
- Model comparison and selection
- Feature importance ranking
- Comparative analysis across drug categories
- Actionable recommendations

## Technologies

**Programming**: Python 3.10+

**Core Libraries**:
- **Data Processing**: pandas, numpy
- **NLP**: nltk, spacy, gensim, textblob/vaderSentiment
- **Machine Learning**: scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn, plotly, wordcloud
- **Dashboard** (optional): Streamlit

**Development Environment**: Google Colab / VS Code, GitHub

## Project Structure

```
medveritas/
│
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and preprocessed data
│   └── features/               # Engineered features
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory data analysis
│   ├── 02_preprocessing.ipynb # Data cleaning and preprocessing
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb      # Model training and evaluation
│   └── 05_nlp_analysis.ipynb  # Topic modeling and text analysis
│
├── src/
│   ├── data_processing.py     # Data loading and cleaning functions
│   ├── feature_engineering.py # Feature creation utilities
│   ├── models.py              # Model training and evaluation
│   ├── nlp_utils.py           # NLP helper functions
│   └── visualization.py       # Plotting and dashboard functions
│
├── models/
│   ├── classification/        # Saved classification models
│   └── regression/            # Saved regression models
│
├── results/
│   ├── figures/               # Generated visualizations
│   ├── reports/               # Analysis reports
│   └── metrics/               # Model performance metrics
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── LICENSE                    # Project license
```

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/medveritas.git
cd medveritas
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (first time only)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

5. **Download dataset**
- Download from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018) or [UCI](https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com)
- Place CSV files in `data/raw/` directory

## Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- Drugs.com for the original patient reviews
- Data@NEU Snowball Program for project support
