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
│   ├── 01_EDA.ipynb           # Comprehensive EDA and modeling notebook
│   └── results/                # Notebook outputs (figures, metrics)
│
├── src/
│   ├── data/
│   │   └── processing.py      # Data loading and cleaning functions
│   ├── features/
│   │   └── engineering.py      # Feature creation utilities
│   ├── models/
│   │   └── train.py            # Model training and evaluation
│   ├── nlp/
│   │   └── utils.py            # NLP helper functions
│   └── visualization/
│       └── plots.py            # Plotting and visualization functions
│
├── app.py                      # Streamlit interactive dashboard
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
├── PRESENTATION_SUMMARY.md    # Presentation slides summary
├── EDA_SUMMARY.md             # EDA implementation summary
└── LICENSE                    # Project license
```

## Project Status

✅ **Phase 1: Data Preprocessing & Exploration** - Complete  
✅ **Phase 2: Feature Engineering & Modeling** - Complete  
✅ **Phase 3: Analysis & Visualization** - Complete  
✅ **Phase 4: Refinement & Presentation** - Complete  

### Completed Deliverables

- ✅ Comprehensive EDA notebook with all analyses
- ✅ Trained classification and regression models (XGBoost best performer)
- ✅ 13+ visualizations (word clouds, heatmaps, ROC curves, topic modeling, etc.)
- ✅ Interactive Streamlit dashboard for drug comparison and prediction
- ✅ Comprehensive analysis report
- ✅ Presentation summary document
- ✅ All code documented and organized

### Model Performance Summary

**Classification (Effective vs Not Effective):**
- XGBoost: 78.58% accuracy, 86.28% ROC-AUC

**Regression (Predict Exact Rating):**
- XGBoost: RMSE 2.63, R² 0.43

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

5. **Dataset**
- The processed dataset is available in `data/processed/medVe_data_final_version.xlsx`
- Original datasets can be downloaded from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018) or [UCI](https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com)

## Usage

### Running the EDA Notebook

1. **Open Jupyter Notebook**
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

2. **Run all cells** to:
   - Load and clean data
   - Perform feature engineering
   - Train models
   - Generate visualizations
   - Save results

### Running the Interactive Dashboard

1. **Start Streamlit app**
```bash
streamlit run app.py
```

2. **Access dashboard** at `http://localhost:8501`

3. **Features available:**
   - **Home**: Overview statistics and data summary
   - **Drug Comparison**: Side-by-side comparison of drugs
   - **Effectiveness Predictor**: Input review text to predict effectiveness
   - **Data Explorer**: Interactive data exploration with filters

### Using Trained Models

Models are saved in `models/` directory:
- Classification models: `models/classification/`
- Regression models: `models/regression/`

Load and use models:
```python
import pickle

# Load classification model
with open('models/classification/xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Use model for predictions
predictions = model.predict(X_test)
```

## Project Outputs

### Visualizations
All visualizations are saved in `results/figures/`:
- Rating distributions by drug/condition
- Word clouds (positive vs negative)
- Correlation heatmaps
- Feature importance charts
- Topic modeling results
- Confusion matrices and ROC curves
- Side effects analysis
- And more...

### Reports
- **Comprehensive Analysis Report**: `results/reports/COMPREHENSIVE_ANALYSIS_REPORT.md`
- **Presentation Summary**: `PRESENTATION_SUMMARY.md`
- **EDA Summary**: `EDA_SUMMARY.md`

### Metrics
Model performance metrics saved in:
- `results/metrics/classification_results.csv`
- `results/metrics/regression_results.csv`

## Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- Drugs.com for the original patient reviews
- Data@NEU Snowball Program for project support
