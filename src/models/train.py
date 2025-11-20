"""
Model training and evaluation utilities for MedVeritas project.
Handles classification and regression model training.
"""

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, roc_curve, confusion_matrix
)
from xgboost import XGBClassifier, XGBRegressor
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


def ensure_dir(directory: str):
    """Ensure directory exists."""
    directory = os.path.normpath(directory)
    os.makedirs(directory, exist_ok=True)


class ModelTrainer:
    """Model training and evaluation class."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
    
    def train_classification_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                   X_test: pd.DataFrame, y_test: pd.Series,
                                   models: Optional[dict] = None) -> dict:
        """
        Train multiple classification models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            models: Dictionary of model names and model objects (optional)
        
        Returns:
            Dictionary of model results
        """
        print("Training classification models...")
        
        if models is None:
            models = {
                'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'Random Forest': RandomForestClassifier(
                    n_estimators=50,
                    max_depth=20,
                    max_samples=0.5,
                    random_state=self.random_state, 
                    n_jobs=-1,
                    verbose=1
                ),
                'XGBoost': XGBClassifier(random_state=self.random_state, n_jobs=-1, eval_metric='logloss')
            }
        
        results = {}
        roc_data = []
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            from scipy.sparse import csr_matrix, hstack
            
            X_train_dense = X_train.copy()
            X_test_dense = X_test.copy()
            
            for col in X_train_dense.columns:
                if pd.api.types.is_sparse(X_train_dense[col].dtype):
                    X_train_dense[col] = X_train_dense[col].sparse.to_dense()
                if pd.api.types.is_sparse(X_test_dense[col].dtype):
                    X_test_dense[col] = X_test_dense[col].sparse.to_dense()
            
            tfidf_cols = [col for col in X_train_dense.columns if col.startswith('tfidf_')]
            dense_cols = [col for col in X_train_dense.columns if not col.startswith('tfidf_')]
            
            if tfidf_cols:
                X_train_tfidf_array = X_train_dense[tfidf_cols].select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
                X_test_tfidf_array = X_test_dense[tfidf_cols].select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
                X_train_tfidf = csr_matrix(X_train_tfidf_array)
                X_test_tfidf = csr_matrix(X_test_tfidf_array)
                
                if dense_cols:
                    X_train_dense_array = X_train_dense[dense_cols].select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
                    X_test_dense_array = X_test_dense[dense_cols].select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
                    X_train_sparse = hstack([X_train_tfidf, csr_matrix(X_train_dense_array)])
                    X_test_sparse = hstack([X_test_tfidf, csr_matrix(X_test_dense_array)])
                else:
                    X_train_sparse = X_train_tfidf
                    X_test_sparse = X_test_tfidf
            else:
                X_train_numeric = X_train_dense.select_dtypes(include=[np.number])
                X_test_numeric = X_test_dense.select_dtypes(include=[np.number])
                X_train_sparse = csr_matrix(X_train_numeric.to_numpy(dtype=np.float64))
                X_test_sparse = csr_matrix(X_test_numeric.to_numpy(dtype=np.float64))
            
            if 'Random Forest' in name or 'RandomForest' in str(type(model)):
                print(f"  Converting sparse matrix to dense for {name} (this may take a moment)...")
                X_train_final = X_train_sparse.toarray() if hasattr(X_train_sparse, 'toarray') else X_train_sparse
                X_test_final = X_test_sparse.toarray() if hasattr(X_test_sparse, 'toarray') else X_test_sparse
            else:
                X_train_final = X_train_sparse
                X_test_final = X_test_sparse
            
            model.fit(X_train_final, y_train)
            y_pred = model.predict(X_test_final)
            y_pred_proba = model.predict_proba(X_test_final)[:, 1] if hasattr(model, 'predict_proba') else None
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': auc,
                'predictions': y_pred,
                'predictions_proba': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            if auc:
                print(f"  ROC-AUC: {auc:.4f}")
            
            # ROC curve data
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_data.append((name, fpr, tpr, auc))
        
        self.models['classification'] = results
        self.results['classification'] = results
        self.results['roc_data'] = roc_data
        
        return results
    
    def train_regression_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series,
                               models: Optional[dict] = None) -> dict:
        """
        Train multiple regression models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            models: Dictionary of model names and model objects (optional)
        
        Returns:
            Dictionary of model results
        """
        print("Training regression models...")
        
        if models is None:
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(
                    n_estimators=50,
                    max_depth=20,
                    max_samples=0.5,
                    random_state=self.random_state, 
                    n_jobs=-1,
                    verbose=1
                ),
                'XGBoost': XGBRegressor(random_state=self.random_state, n_jobs=-1)
            }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            from scipy.sparse import csr_matrix, hstack
            
            X_train_dense = X_train.copy()
            X_test_dense = X_test.copy()
            
            for col in X_train_dense.columns:
                if pd.api.types.is_sparse(X_train_dense[col].dtype):
                    X_train_dense[col] = X_train_dense[col].sparse.to_dense()
                if pd.api.types.is_sparse(X_test_dense[col].dtype):
                    X_test_dense[col] = X_test_dense[col].sparse.to_dense()
            
            tfidf_cols = [col for col in X_train_dense.columns if col.startswith('tfidf_')]
            dense_cols = [col for col in X_train_dense.columns if not col.startswith('tfidf_')]
            
            if tfidf_cols:
                X_train_tfidf_array = X_train_dense[tfidf_cols].select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
                X_test_tfidf_array = X_test_dense[tfidf_cols].select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
                X_train_tfidf = csr_matrix(X_train_tfidf_array)
                X_test_tfidf = csr_matrix(X_test_tfidf_array)
                
                if dense_cols:
                    X_train_dense_array = X_train_dense[dense_cols].select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
                    X_test_dense_array = X_test_dense[dense_cols].select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
                    X_train_sparse = hstack([X_train_tfidf, csr_matrix(X_train_dense_array)])
                    X_test_sparse = hstack([X_test_tfidf, csr_matrix(X_test_dense_array)])
                else:
                    X_train_sparse = X_train_tfidf
                    X_test_sparse = X_test_tfidf
            else:
                X_train_numeric = X_train_dense.select_dtypes(include=[np.number])
                X_test_numeric = X_test_dense.select_dtypes(include=[np.number])
                X_train_sparse = csr_matrix(X_train_numeric.to_numpy(dtype=np.float64))
                X_test_sparse = csr_matrix(X_test_numeric.to_numpy(dtype=np.float64))
            
            if 'Random Forest' in name or 'RandomForest' in str(type(model)):
                print(f"  Converting sparse matrix to dense for {name} (this may take a moment)...")
                X_train_final = X_train_sparse.toarray() if hasattr(X_train_sparse, 'toarray') else X_train_sparse
                X_test_final = X_test_sparse.toarray() if hasattr(X_test_sparse, 'toarray') else X_test_sparse
            else:
                X_train_final = X_train_sparse
                X_test_final = X_test_sparse
            
            model.fit(X_train_final, y_train)
            y_pred = model.predict(X_test_final)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'predictions': y_pred
            }
            
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R² Score: {r2:.4f}")
        
        self.models['regression'] = results
        self.results['regression'] = results
        
        return results
    
    def get_feature_importance(self, model_name: str, feature_names: list, 
                               model_type: str = 'classification') -> dict:
        """
        Get feature importance from trained model.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            model_type: Type of model ('classification' or 'regression')
        
        Returns:
            Dictionary of feature importance
        """
        if model_type not in self.models:
            return {}
        
        if model_name not in self.models[model_type]:
            return {}
        
        model = self.models[model_type][model_name]['model']
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            return {}
        
        # Create dictionary
        feature_importance = dict(zip(feature_names, importances))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return dict(sorted_features)
    
    def save_model(self, model_name: str, model_type: str, output_dir: str = 'models'):
        """
        Save trained model to disk.
        
        Args:
            model_name: Name of the model
            model_type: Type of model ('classification' or 'regression')
            output_dir: Output directory
        """
        if model_type not in self.models:
            print(f"No {model_type} models found.")
            return
        
        if model_name not in self.models[model_type]:
            print(f"Model '{model_name}' not found.")
            return
        
        output_dir = os.path.normpath(output_dir)
        ensure_dir(output_dir)
        ensure_dir(os.path.join(output_dir, model_type))
        
        model = self.models[model_type][model_name]['model']
        filepath = os.path.normpath(os.path.join(output_dir, model_type, f'{model_name.lower().replace(" ", "_")}.pkl'))
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Saved model: {filepath}")
    
    def save_results(self, output_dir: str = 'results/metrics'):
        """
        Save model results to CSV.
        
        Args:
            output_dir: Output directory
        """
        output_dir = os.path.normpath(output_dir)
        ensure_dir(output_dir)
        
        # Classification results
        if 'classification' in self.results:
            classification_results = []
            for name, metrics in self.results['classification'].items():
                classification_results.append({
                    'Model': name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1_Score': metrics['f1_score'],
                    'ROC_AUC': metrics['roc_auc']
                })
            
            df_class = pd.DataFrame(classification_results)
            filepath = os.path.normpath(os.path.join(output_dir, 'classification_results.csv'))
            df_class.to_csv(filepath, index=False)
            print(f"Saved classification results: {filepath}")
        
        # Regression results
        if 'regression' in self.results:
            regression_results = []
            for name, metrics in self.results['regression'].items():
                regression_results.append({
                    'Model': name,
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'R2_Score': metrics['r2_score']
                })
            
            df_reg = pd.DataFrame(regression_results)
            filepath = os.path.normpath(os.path.join(output_dir, 'regression_results.csv'))
            df_reg.to_csv(filepath, index=False)
            print(f"Saved regression results: {filepath}")

