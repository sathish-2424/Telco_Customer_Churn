# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ChurnPredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = None
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        
        print("Models trained successfully!")
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            auc = roc_auc_score(y_test, y_pred_proba)
            results[name] = {
                'AUC': auc,
                'Classification Report': classification_report(y_test, y_pred),
                'Predictions': y_pred,
                'Probabilities': y_pred_proba
            }
            
            print(f"\n{name} Results:")
            print(f"AUC: {auc:.4f}")
            print("Classification Report:")
            print(results[name]['Classification Report'])
        
        # Find best model based on AUC
        best_model_name = max(results.keys(), key=lambda k: results[k]['AUC'])
        self.best_model = self.models[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        return results, best_model_name
    
    def plot_feature_importance(self, X, model_name=None):
        """Plot feature importance"""
        if model_name:
            model = self.models[model_name]
        else:
            model = self.best_model
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(f'Top 15 Feature Importance - {model_name or "Best Model"}')
            plt.tight_layout()
            plt.show()
            
            self.feature_importance = importance_df
            return importance_df
    
    def save_model(self, model_path='models/best_churn_model.pkl'):
        """Save the best model"""
        if self.best_model:
            joblib.dump(self.best_model, model_path)
            print(f"Best model saved to {model_path}")
