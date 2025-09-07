# src/data_preprocessing_fixed.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ChurnDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def load_data(self, filepath):
        """Load the Telco Customer Churn dataset"""
        df = pd.read_csv(filepath)
        return df
    
    def explore_data(self, df):
        """Comprehensive EDA"""
        print("Dataset Shape:", df.shape)
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        print("\nData Types:")
        print(df.dtypes)
        
        print("\nChurn Distribution:")
        print(df['Churn'].value_counts(normalize=True))
        
        return df
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        df_clean = df.copy()
        
        # Convert TotalCharges to numeric, replace empty strings with NaN first
        df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Handle missing values in TotalCharges
        # For customers with 0 tenure, TotalCharges should be equal to MonthlyCharges
        mask_zero_tenure = df_clean['tenure'] == 0
        df_clean.loc[mask_zero_tenure, 'TotalCharges'] = df_clean.loc[mask_zero_tenure, 'MonthlyCharges']
        
        # Fill remaining NaN values with median
        df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)
        
        # Convert binary categorical to numeric
        binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_columns:
            df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
        
        # Convert target variable
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
        
        # Handle 'No internet service' and 'No phone service' values
        internet_dependent_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                  'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        for col in internet_dependent_cols:
            df_clean[col] = df_clean[col].replace('No internet service', 'No')
        
        # Handle MultipleLines
        df_clean['MultipleLines'] = df_clean['MultipleLines'].replace('No phone service', 'No')
        
        return df_clean
    
    def encode_features(self, df):
        """Encode categorical features"""
        categorical_columns = ['gender', 'MultipleLines', 'InternetService', 
                             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                             'TechSupport', 'StreamingTV', 'StreamingMovies',
                             'Contract', 'PaymentMethod']
        
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        print("After encoding - checking for missing values:")
        print(df_encoded.isnull().sum().sum())
        
        return df_encoded
