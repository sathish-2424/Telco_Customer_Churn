# src/feature_engineering_fixed.py
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass
    
    def create_features(self, df):
        """Create advanced features with proper NaN handling"""
        df_features = df.copy()
        
        # Ensure no division by zero and handle edge cases
        df_features['tenure'] = df_features['tenure'].fillna(0)
        df_features['MonthlyCharges'] = df_features['MonthlyCharges'].fillna(df_features['MonthlyCharges'].median())
        df_features['TotalCharges'] = df_features['TotalCharges'].fillna(df_features['TotalCharges'].median())
        
        # Total spending (already exists as TotalCharges, but let's verify)
        df_features['TotalSpending'] = df_features['MonthlyCharges'] * df_features['tenure']
        
        # Average monthly spending (handle division by zero)
        df_features['AvgMonthlySpending'] = np.where(
            df_features['tenure'] > 0,
            df_features['TotalCharges'] / df_features['tenure'],
            df_features['MonthlyCharges']
        )
        
        # Customer lifetime value
        df_features['CLV'] = df_features['MonthlyCharges'] * df_features['tenure']
        
        # Service usage score
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Ensure service columns are numeric
        for col in service_cols:
            if col in df_features.columns:
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)
        
        df_features['ServiceUsageScore'] = df_features[service_cols].sum(axis=1)
        
        # Tenure groups (create dummy variables instead of categorical)
        df_features['Tenure_0_12'] = (df_features['tenure'] <= 12).astype(int)
        df_features['Tenure_13_24'] = ((df_features['tenure'] > 12) & (df_features['tenure'] <= 24)).astype(int)
        df_features['Tenure_25_48'] = ((df_features['tenure'] > 24) & (df_features['tenure'] <= 48)).astype(int)
        df_features['Tenure_48_plus'] = (df_features['tenure'] > 48).astype(int)
        
        # Monthly charges groups
        charges_q25 = df_features['MonthlyCharges'].quantile(0.25)
        charges_q75 = df_features['MonthlyCharges'].quantile(0.75)
        
        df_features['Charges_Low'] = (df_features['MonthlyCharges'] <= charges_q25).astype(int)
        df_features['Charges_High'] = (df_features['MonthlyCharges'] >= charges_q75).astype(int)
        
        # Contract risk score (numeric)
        contract_risk_map = {0: 3, 1: 2, 2: 1}  # Assuming 0=Month-to-month, 1=One year, 2=Two year
        df_features['ContractRisk'] = df_features['Contract'].map(contract_risk_map).fillna(2)
        
        # Senior citizen family interaction
        df_features['SeniorCitizenFamily'] = (
            df_features['SeniorCitizen'] * (df_features['Partner'] + df_features['Dependents'])
        )
        
        # Payment method risk (Electronic check is typically riskier)
        # Create binary indicators for payment methods
        payment_dummies = pd.get_dummies(df_features['PaymentMethod'], prefix='Payment')
        df_features = pd.concat([df_features, payment_dummies], axis=1)
        
        # Internet service indicators
        internet_dummies = pd.get_dummies(df_features['InternetService'], prefix='Internet')
        df_features = pd.concat([df_features, internet_dummies], axis=1)
        
        # Check for any remaining NaN values
        print("NaN values after feature engineering:")
        nan_cols = df_features.columns[df_features.isnull().any()].tolist()
        if nan_cols:
            print(f"Columns with NaN: {nan_cols}")
            # Fill any remaining NaN values
            for col in nan_cols:
                if df_features[col].dtype in ['float64', 'int64']:
                    df_features[col].fillna(df_features[col].median(), inplace=True)
                else:
                    df_features[col].fillna(df_features[col].mode()[0], inplace=True)
        else:
            print("No NaN values found!")
        
        return df_features
    
    def select_features(self, df, target_col='Churn'):
        """Select relevant features for modeling"""
        # Drop unnecessary columns
        drop_cols = ['customerID']
        
        # Get all columns except target and drop columns
        feature_cols = [col for col in df.columns if col not in drop_cols + [target_col]]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Final check for any NaN or infinite values
        print(f"Final feature matrix shape: {X.shape}")
        print(f"NaN values in X: {X.isnull().sum().sum()}")
        print(f"Infinite values in X: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")
        
        # Replace any infinite values with NaN then fill with median
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col].fillna(X[col].median(), inplace=True)
        
        return X, y
