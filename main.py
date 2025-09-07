# main_fixed.py
from src.data_preprocessing import ChurnDataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ChurnPredictor
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 Starting Customer Churn Prediction Project")
    
    # Initialize classes
    processor = ChurnDataProcessor()
    engineer = FeatureEngineer()
    predictor = ChurnPredictor()
    
    # Load and explore data
    print("\n📊 Loading and exploring data...")
    try:
        df = processor.load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df = processor.explore_data(df)
    except FileNotFoundError:
        print("❌ Telco dataset not found. Please download it from Kaggle.")
        print("Using sample data instead...")
        # Create sample data as fallback
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'customerID': [f'ID_{i}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(0, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
            'MonthlyCharges': np.round(np.random.uniform(18.25, 118.75, n_samples), 2),
            'TotalCharges': np.round(np.random.uniform(18.8, 8684.8, n_samples), 2).astype(str),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
        })
        processor.explore_data(df)
    
    # Clean data
    print("\n🧹 Cleaning data...")
    df_clean = processor.clean_data(df)
    print("Data cleaned successfully!")
    
    # Encode features
    print("\n🔤 Encoding categorical features...")
    df_encoded = processor.encode_features(df_clean)
    print("Features encoded successfully!")
    
    # Feature engineering
    print("\n⚙️ Engineering features...")
    df_features = engineer.create_features(df_encoded)
    print("Features engineered successfully!")
    
    # Select features
    print("\n🎯 Selecting features...")
    X, y = engineer.select_features(df_features)
    print(f"Selected {X.shape[1]} features for {X.shape[0]} samples")
    
    # Prepare data
    print("\n🔄 Preparing data for modeling...")
    X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train models
    print("\n🤖 Training models...")
    models = predictor.train_models(X_train, y_train)
    print("Models trained successfully!")
    
    # Evaluate models
    print("\n📈 Evaluating models...")
    results, best_model_name = predictor.evaluate_models(X_test, y_test)
    
    # Feature importance
    print("\n🎯 Analyzing feature importance...")
    predictor.plot_feature_importance(X, best_model_name)
    
    # Save model
    print("\n💾 Saving best model...")
    predictor.save_model()
    
    # Save processed data
    print("\n💾 Saving processed data...")
    df_features.to_csv('data/processed_churn_data.csv', index=False)
    
    print("\n✅ Project completed successfully!")
    print(f"Best model: {best_model_name}")
    print(f"Best AUC: {results[best_model_name]['AUC']:.4f}")

if __name__ == "__main__":
    main()