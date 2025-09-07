# create_directories_fixed.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('src', exist_ok=True)

print("✅ Directories created successfully!")

# Create sample data for demo (all numeric values)
np.random.seed(42)
n_samples = 1000

sample_data = {
    'customerID': [f'ID_{i}' for i in range(n_samples)],
    'gender': np.random.choice([0, 1], n_samples),  # 0=Male, 1=Female
    'SeniorCitizen': np.random.choice([0, 1], n_samples),
    'Partner': np.random.choice([0, 1], n_samples),
    'Dependents': np.random.choice([0, 1], n_samples),
    'tenure': np.random.randint(0, 73, n_samples),
    'PhoneService': np.random.choice([0, 1], n_samples),
    'MultipleLines': np.random.choice([0, 1, 2], n_samples),  # 0=No, 1=Yes, 2=No phone
    'InternetService': np.random.choice([0, 1, 2], n_samples),  # 0=DSL, 1=Fiber, 2=No
    'OnlineSecurity': np.random.choice([0, 1, 2], n_samples),
    'OnlineBackup': np.random.choice([0, 1, 2], n_samples),
    'DeviceProtection': np.random.choice([0, 1, 2], n_samples),
    'TechSupport': np.random.choice([0, 1, 2], n_samples),
    'StreamingTV': np.random.choice([0, 1, 2], n_samples),
    'StreamingMovies': np.random.choice([0, 1, 2], n_samples),
    'Contract': np.random.choice([0, 1, 2], n_samples),  # 0=Month-to-month, 1=One year, 2=Two year
    'PaperlessBilling': np.random.choice([0, 1], n_samples),
    'PaymentMethod': np.random.choice([0, 1, 2, 3], n_samples),
    'MonthlyCharges': np.round(np.random.uniform(18.25, 118.75, n_samples), 2),
    'TotalCharges': np.round(np.random.uniform(18.8, 8684.8, n_samples), 2),
    'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])  # 27% churn rate
}

# Create sample processed data
df = pd.DataFrame(sample_data)
df.to_csv('data/processed_churn_data.csv', index=False)
print("✅ Sample data created!")

# Create and save a simple model
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Make sure all data is numeric
print("Data types before training:")
print(X.dtypes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Test the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, 'models/best_churn_model.pkl')
print("✅ Sample model created and saved!")

print(f"\nModel expects {model.n_features_in_} features:")
print("Feature names:", list(X.columns))

print("\n🚀 Now you can run: streamlit run app.py")
