# 📊 Telco Customer Churn Analysis & Prediction

## Overview

This project analyzes the Telco Customer Churn dataset to uncover churn drivers, build predictive models, and suggest actionable business strategies that can help reduce customer attrition. It covers the full data analytics pipeline—from cleaning, EDA, and SQL queries to advanced modeling and dashboard visualization.

## 📁 Data

- **Dataset:** `WA_Fn-UseC_-Telco-Customer-Churn.csv` *(public, from Kaggle)*
- **Cleaned Dataset:** `telco_churn_cleaned.csv`
- **Features:** Customer demographics, tenure, service subscriptions, charges, contract/payment data, churn label.

## 🚀 Project Structure

```
project/
│
├── data/
│    ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│    └── telco_churn_cleaned.csv
├── notebooks/
│    ├── 1_data_cleaning_preprocessing.ipynb
│    ├── 2_exploratory_data_analysis.ipynb
│    ├── 3_predictive_modeling.ipynb
│    ├── 4_sql_queries.sql
│    └── 5_Visualization_Dashboard.ipynb
├── README.md
└── requirements.txt
```

## 1. Data Cleaning & Preprocessing

- Remove duplicates and handle blanks/missing values (especially `TotalCharges`).
- Convert data types; standardize binary columns.
- Normalize service feature columns (replace "No internet service" with "No").
- Encode categorical features using one-hot encoding.
- Optionally remove top 1% outliers in charges/tenure.
- Save the processed results as `telco_churn_cleaned.csv`.

**Run:**  
`python notebooks/1_data_cleaning_preprocessing.ipynb`

## 2. Exploratory Data Analysis (EDA)

- Visualize churn rate and class balance.
- Explore distributions of tenure and charges by churn status.
- Analyze churn by contract, demographics, and service use.
- Identify high-risk segments.
- Correlation heatmaps to spot relationships between features.

**Run:**  
`python notebooks/2_exploratory_data_analysis.ipynb`

## 3. SQL Querying (Optional)

- Run descriptive and diagnostic queries for insights such as average charges by churn, churn rate by contract, tenure trends, and customer segmentation.
- Example queries in `notebooks/4_sql_queries.sql`.
- Import CSV into MySQL or another RDBMS for analysis.

## 4. Predictive Modeling

- 💡 **Train-test split** and (optional) feature scaling.
- **Models:** Logistic Regression (baseline), Decision Tree, Random Forest, XGBoost (advanced).
- Evaluate using accuracy, precision, recall, F1-score, ROC AUC.
- Extract and plot feature importance for business action.

**Run:**  
`python notebooks/3_predictive_modeling.ipynb`

## 5. Visualization Dashboard

- Build a dashboard in **Power BI** or Tableau.
- Core visuals:
    - Churn rate (KPI)
    - Churn by contract, demographic, and services
    - Box/violin/histogram plots for charges and tenure by churn
    - Feature importance chart (from modeling)
- Add slicers and interactive filters for deeper analysis.
- Example included: `notebooks/5_dashboard_powerbi.pbx`

## 6. Business Recommendations

- Focus retention strategies on high-churn segments (month-to-month, high charges, short tenure).
- Targeted offers for at-risk groups (senior citizens, customers with no add-on services).
- Use model predictions to inform personalized campaigns.

## Requirements

- **Python ≥3.7**
- Packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost` (see `requirements.txt`)
- (Optional) MySQL or SQLite for SQL analysis
- Power BI or Tableau for dashboarding

## How to Use

1. Download the original dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
2. Run the data cleaning notebook to generate the cleaned dataset.
3. Explore data and visualize insights in the EDA notebook.
4. Optionally import cleaned data to SQL for further analysis.
5. Build and evaluate predictive models in the modeling notebook.
6. Visualize final findings and insights in Power BI or Tableau.
7. Use business recommendations in reporting/presentations.

## Credits

- **Dataset:** [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Project workflow, modeling, and dashboarding: [Your Name]

## Key Highlights

- End-to-end real-world workflow, ready for interviews or presentations.
- Covers data wrangling, analytics, SQL, modeling, and business communication.
- Easily extensible to other churn or customer retention projects.

**Questions or customizations? Just ask in the issues or comments!**
