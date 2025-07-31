-- Create database and select it
CREATE DATABASE IF NOT EXISTS churn_analysis;
USE churn_analysis;

-- Create TelcoChurn table
CREATE TABLE IF NOT EXISTS TelcoChurn (
    customerID VARCHAR(50),
    tenure INT,
    MonthlyCharges DECIMAL(10,2),
    TotalCharges DECIMAL(10,2),
    gender_Male TINYINT(1),
    SeniorCitizen_Yes TINYINT(1),
    Partner_Yes TINYINT(1),
    Dependents_Yes TINYINT(1),
    MultipleLines_Yes TINYINT(1),
    InternetService_Fiber_optic TINYINT(1),
    InternetService_No TINYINT(1),
    OnlineSecurity_Yes TINYINT(1),
    OnlineBackup_Yes TINYINT(1),
    DeviceProtection_Yes TINYINT(1),
    TechSupport_Yes TINYINT(1),
    StreamingTV_Yes TINYINT(1),
    StreamingMovies_Yes TINYINT(1),
    Contract_One_year TINYINT(1),
    Contract_Two_year TINYINT(1),
    PaperlessBilling_Yes TINYINT(1),
    PaymentMethod_Credit_card_automatic TINYINT(1),
    PaymentMethod_Electronic_check TINYINT(1),
    PaymentMethod_Mailed_check TINYINT(1),
    Churn TINYINT(1)
);

-- Enable local infile to allow data import (run as admin)
SET GLOBAL local_infile = 1;

-- Load data from CSV (update path accordingly)
LOAD DATA LOCAL INFILE 'C:/path/to/telco_churn_cleaned.csv'
INTO TABLE TelcoChurn
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;

-- Sample queries

-- Count total rows
SELECT COUNT(*) AS total_rows FROM TelcoChurn;

-- Sample 10 rows
SELECT * FROM TelcoChurn LIMIT 10;

-- Select churned customers
SELECT * FROM TelcoChurn WHERE Churn = 1;

-- Average monthly charges by churn status
SELECT Churn, AVG(MonthlyCharges) AS AvgMonthlyCharge
FROM TelcoChurn
GROUP BY Churn;

-- Churn rate by contract type
SELECT Contract_One_year, Contract_Two_year, AVG(Churn) AS ChurnRate
FROM TelcoChurn
GROUP BY Contract_One_year, Contract_Two_year;

-- Overall churn percentage
SELECT (SUM(Churn) / COUNT(*)) * 100 AS ChurnRatePercent
FROM TelcoChurn;
