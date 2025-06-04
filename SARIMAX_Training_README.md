# SARIMAX Model Training Guide

## Overview

This document explains how we trained our SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) model for ridership prediction. The model incorporates seasonal patterns and external factors to forecast public transport ridership.

## Model Architecture

### SARIMAX Configuration

- **Model Type**: SARIMAX(1, 1, 1)x(1, 0, 1, 24)
- **ARIMA Components**: (p=1, d=1, q=1)
  - AR(1): Autoregressive term of order 1
  - I(1): First-order differencing for stationarity
  - MA(1): Moving average term of order 1
- **Seasonal Components**: (P=1, D=0, Q=1, S=24)
  - Seasonal period of 24 hours (daily seasonality)
  - Seasonal AR(1) and MA(1) terms

### Target Variable

- **Dependent Variable**: Hourly ridership aggregated across all routes
- **Data Aggregation**: Raw data aggregated to hourly totals for temporal modeling

## Data Preparation

### 1. Data Loading and Cleaning

```python
# Load cleaned dataset
data_path = 'model/data/cleaned_data.csv'
komuter_data = pd.read_csv(data_path)

# Clean column names and string data
komuter_data.columns = komuter_data.columns.str.strip()
for col in komuter_data.select_dtypes(include='object').columns:
    komuter_data[col] = komuter_data[col].str.strip()
```

### 2. Temporal Aggregation

The raw ridership data is aggregated from individual trip records to hourly totals:

```python
# Create datetime index
komuter_data['datetime'] = pd.to_datetime(komuter_data['date'] + ' ' + komuter_data['time'])

# Aggregate by hour
hourly_data = komuter_data.groupby('datetime').agg({
    'ridership': 'sum',                    # Total ridership per hour
    'day_of_week': 'first',               # Consistent temporal features
    'is_weekend': 'first',
    'is_holiday': 'first'
}).reset_index()
```

### 3. Feature Engineering

Additional time-based features are created to capture temporal patterns:

```python
# Time-based features
hourly_data['hour_of_day'] = hourly_data['datetime'].dt.hour
hourly_data['month'] = hourly_data['datetime'].dt.month
hourly_data['day_of_month'] = hourly_data['datetime'].dt.day
```

## Exogenous Variables (External Factors)

The model incorporates several external variables that influence ridership patterns:

| Variable      | Type        | Description                      | Significance                            |
| ------------- | ----------- | -------------------------------- | --------------------------------------- |
| `day_of_week` | Categorical | Day of week (1=Monday, 7=Sunday) | Not significant (p=0.744)               |
| `is_weekend`  | Binary      | Weekend indicator                | **Highly significant** (p<0.001)        |
| `is_holiday`  | Binary      | Holiday indicator                | **Highly significant** (p<0.001)        |
| `month`       | Continuous  | Month of year                    | Marginally significant (p=0.084)        |
| `hour_of_day` | Continuous  | Hour of day (0-23)               | Used in training but not in final model |

### Key Findings from Exogenous Variables:

- **Weekend Effect**: Strong negative impact (-541 units), indicating reduced ridership on weekends
- **Holiday Effect**: Positive impact (+389 units), suggesting increased ridership during holidays
- **Seasonal Effect**: Monthly variations show marginal significance

## Training Process

### 1. Data Splitting

```python
# 80/20 split for training and testing
train_size = int(len(hourly_data) * 0.8)
train_data = hourly_data[:train_size]
test_data = hourly_data[train_size:]
```

**Training Data**: 106 observations (80% of total)
**Testing Data**: Remaining 20% for validation

### 2. Model Configuration

```python
# SARIMAX parameters
p, d, q = 1, 1, 1        # Non-seasonal ARIMA order
P, D, Q, S = 1, 0, 1, 24 # Seasonal ARIMA order

# Model initialization
model = SARIMAX(
    train_data['ridership'],
    exog=train_data[exog_vars],
    order=(p, d, q),
    seasonal_order=(P, D, Q, S),
    enforce_stationarity=False,
    enforce_invertibility=False
)
```

### 3. Model Fitting

```python
# Fit model with optimization settings
results = model.fit(
    disp=False,           # Suppress optimization output
    maxiter=100           # Maximum iterations for convergence
)
```

### 4. Model Persistence

```python
# Save trained model
with open('sarimax_model_weekly.pkl', 'wb') as model_file:
    pickle.dump(results, model_file)

# Save model summary
with open('sarimax_model_summary.txt', 'w') as f:
    f.write(str(results.summary()))
```

## Model Performance

### Fit Statistics

| Metric         | Value    |
| -------------- | -------- |
| Log Likelihood | -806.493 |
| AIC            | 1630.985 |
| BIC            | 1652.311 |
| HQIC           | 1639.529 |

### Coefficient Analysis

#### ARIMA Components

| Component       | Coefficient | Std Error | P-value | Status                |
| --------------- | ----------- | --------- | ------- | --------------------- |
| AR(1)           | -0.6359     | 0.209     | 0.002   | ✅ Significant        |
| MA(1)           | 0.9083      | 0.120     | 0.000   | ✅ Highly Significant |
| Seasonal AR(24) | -0.2301     | 2.201     | 0.917   | ❌ Not Significant    |
| Seasonal MA(24) | 0.2416      | 2.163     | 0.911   | ❌ Not Significant    |

## Model Diagnostics

### ✅ Strengths

1. **No Autocorrelation**: Ljung-Box test (Q=0.38, p=0.54) shows no residual autocorrelation
2. **Significant Temporal Patterns**: AR(1) and MA(1) components effectively capture time dependencies
3. **Strong External Factors**: Weekend and holiday effects are highly significant
4. **Business Logic**: Model captures intuitive ridership patterns

### ⚠️ Areas of Concern

1. **Non-Normal Residuals**: Jarque-Bera test indicates extreme kurtosis (15.18)
2. **Heteroskedasticity**: Non-constant variance detected (H-statistic=0.17, p<0.01)
3. **Ineffective Seasonality**: 24-hour seasonal components are not significant
4. **Numerical Instability**: Covariance matrix condition number is very high (5.89e+26)

## Dependencies

### Required Libraries

```python
# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Statistical modeling
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Utilities
import itertools
```

### Installation

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels
```

## Training Execution

### Command Line

```bash
cd model/models
python train_sarimax.py
```

### Expected Outputs

1. **Model File**: `sarimax_model_weekly.pkl` - Trained model for inference
2. **Summary Report**: `sarimax_model_summary.txt` - Detailed statistical summary
3. **Visualization**: `sarimax_model_weekly_plots.png` - Actual vs predicted plots
4. **Console Output**: Training progress and evaluation metrics

### Sample Console Output

```
Starting SARIMAX model training with cleaned_data.csv...
Loading data from model/data/cleaned_data.csv...
Data loaded successfully. Shape: (133, 8)
Date range: 2025-01-01 08:00:00 to 2025-01-08 09:00:00
Training data: 106 samples
Testing data: 27 samples
Final exogenous variables: ['day_of_week', 'is_weekend', 'is_holiday', 'month']
Initializing SARIMAX model with weekly seasonality...
Fitting model...
Model fitting complete!

Model Evaluation Metrics:
MAE: 245.67
RMSE: 412.34
R² Score: 0.3421

Model training complete and saved as 'sarimax_model_weekly.pkl'
```

## Usage for Prediction

### Loading Trained Model

```python
import pickle
with open('sarimax_model_weekly.pkl', 'rb') as f:
    trained_model = pickle.load(f)

# Make predictions
forecast = trained_model.get_forecast(
    steps=24,  # Predict next 24 hours
    exog=future_exog_data
)
predictions = forecast.predicted_mean
```

## Recommendations for Improvement

### Immediate Actions

1. **Outlier Investigation**: Analyze extreme values causing high kurtosis
2. **Variance Stabilization**: Apply log transformation or Box-Cox transformation
3. **Model Simplification**: Remove non-significant seasonal components

### Advanced Enhancements

1. **GARCH Models**: Address heteroskedasticity with volatility modeling
2. **Alternative Seasonality**: Try different seasonal periods (weekly: S=168)
3. **Additional Features**: Include weather, events, or route-specific variables
4. **Ensemble Methods**: Combine SARIMAX with other forecasting models

## File Structure

```
model/
├── models/
│   ├── train_sarimax.py           # Main training script
│   └── sarimax_model_weekly.pkl   # Saved trained model
├── data/
│   └── cleaned_data.csv          # Preprocessed training data
├── inference/
│   └── predict_sarimax.py        # Prediction script
└── utils/
    └── feature_engineer.py       # Feature engineering utilities
```

## Conclusion

The SARIMAX model successfully captures key ridership patterns, particularly weekend and holiday effects. While the model shows moderate predictive performance, addressing the residual diagnostics issues (non-normality and heteroskedasticity) would significantly improve its reliability for production forecasting applications.

The model serves as a solid baseline for time series forecasting in the ridership prediction system and provides interpretable insights into factors affecting public transport usage patterns.
