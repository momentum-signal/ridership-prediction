# SARIMAX Model Performance Assessment

## Model Specification

- **Model**: SARIMAX(1, 1, 1)x(1, 0, 1, 24)
- **Dependent Variable**: Ridership
- **Sample Size**: 106 observations
- **Seasonal Period**: 24 (indicating hourly data with daily seasonality)

## Model Fit Statistics

| Metric         | Value    |
| -------------- | -------- |
| Log Likelihood | -806.493 |
| AIC            | 1630.985 |
| BIC            | 1652.311 |
| HQIC           | 1639.529 |

## Coefficient Analysis

### Exogenous Variables

| Variable    | Coefficient | Std Error | P-value | Significance           | Interpretation                                     |
| ----------- | ----------- | --------- | ------- | ---------------------- | -------------------------------------------------- |
| day_of_week | 8.0028      | 24.497    | 0.744   | Not significant        | Day of week has no significant impact on ridership |
| is_weekend  | -541.4570   | 131.750   | 0.000   | Highly significant     | Weekend reduces ridership by ~541 units            |
| is_holiday  | 388.7052    | 79.355    | 0.000   | Highly significant     | Holidays increase ridership by ~389 units          |
| month       | 248.9845    | 144.068   | 0.084   | Marginally significant | Seasonal monthly effect (p < 0.1)                  |

### ARIMA Components

| Component       | Coefficient | Std Error | P-value | Significance       |
| --------------- | ----------- | --------- | ------- | ------------------ |
| AR(1)           | -0.6359     | 0.209     | 0.002   | Significant        |
| MA(1)           | 0.9083      | 0.120     | 0.000   | Highly significant |
| Seasonal AR(24) | -0.2301     | 2.201     | 0.917   | Not significant    |
| Seasonal MA(24) | 0.2416      | 2.163     | 0.911   | Not significant    |

## Diagnostic Tests

### Residual Analysis

- **Ljung-Box Test (Q)**: 0.38 (p = 0.54)
  - **Result**: No significant autocorrelation in residuals ✓
  - **Interpretation**: Model adequately captures temporal dependencies

### Normality Test

- **Jarque-Bera Test**: 488.06 (p = 0.00)
  - **Result**: Residuals are NOT normally distributed ✗
  - **Skewness**: -0.06 (nearly symmetric)
  - **Kurtosis**: 15.18 (extremely heavy-tailed)

### Heteroskedasticity Test

- **H-statistic**: 0.17 (p = 0.00)
  - **Result**: Significant heteroskedasticity present ✗
  - **Interpretation**: Variance is not constant over time

## Key Findings

### Strengths

1. **Temporal Structure**: Model successfully captures autocorrelation patterns (no residual autocorrelation)
2. **Weekend Effect**: Strong negative weekend impact (-541 units, highly significant)
3. **Holiday Effect**: Significant positive holiday impact (+389 units)
4. **ARIMA Components**: AR(1) and MA(1) terms are statistically significant

### Weaknesses

1. **Non-normal Residuals**: Extremely high kurtosis (15.18) indicates outliers or model misspecification
2. **Heteroskedasticity**: Non-constant variance suggests volatility clustering
3. **Seasonal Components**: Seasonal AR and MA terms are not significant
4. **Model Stability**: Covariance matrix issues (condition number 5.89e+26)

## Model Performance Assessment

### Overall Rating: ⚠️ **Moderate with Concerns**

**Positive Aspects:**

- Captures key business patterns (weekend/holiday effects)
- No residual autocorrelation
- Reasonable fit for prediction purposes

**Areas of Concern:**

- Severe non-normality and heavy tails in residuals
- Heteroskedasticity indicates model may not capture all variance patterns
- Numerical instability in covariance matrix
- Ineffective seasonal modeling

## Recommendations

### Immediate Actions

1. **Outlier Detection**: Investigate extreme values causing high kurtosis
2. **Variance Modeling**: Consider GARCH-type models for heteroskedasticity
3. **Transformation**: Apply log or Box-Cox transformation to stabilize variance

### Model Improvements

1. **Alternative Specifications**:
   - Try different seasonal periods or simpler ARIMA structure
   - Consider removing non-significant seasonal components
2. **Robust Estimation**: Use robust standard errors to handle non-normality
3. **Additional Variables**: Include more exogenous variables to explain variance

### Validation Steps

1. **Out-of-sample Testing**: Evaluate forecast accuracy on holdout data
2. **Cross-validation**: Assess model stability across different time periods
3. **Residual Analysis**: Perform detailed residual diagnostics

## Conclusion

While the SARIMAX model captures important ridership patterns (weekend/holiday effects), significant diagnostic issues limit its reliability. The model is suitable for initial insights but requires refinement before deployment for critical forecasting applications. Priority should be given to addressing the non-normality and heteroskedasticity issues to improve model robustness.
