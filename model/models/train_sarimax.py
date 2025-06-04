import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools

# Load and aggregate data by hour using cleaned_data.csv
def load_and_aggregate_hourly_data():
    data_path = 'model/data/cleaned_data.csv'  # Using cleaned data
    print(f"Loading data from {data_path}...")
    komuter_data = pd.read_csv(data_path)
    komuter_data.columns = komuter_data.columns.str.strip()  # Fix column names

    # Strip whitespace from all string columns
    for col in komuter_data.select_dtypes(include='object').columns:
        komuter_data[col] = komuter_data[col].str.strip()

    # Create datetime column
    komuter_data['datetime'] = pd.to_datetime(komuter_data['date'] + ' ' + komuter_data['time'])
    
    # Aggregate by hour to get total ridership per hour
    print("Aggregating data by hour...")
    hourly_data = komuter_data.groupby('datetime').agg({
        'ridership': 'sum',
        'day_of_week': 'first',  # These should be consistent for same datetime
        'is_weekend': 'first',
        'is_holiday': 'first'
    }).reset_index()
    
    # Add additional time-based features
    hourly_data['hour_of_day'] = hourly_data['datetime'].dt.hour
    hourly_data['month'] = hourly_data['datetime'].dt.month
    hourly_data['day_of_month'] = hourly_data['datetime'].dt.day
    
    print(f"Data loaded successfully. Shape: {hourly_data.shape}")
    print(f"Date range: {hourly_data['datetime'].min()} to {hourly_data['datetime'].max()}")
    
    return hourly_data

# Encode the 'origin' and 'destination' columns (kept for backward compatibility)
def encode_data(data):
    label_encoder = LabelEncoder()
    data['origin_encoded'] = label_encoder.fit_transform(data['origin'])
    data['destination_encoded'] = label_encoder.fit_transform(data['destination'])
    return data

# Train SARIMAX with weekly seasonality and additional features
def train_weekly_seasonal_sarimax(train_data, test_data, exog_vars):
    print("Initializing SARIMAX model with weekly seasonality...")
    print(f"Using exogenous variables: {exog_vars}")
    
    # Weekly seasonality for hourly data: 24 hours * 7 days = 168
    p, d, q = 1, 1, 1
    P, D, Q, S = 1, 0, 1, 24  # Daily seasonality instead of weekly
    
    try:
        # Aggregate data to daily level to reduce memory usage
        train_data = train_data.resample('D', on='datetime').sum().reset_index()
        test_data = test_data.resample('D', on='datetime').sum().reset_index()

        # Update exogenous variables to match daily aggregation
        exog_vars = ['day_of_week', 'is_weekend', 'is_holiday', 'month']
    
        model = SARIMAX(
            train_data['ridership'],
            exog=train_data[exog_vars],
            order=(p, d, q),
            seasonal_order=(P, D, Q, S),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        print("Fitting model...")
        results = model.fit(disp=False, maxiter=100)
        print("Model fitting complete!")
        
        # Debugging: Check the type of the results object
        print(f"Type of results object: {type(results)}")

        # Ensure the results object is correctly typed
        from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
        if not isinstance(results, SARIMAXResultsWrapper):
            raise TypeError("The results object is not a valid SARIMAXResultsWrapper instance. Check the model.fit() method.")

        # Proceed with forecast and summary operations
        forecast = results.get_forecast(
            steps=len(test_data),
            exog=test_data[exog_vars]
        )
        forecasted_values = forecast.predicted_mean
        
        # Align indices of forecasted values and actual values
        forecasted_values = forecasted_values.reset_index(drop=True)
        actual_values = test_data['ridership'].reset_index(drop=True)

        # Ensure sizes match
        min_length = min(len(forecasted_values), len(actual_values))
        forecasted_values = forecasted_values.iloc[:min_length]
        actual_values = actual_values.iloc[:min_length]

        # Calculate evaluation metrics
        mae = mean_absolute_error(actual_values, forecasted_values)
        rmse = np.sqrt(mean_squared_error(actual_values, forecasted_values))
        r2 = r2_score(actual_values, forecasted_values)
        
        print("\nModel Evaluation Metrics:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Create visualization plots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Actual vs Predicted
        plt.subplot(2, 1, 1)
        plt.plot(test_data['datetime'].iloc[:min(500, len(test_data))], 
                test_data['ridership'].iloc[:min(500, len(test_data))], 
                label='Actual', color='blue', alpha=0.7)
        plt.plot(test_data['datetime'].iloc[:min(500, len(test_data))], 
                forecasted_values.iloc[:min(500, len(forecasted_values))], 
                label='Predicted', color='red', alpha=0.7)
        plt.title('SARIMAX Model (Weekly Seasonality): Actual vs Predicted Ridership (First 500 points)')
        plt.xlabel('Date')
        plt.ylabel('Ridership')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Plot 2: Residuals
        plt.subplot(2, 1, 2)
        residuals = test_data['ridership'] - forecasted_values
        plt.scatter(forecasted_values, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('sarimax_model_weekly_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save the trained model
        print("Saving model...")
        with open('sarimax_model_weekly.pkl', 'wb') as model_file:
            pickle.dump(results, model_file)

        # Also save model summary
        with open('sarimax_model_summary.txt', 'w') as f:
            f.write(str(results.summary()))
        
        return results, forecasted_values
        
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None

# Main function to run the training and save the model
if __name__ == "__main__":
    print("Starting SARIMAX model training with cleaned_data.csv...")
    
    # Step 1: Load and aggregate the data by hour
    hourly_data = load_and_aggregate_hourly_data()
    
    if hourly_data is None or len(hourly_data) == 0:
        print("Error: No data loaded!")
        exit(1)
    
    # Step 2: Sort by datetime to ensure proper time series order
    hourly_data = hourly_data.sort_values('datetime').reset_index(drop=True)
    
    # Step 3: Split the data into training and testing sets
    train_size = int(len(hourly_data) * 0.8)
    train_data, test_data = hourly_data[:train_size], hourly_data[train_size:]
    
    print(f"Training data: {len(train_data)} samples")
    print(f"Testing data: {len(test_data)} samples")
    
    # Step 4: Define exogenous variables (features to help with prediction)
    exog_vars = ['day_of_week', 'hour_of_day', 'is_weekend', 'is_holiday', 'month']
    
    # Check if all exogenous variables exist in the data
    missing_vars = [var for var in exog_vars if var not in train_data.columns]
    if missing_vars:
        print(f"Warning: Missing variables {missing_vars}, removing from exog_vars")
        exog_vars = [var for var in exog_vars if var in train_data.columns]
    
    print(f"Final exogenous variables: {exog_vars}")
    
    # Step 5: Train the SARIMAX model with weekly seasonality
    results, forecasted_values = train_weekly_seasonal_sarimax(train_data, test_data, exog_vars)
    
    if results is not None:
        print("Model training complete and saved as 'sarimax_model_weekly.pkl'")
        print("Model summary saved as 'sarimax_model_summary.txt'")
        print("Visualization saved as 'sarimax_model_weekly_plots.png'")
    else:
        print("Model training failed!")
