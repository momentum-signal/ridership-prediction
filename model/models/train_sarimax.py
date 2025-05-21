import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# Load the original dataset and preprocess it
def load_and_aggregate_data():
    # Load the dataset
    data_path = 'model/data/komuter_2025.csv'  # Path to your data file
    komuter_data = pd.read_csv(data_path)
    # Strip whitespace from column names
    komuter_data.columns = komuter_data.columns.str.strip()
    print("Columns in CSV:", komuter_data.columns.tolist())

    # Check for required columns
    if 'date' in komuter_data.columns and 'time' in komuter_data.columns:
        # Strip whitespace from values in 'date' and 'time'
        komuter_data['date'] = komuter_data['date'].astype(str).str.strip()
        komuter_data['time'] = komuter_data['time'].astype(str).str.strip()
        komuter_data['datetime'] = pd.to_datetime(
            komuter_data['date'] + ' ' + komuter_data['time'], format='%Y-%m-%d %H:%M'
        )
    elif 'datetime' in komuter_data.columns:
        komuter_data['datetime'] = pd.to_datetime(komuter_data['datetime'])
    else:
        raise ValueError("CSV must contain either 'date' and 'time' columns or a 'datetime' column. Found columns: " + str(komuter_data.columns.tolist()))

    # Aggregate by date, origin, and destination, summing the ridership for each
    aggregated_data = komuter_data.groupby(['datetime', 'origin', 'destination']).agg({'ridership': 'sum'}).reset_index()

    # Add features like 'day_of_week' and 'hour_of_day'
    aggregated_data['day_of_week'] = aggregated_data['datetime'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    aggregated_data['hour_of_day'] = aggregated_data['datetime'].dt.hour

    return aggregated_data

# Encode the 'origin' and 'destination' columns
def encode_data(data):
    label_encoder = LabelEncoder()
    data['origin_encoded'] = label_encoder.fit_transform(data['origin'])
    data['destination_encoded'] = label_encoder.fit_transform(data['destination'])
    return data

# Train the SARIMAX model
def train_sarimax_model(train_data, test_data):
    # Define and fit the SARIMAX model
    sarimax_model = SARIMAX(train_data['ridership'],
                            exog=train_data[['origin_encoded', 'destination_encoded']],
                            order=(1, 0, 0),  # ARIMA(p=1, d=0, q=0)
                            seasonal_order=(1, 0, 0, 7),  # SARIMA(P=1, D=0, Q=0, S=7 for weekly seasonality
                            enforce_stationarity=False,
                            enforce_invertibility=False)

    sarimax_results = sarimax_model.fit(disp=False)

    # Forecast the test data
    forecast = sarimax_results.get_forecast(steps=len(test_data), exog=test_data[['origin_encoded', 'destination_encoded']])
    forecasted_values = forecast.predicted_mean

    # Calculate MAE for SARIMAX model
    mae = mean_absolute_error(test_data['ridership'], forecasted_values)
    print(f"Final MAE with SARIMAX model: {mae}")

    # Save the trained model
    with open('sarimax_model.pkl', 'wb') as model_file:
        pickle.dump(sarimax_results, model_file)

    return sarimax_results, forecasted_values

# Main function to run the training and save the model
if __name__ == "__main__":
    # Step 1: Load and aggregate the data
    aggregated_data = load_and_aggregate_data()

    # Step 2: Encode the data
    encoded_data = encode_data(aggregated_data)

    # Step 3: Split the data into training and testing sets
    train_size = int(len(encoded_data) * 0.8)
    train_data, test_data = encoded_data[:train_size], encoded_data[train_size:]

    # Step 4: Train the SARIMAX model
    train_sarimax_model(train_data, test_data)

    print("Model training complete and saved as 'sarimax_model.pkl'")
