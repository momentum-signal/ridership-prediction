import pickle
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from decimal import Decimal

# Load the SARIMAX model from the saved file
def load_model():
    with open('sarimax_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Load the original dataset and preprocess it
def load_data():
    data_path = 'model/data/komuter_2025.csv'  # Path to your data file
    komuter_data = pd.read_csv(data_path)
    
    # Strip whitespace from column names
    komuter_data.columns = komuter_data.columns.str.strip()
    print("Columns in CSV:", komuter_data.columns.tolist())

    # Encode 'origin' and 'destination' columns
    label_encoder = LabelEncoder()
    komuter_data['origin_encoded'] = label_encoder.fit_transform(komuter_data['origin'])
    komuter_data['destination_encoded'] = label_encoder.fit_transform(komuter_data['destination'])
    
    return komuter_data

# Function to get user input and make prediction
def get_user_input():
    # Load the original data to get real origin and destination values
    komuter_data = load_data()

    # Display list of available stations (origin and destination are the same stations in the dataset)
    print("\nAvailable stations:")
    stations = komuter_data['origin'].unique()  # List of unique stations from the dataset
    for idx, station in enumerate(stations, 1):
        print(f"{idx}. {station}")

    # Get user input for origin and destination (choose from available stations)
    origin_choice = int(input("Choose origin station (Enter number): "))
    origin = stations[origin_choice - 1]

    destination_choice = int(input("Choose destination station (Enter number): "))
    destination = stations[destination_choice - 1]

    # Display selected route
    print(f"\nYou selected the route: {origin} to {destination}")

    # Get the actual date and time
    date = input("\nEnter date (YYYY-MM-DD): ")
    time = input("Enter time (HH:MM): ")

    # Convert date and time to datetime object
    try:
        datetime_obj = datetime.strptime(date + " " + time, "%Y-%m-%d %H:%M")
    except ValueError:
        print("Invalid date or time format. Please enter the correct format.")
        return None

    # Encode the origin and destination based on the dataset (encoding the actual station names)
    origin_encoded = komuter_data[komuter_data['origin'] == origin]['origin_encoded'].values[0]
    destination_encoded = komuter_data[komuter_data['destination'] == destination]['destination_encoded'].values[0]

    # Extract day of week and hour of day
    day_of_week = datetime_obj.weekday()  # 0 = Monday, 6 = Sunday
    hour_of_day = datetime_obj.hour

    # Return the input data as a DataFrame
    return pd.DataFrame({
        'origin_encoded': [origin_encoded],
        'destination_encoded': [destination_encoded],
        'day_of_week': [day_of_week],
        'hour_of_day': [hour_of_day]
    })

# Make predictions based on user input
def make_prediction():
    model = load_model()  # Load the trained SARIMAX model
    new_data = get_user_input()  # Get user input as a DataFrame
    
    if new_data is None:
        return

    # Forecast using the loaded model
    forecast = model.get_forecast(steps=1, exog=new_data[['origin_encoded', 'destination_encoded']])
    predicted_value = int(forecast.predicted_mean.iloc[0])  # Get the first (and only) prediction

    # Based on the predicted ridership, give a suggestion
    # Define the threshold for a "good time to travel"
    threshold = 10  # Adjusted threshold for better fit with the ridership scale

    if predicted_value < threshold:
        print(f"Predicted Ridership: {predicted_value}. It seems like a good time to travel.")
    else:
        print(f"Predicted Ridership: {predicted_value}. It might not be a good time to travel. Consider alternate options.")


# Main function to run the prediction
if __name__ == "__main__":
    print("\nWelcome to the Ridership Prediction System!")
    make_prediction()
