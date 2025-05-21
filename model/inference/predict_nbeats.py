import torch
import pandas as pd
import numpy as np
from datetime import datetime
from model.models.train_nbeats import NBeats
from sklearn.preprocessing import MinMaxScaler


class NBeatsPredictor:
    def __init__(self, model_path: str, training_data_path: str = "../data/cleaned_data.csv"):
        # Device setup
        self.device = torch.device("cpu")

        # Load model with version mismatch handling
        self.model = NBeats.load_from_checkpoint(model_path, strict=False).to(self.device)
        self.model.eval()

        # Load and preprocess training data
        self.df = pd.read_csv(training_data_path)
        self.df['datetime'] = pd.to_datetime(self.df['date'] + ' ' + self.df['time'])

        # Feature engineering and encoders
        self._prepare_encoders()

        # Calculate baseline statistics for routes
        self._calculate_baselines()

        # Train feature scaler on sample data with consistent feature set
        self._train_scalers()

    def _prepare_encoders(self):
        """Initialize all required encoders and normalizers"""
        # Station encoding with frequency-based weighting
        origin_counts = self.df['origin'].value_counts(normalize=True)
        dest_counts = self.df['destination'].value_counts(normalize=True)
        combined_weights = (origin_counts + dest_counts).sort_values(ascending=False)
        self.station_encoder = {station: i for i, station in enumerate(combined_weights.index)}

        # Extract hour for all rows
        self.df['hour'] = self.df['datetime'].dt.hour

    def _calculate_baselines(self):
        """Calculate advanced baseline statistics"""
        # Route-time specific averages
        self.df['route'] = self.df['origin'] + "→" + self.df['destination']
        self.route_stats = self.df.groupby(['route', self.df['datetime'].dt.weekday])['ridership'].agg(['mean', 'std', 'count'])

        # Global statistics
        self.global_mean = self.df['ridership'].mean()
        self.global_std = self.df['ridership'].std()

    def _create_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Create normalized feature matrix with 5 features to match model input size"""
        features = []
        for _, row in df.iterrows():
            # Ensure stations are known
            if row['origin'] not in self.station_encoder or row['destination'] not in self.station_encoder:
                continue
            features.append([
                np.sin(2 * np.pi * row['hour'] / 24),         # feature 1
                np.cos(2 * np.pi * row['hour'] / 24),         # feature 2
                row['datetime'].weekday(),                      # feature 3
                self.station_encoder[row['origin']],            # feature 4
                self.station_encoder[row['destination']]        # feature 5
            ])
        return np.array(features)

    def _train_scalers(self):
        """Train feature normalizers on a sample of data"""
        self.scaler = MinMaxScaler()
        sample_df = self.df.sample(min(1000, len(self.df)))  # Sample safely
        features = self._create_feature_matrix(sample_df)
        self.scaler.fit(features)

    def predict_with_confidence(self, dt: datetime, origin: str, destination: str) -> tuple:
        """Make high-accuracy prediction with reliable error estimate"""
        route = f"{origin}→{destination}"
        day_of_week = dt.weekday()

        if origin not in self.station_encoder or destination not in self.station_encoder:
            raise ValueError(f"Unknown station(s): {origin} or {destination}")

        try:
            # Get historical stats for this route and day
            if (route, day_of_week) in self.route_stats.index:
                route_mean = self.route_stats.loc[(route, day_of_week), 'mean']
                route_std = self.route_stats.loc[(route, day_of_week), 'std']
                samples = self.route_stats.loc[(route, day_of_week), 'count']
            else:
                route_mean = self.global_mean
                route_std = self.global_std
                samples = 0

            # Prepare features for prediction
            features = np.array([[
                np.sin(2 * np.pi * dt.hour / 24),
                np.cos(2 * np.pi * dt.hour / 24),
                dt.weekday(),
                self.station_encoder[origin],
                self.station_encoder[destination]
            ]])

            # Normalize features
            features = self.scaler.transform(features)

            # Convert to tensor
            input_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

            # Model prediction
            with torch.no_grad():
                raw_pred = self.model(input_tensor).item()

            # Blend model prediction and historical mean, weighted by sample count
            blend_weight = min(0.8, samples / (samples + 100))
            final_pred = raw_pred * blend_weight + route_mean * (1 - blend_weight)

            # Calculate confidence interval error percentage
            base_error = 0.2 + (0.8 * np.exp(-samples / 50))  # From 20% to 100%
            error_pct = min(100, int(100 * base_error * (route_std / route_mean if route_mean > 0 else 1)))

            return int(final_pred), error_pct

        except Exception as e:
            print(f"Prediction warning: {str(e)}")
            fallback_mean = self.route_stats.loc[(route, day_of_week), 'mean'] if (route, day_of_week) in self.route_stats.index else self.global_mean
            return int(fallback_mean), 50  # Safe fallback


if __name__ == "__main__":
    # Initialize predictor
    predictor = NBeatsPredictor(
        model_path="../saved_models/nbeats-best.ckpt",
        training_data_path="../data/cleaned_data.csv"
    )

    # Example test cases
    test_cases = [
        (datetime(2025, 1, 4, 6, 0), "Pulau Sebang (Tampin)", "KL Sentral", 25),
        (datetime(2025, 1, 2, 19, 0), "KL Sentral", "Serdang", 100),
        (datetime(2025, 1, 3, 10, 0), "KL Sentral", "Seremban", 32),
        (datetime(2025, 1, 4, 15, 0), "Batu Caves", "KL Sentral", 100),
        (datetime(2025, 1, 4, 15, 0), "Batu Caves", "Kampung Batu", 28),
    ]

    print("High-Accuracy Prediction Results:")
    print("=" * 60)
    for dt, orig, dest, expected in test_cases:
        pred, error_pct = predictor.predict_with_confidence(dt, orig, dest)
        actual_error = abs(pred - expected) / expected * 100

        print(f"{orig} -- {dest} at {dt.strftime('%a %H:%M')}:")
        print(f"  Predicted: {pred} ±{error_pct}%")
        print(f"  Expected: {expected}")
        print(f"  Actual Error: {actual_error:.1f}%")
        print("-" * 60)
