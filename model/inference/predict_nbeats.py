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

        # Feature engineering
        self._prepare_encoders()
        self._calculate_baselines()
        self._train_scalers()

    def _prepare_encoders(self):
        """Initialize all required encoders and normalizers"""
        # Station encoding with frequency-based weighting
        origin_counts = self.df['origin'].value_counts(normalize=True)
        dest_counts = self.df['destination'].value_counts(normalize=True)
        combined_weights = (origin_counts + dest_counts).sort_values(ascending=False)
        self.station_encoder = {station: i for i, station in enumerate(combined_weights.index)}

        # Time features analysis
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['day_part'] = self.df['hour'].apply(self._get_day_part)

    def _get_day_part(self, hour: int) -> int:
        """Categorize hours into meaningful periods"""
        if 5 <= hour < 9:
            return 0  # Early morning
        elif 9 <= hour < 12:
            return 1  # Late morning
        elif 12 <= hour < 14:
            return 2  # Noon
        elif 14 <= hour < 17:
            return 3  # Afternoon
        elif 17 <= hour < 20:
            return 4  # Evening
        else:
            return 5  # Night

    def _calculate_baselines(self):
        """Calculate advanced baseline statistics"""
        # Route-time specific averages
        self.df['route'] = self.df['origin'] + "→" + self.df['destination']
        self.route_stats = self.df.groupby(['route', 'day_part'])['ridership'].agg(['mean', 'std', 'count'])

        # Global statistics
        self.global_mean = self.df['ridership'].mean()
        self.global_std = self.df['ridership'].std()

    def _train_scalers(self):
        """Train feature normalizers"""
        self.scaler = MinMaxScaler()
        features = self._create_feature_matrix(self.df.sample(1000))  # Sample for scaling
        self.scaler.fit(features)

    def _create_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Create normalized feature matrix"""
        features = []
        for _, row in df.iterrows():
            features.append([
                np.sin(2 * np.pi * row['hour'] / 24),
                np.cos(2 * np.pi * row['hour'] / 24),
                row['datetime'].weekday(),
                self.station_encoder[row['origin']],
                self.station_encoder[row['destination']],
                row['day_part']
            ])
        return np.array(features)

    def predict_with_confidence(self, dt: datetime, origin: str, destination: str) -> tuple:
        """Make high-accuracy prediction with reliable error estimate"""
        route = f"{origin}→{destination}"
        day_part = self._get_day_part(dt.hour)
        try:

            # Get historical statistics
            if (route, day_part) in self.route_stats.index:
                route_mean = self.route_stats.loc[(route, day_part), 'mean']
                route_std = self.route_stats.loc[(route, day_part), 'std']
                samples = self.route_stats.loc[(route, day_part), 'count']
            else:
                route_mean = self.global_mean
                route_std = self.global_std
                samples = 0

            # Prepare features
            features = np.array([[
                np.sin(2 * np.pi * dt.hour / 24),
                np.cos(2 * np.pi * dt.hour / 24),
                dt.weekday(),
                self.station_encoder[origin],
                self.station_encoder[destination],
                day_part
            ]])

            # Normalize features
            features = self.scaler.transform(features)

            # Get model prediction
            input_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                raw_pred = self.model(input_tensor).item()

            # Blend prediction with historical data
            blend_weight = min(0.8, samples / (samples + 100))  # More weight to model when we have more data
            final_pred = raw_pred * blend_weight + route_mean * (1 - blend_weight)

            # Calculate confidence (lower error% for more samples)
            base_error = 0.2 + (0.8 * np.exp(-samples / 50))  # Ranges from 20% to 100%
            error_pct = min(100, int(100 * base_error * (route_std / route_mean if route_mean > 0 else 1)))

            return int(final_pred), error_pct

        except Exception as e:
            print(f"Prediction warning: {str(e)}")
            route_mean = self.route_stats.loc[(route, day_part), 'mean'] if (route,
                                                                             day_part) in self.route_stats.index else self.global_mean
            return int(route_mean), 50  # Default fallback


if __name__ == "__main__":
    # Initialize predictor
    predictor = NBeatsPredictor(
        model_path="../saved_models/nbeats-best.ckpt",
        training_data_path="../data/cleaned_data.csv"
    )

    # Test cases
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