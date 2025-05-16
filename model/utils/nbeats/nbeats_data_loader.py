import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple  # Added missing import


class RidershipDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


def prepare_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader, MinMaxScaler]:
    """Prepare data for training and validation

    Args:
        df: Input DataFrame with features and target
        test_size: Proportion of data to use for validation

    Returns:
        Tuple of (train_loader, val_loader, scaler)
    """
    # Ensure no NaN values
    df = df.dropna()

    # Separate features and target
    feature_cols = ['hour_sin', 'hour_cos', 'lag_1_week', 'day_of_week', 'is_weekend']
    features = df[feature_cols].values
    target = df['ridership'].values.reshape(-1, 1)

    # Normalize features
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # Split data (using lowercase variable names)
    x_train, x_val, y_train, y_val = train_test_split(
        features, target, test_size=test_size, random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = RidershipDataset(x_train, y_train)
    val_dataset = RidershipDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, scaler