import numpy as np


def add_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['lag_1_week'] = df.groupby(['origin', 'destination'])['ridership'].shift(24 * 7)
    return df