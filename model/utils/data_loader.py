import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    return df.sort_values(['origin', 'destination', 'datetime'])
