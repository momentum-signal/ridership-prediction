#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
import numpy as np
from lightning.pytorch import Trainer
#from pytorch_lightning import Trainer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE
from pytorch_lightning.callbacks import EarlyStopping


df = pd.read_csv("/Users/nik/github/ridership-prediction/model/data/cleaned_data.csv")


df.head()


df.columns


df.dtypes


# pre-process by combine datetime, change datetime64 dtypes 
df['date'] = df['date'].astype(str)
df['time'] = df['time'].astype(str)
df['timestamp'] = df['date']+' '+df['time']
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M')


df['route'] = df['origin']+'_to_'+df['destination']


# sort by route, then time
df = df.sort_values(by=['route', 'timestamp']).reset_index(drop=True)


# create new time_idx, the time series
df['time_idx'] = df.groupby('route').cumcount()
#covariate for dayofweek
df['dayofweek_sin'] = np.sin(2 * np.pi * (df['day_of_week'] - 1) / 7)
df['dayofweek_cos'] = np.cos(2 * np.pi * (df['day_of_week'] - 1) / 7)
#covariate holiday and weekend is hot code, taken care of
print(df.head(20))


group_counts = df.groupby("route").size()
valid_groups = group_counts[group_counts > 40].index
df = df[df["route"].isin(valid_groups)]


max_encoder_length = 30  # lookback
max_prediction_length = 7  # forecast horizon

training = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="ridership",
    group_ids=["route"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["route"],
    time_varying_known_reals=["time_idx", "dayofweek_sin", "dayofweek_cos", "is_weekend", "is_holiday"],
    time_varying_unknown_reals=["ridership"],
    target_normalizer=NaNLabelEncoder(),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)


#print total series loaded
total_series = df["route"].nunique()
print(f"Total series loaded: {total_series}")


train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=1)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.02,
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    loss=SMAPE(),
    log_interval=10,
)

trainer = Trainer(max_epochs=30, gradient_clip_val=0.1)
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
)
raw_predictions, x = tft.predict(train_dataloader, mode="raw", return_x=True)







