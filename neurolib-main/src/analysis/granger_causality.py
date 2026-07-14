import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

def compute_granger_causality(df: pd.DataFrame, pre_channel: int, post_channel: int, resample_interval: str = '1s', max_lag: int = 5, sample_mode: bool = False) -> dict:
    """
    Compute Granger causality test to determine if one neura channel predicts another.

    1) Create a copy of the input data and ensure Time is in datetime format
    2) Apply sample mode if enabled (limit to 200 data points)
    3) Set Time as index for time series analysis
    4) Filter data to include only pre and post channels
    5) Resample data to regular intervals
    6) Extract time series for each channel
    7) Validate that both channels have data
    8) Combine both time series into a single DataFrame
    9) Remove any rows with NaN values
    10) Check if there's enough data for the test
    11) Perform the Granger causality test
    12) Extract p-values and identify significant lags
    13) Return test results or error information

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter pre_channel: int
                    :parameter post_channel: int
                    :parameter resample_interval: str
                    :parameter max_lag: int
                    :parameter sample_mode: bool

                Returns:
                    :return result: dict
    """

    data = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(data['Time']):
        data['Time'] = pd.to_datetime(data['Time'])

    if sample_mode and len(data) > 200:
        data = data.sample(n=200, random_state=42)

    # Time as index
    data.set_index('Time', inplace=True)

    channel_data = data[data['channel'].isin([pre_channel, post_channel])]

    resample_interval = resample_interval.replace('S', 's')
    resampled_data = channel_data.groupby('channel').resample(resample_interval).size().unstack(level=0).fillna(0)

    # 1) Extract time series for ech channel
    if pre_channel in resampled_data.columns and post_channel in resampled_data.columns:
        pre_series = resampled_data[pre_channel]
        post_series = resampled_data[post_channel]
    else:
        missing_channels = []
        if pre_channel not in resampled_data.columns:
            missing_channels.append(f"pre_channel {pre_channel}")
        if post_channel not in resampled_data.columns:
            missing_channels.append(f"post_channel {post_channel}")

        return {
            'pre_channel': pre_channel,
            'post_channel': post_channel,
            'error': f"Missing data for {', '.join(missing_channels)}",
            'p_values': {},
            'significant_lags': [],
            'resampled_data': resampled_data
        }

    # 2) COmbine both time series into a single df
    granger_data = pd.concat([pre_series, post_series], axis=1)
    granger_data.columns = [f'channel_{pre_channel}', f'channel_{post_channel}']

    granger_data = granger_data.dropna()

    # 3) Check if we have enough data for the test
    if len(granger_data) <= max_lag:
        return {
            'pre_channel': pre_channel,
            'post_channel': post_channel,
            'error': f"Not enough data points ({len(granger_data)}) for Granger test with max_lag={max_lag}",
            'p_values': {},
            'significant_lags': [],
            'resampled_data': granger_data
        }

    # 4) Perform the Granger causality test
    try:
        # Remove verbose parameter
        granger_results = grangercausalitytests(granger_data, max_lag)

        p_values = {lag: granger_results[lag][0]['ssr_chi2test'][1] for lag in range(1, max_lag+1)}
        significant_lags = [lag for lag, p in p_values.items() if p < 0.05]

        return {
            'pre_channel': pre_channel,
            'post_channel': post_channel,
            'p_values': p_values,
            'significant_lags': significant_lags,
            'resampled_data': granger_data
        }
    except Exception as e:
        return {
            'pre_channel': pre_channel,
            'post_channel': post_channel,
            'error': str(e),
            'p_values': {},
            'significant_lags': [],
            'resampled_data': granger_data
        }
