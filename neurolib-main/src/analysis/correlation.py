import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags


# Cross-correlation
def compute_cross_correlation(df: pd.DataFrame, channel1: int, channel2: int, bin_size_ms: int = 10, sample_mode: bool = False) -> dict:
    """
    Compute cross-correlation between two neural channels to analyse their temporal relationship.

    1) Create a copy of the input data
    2) Convert Time column to datetime format
    3) Apply sample mode if enabled (limit to 1000 data points)
    4) Extract data for both channels
    5) Validate that both channels have data
    6) Determine time range and create bins
    7) Define function to bin spike counts for each channel
    8) Get binned spike counts for both channels
    9) Validate that binned data is not empty
    10) Compute cross-correlation between the two series
    11) Calculate time lags for correlation values
    12) Return correlation results with lag times

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter channel1: int
                    :parameter channel2: int
                    :parameter bin_size_ms: int
                    :parameter sample_mode: bool

                Returns:
                    :return result: dict
    """
    data = df.copy()
    data['Time'] = pd.to_datetime(data['Time'])

    # Sample mode: limit the amount of data processed
    if sample_mode and len(data) > 1000:
        # Take a random sample of 1000 data points
        data = data.sample(n=1000, random_state=42)

    # Check if there's data for both channels
    ch1_data = data[data['channel'] == channel1]
    ch2_data = data[data['channel'] == channel2]

    if ch1_data.empty or ch2_data.empty:
        return {
            'error': f"No data for channel(s): {channel1 if ch1_data.empty else ''} {channel2 if ch2_data.empty else ''}",
            'data': {'channel1': channel1, 'channel2': channel2}
        }

    start_time = data['Time'].min()
    end_time = data['Time'].max()
    duration_ms = (end_time - start_time).total_seconds() * 1000
    num_bins = int(np.ceil(duration_ms / bin_size_ms))

    bin_edges = pd.date_range(start=start_time, end=end_time, periods=num_bins + 1)

    def get_binned_counts(channel):
        ch_data = data[data['channel'] == channel]
        return pd.cut(ch_data['Time'], bins=bin_edges).value_counts().sort_index().values

    series1 = get_binned_counts(channel1)
    series2 = get_binned_counts(channel2)

    # Check if binned data is empty
    if len(series1) == 0 or len(series2) == 0:
        return {
            'error': f"Empty binned data for channel(s): {channel1 if len(series1) == 0 else ''} {channel2 if len(series2) == 0 else ''}",
            'data': {'channel1': channel1, 'channel2': channel2}
        }

    corr = correlate(series1 - np.mean(series1), series2 - np.mean(series2), mode='full')
    lags = correlation_lags(len(series1), len(series2), mode='full') * bin_size_ms / 1000

    return {
        'lags': lags,
        'correlation': corr,
        'channel1': channel1,
        'channel2': channel2
    }


# Autocorrelation
def compute_auto_correlation(df: pd.DataFrame, channel: int, bin_size_ms: int = 1, sample_mode: bool = False) -> dict:
    """
    Compute auto-correlation for a single neural channel to analyse its temporal patterns.

    1) Create a copy of the input data
    2) Convert Time column to datetime format
    3) Apply sample mode if enabled (limit to 1000 data points)
    4) Extract data for the specified channel
    5) Validate that the channel has data
    6) Determine time range and create bins
    7) Bin spike counts for the channel
    8) Validate that binned data is not empty
    9) Compute auto-correlation of the series with itself
    10) Calculate time lags for correlation values
    11) Return correlation results with lag times

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter channel: int
                    :parameter bin_size_ms: int
                    :parameter sample_mode: bool

                Returns:
                    :return result: dict
    """
    data = df.copy()
    data['Time'] = pd.to_datetime(data['Time'])

    # Sample mode: limit the amount of data processed
    if sample_mode and len(data) > 1000:
        # Take a random sample of 1000 data points
        data = data.sample(n=1000, random_state=42)

    # Check if there's data for the channel
    ch_data = data[data['channel'] == channel]
    if ch_data.empty:
        return {
            'error': f"No data for channel {channel}",
            'data': {'channel': channel}
        }

    start_time = data['Time'].min()
    end_time = data['Time'].max()
    duration_ms = (end_time - start_time).total_seconds() * 1000
    num_bins = int(np.ceil(duration_ms / bin_size_ms))

    bin_edges = pd.date_range(start=start_time, end=end_time, periods=num_bins + 1)
    binned = pd.cut(ch_data['Time'], bins=bin_edges).value_counts().sort_index().values

    # Check if binned data is empty
    if len(binned) == 0:
        return {
            'error': f"Empty binned data for channel {channel}",
            'data': {'channel': channel}
        }

    auto_corr = correlate(binned - np.mean(binned), binned - np.mean(binned), mode='full')
    lags = correlation_lags(len(binned), len(binned), mode='full') * bin_size_ms / 1000  # Convert to seconds

    return {
        'lags': lags,
        'correlation': auto_corr,
        'channel': channel
    }
