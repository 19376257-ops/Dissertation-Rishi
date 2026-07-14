import numpy as np
import pandas as pd

def compute_isi(df: pd.DataFrame, channel: int = None) -> dict:
    """
    Compute inter-spike intervals (ISI) for neural activity data.

    1) Init result dictionary
    2) Filter data by channel if specified
    3) Sort data by time
    4) Calculate time differences between consecutive spikes
    5) Handle datetime or numeric time formats appropriately
    6) Remove NaN values (first entry has no previous spike)
    7) Calculate basic statistics (mean, standard deviation)
    8) Return ISI values and statistics

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter channel: int

                Returns:
                    :return result: dict
    """
    result = {}

    if channel is not None:
        channel_data = df[df['channel'] == channel].copy()
        result['channel'] = channel
    else:
        channel_data = df.copy()

    channel_data = channel_data.sort_values('Time')

    if pd.api.types.is_datetime64_any_dtype(channel_data['Time']):
        isi = channel_data['Time'].diff().dt.total_seconds()
    else:
        isi = channel_data['Time'].diff()

    isi = isi.dropna().values

    # Calc stats
    result['isi'] = isi
    result['mean_isi'] = np.mean(isi) if len(isi) > 0 else np.nan
    result['std_isi'] = np.std(isi) if len(isi) > 0 else np.nan

    return result
