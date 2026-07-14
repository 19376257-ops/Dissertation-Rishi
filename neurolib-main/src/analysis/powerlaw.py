import pandas as pd
import powerlaw

def compute_powerlaw(df: pd.DataFrame, channel: int = None) -> dict:
    """
    Compute power-law fit for inter-spike intervals to analyse neural activity patterns.

    1) Initialise result dictionary
    2) Filter data by channel if specified
    3) Sort data by time
    4) Calculate inter-spike intervals (ISI)
    5) Handle datetime or numeric time formats appropriately
    6) Remove NaN values (first entry has no previous spike)
    7) Check if there's enough data for fitting (at least 10 points)
    8) Fit power-law distribution to the ISI data
    9) Extract fit parameters (alpha, xmin, D)
    10) Handle exceptions and insufficient data cases
    11) Return fit results or error information

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

    if len(isi) > 10:  # Need a reasonable amount of data for fitting
        try:
            fit = powerlaw.Fit(isi)

            result['fit'] = fit
            result['alpha'] = fit.alpha
            result['xmin'] = fit.xmin
            result['D'] = fit.D

        except Exception as e:
            result['error'] = str(e)
    else:
        result['error'] = f"Not enough data points ({len(isi)}) for power-law fitting"

    return result
