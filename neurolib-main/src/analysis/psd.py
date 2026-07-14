import numpy as np
import pandas as pd
from scipy.signal import welch

def compute_power_spectral_density(df: pd.DataFrame, channel: int, fs: float, nperseg: int = None,
    noverlap: int = None) -> dict:
    """
    Estimate the power spectral density (PSD) of amplitude time-series for a given channel.

    1) Filter data to include only the specified channel
    2) Check if there are amplitude values for the channel
    3) Extract amplitude values
    4) Set default values for nperseg if not provided
    5) Set default values for noverlap if not provided
    6) Compute power spectral density using Welch's method
    7) Return frequencies and corresponding PSD values

                Parameters:
                    :parameter df: pandas.DataFrame
                    :parameter channel: int
                    :parameter fs: float
                    :parameter nperseg: int, optional
                    :parameter noverlap: int, optional

                Returns:
                    :return result: dict {'freqs': np.ndarray, 'psd': np.ndarray}
    """
    data = df[df['channel'] == channel]
    if data.empty:
        return {'freqs': np.array([]), 'psd': np.array([])}

    amps = data['Amplitude'].values

    if nperseg is None:
        nperseg = 64
    if noverlap is None:
        noverlap = None

    # PSD
    freqs, psd_vals = welch(
        amps,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap
    )

    return {'freqs': freqs, 'psd': psd_vals}
