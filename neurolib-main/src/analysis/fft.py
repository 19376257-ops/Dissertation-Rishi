import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

def compute_fft(df: pd.DataFrame, channel: int, resample_interval: int, freq_cap: float) -> dict:
    """
    Compute Fast Fourier Transform (FFT) of neural signal to analyse frequency components.

    1) Select channel data and extract time series
    2) Resample data to uniform time grid
    3) Remove DC component (mean) from signal
    4) Compute FFT of the signal
    5) Apply frequency cap to limit the range
    6) Return frequencies and their corresponding amplitudes

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter channel: int
                    :parameter resample_interval: int
                    :parameter freq_cap: float

                Returns:
                    :return result: dict
    """
    # 1) Select channel data
    ts = df[df['channel']==channel][['Time','Amplitude']].set_index('Time')

    # Check if there's any data for this channel
    if ts.empty:
        return {'freqs': np.array([]), 'amplitudes': np.array([])}

    # 2) Resample to uniform grid
    sig = ts.resample(f"{resample_interval}ms").mean().interpolate()

    # Error hndling
    if sig.empty or len(sig) < 2:
        return {'freqs': np.array([]), 'amplitudes': np.array([])}

    # 3) Remove DC
    signal = sig['Amplitude'] - sig['Amplitude'].mean()
    N = len(signal)
    T = resample_interval / 1000.0

    # 4)Compute FFT
    yf = fft(signal)
    xf = fftfreq(N, T)[:N//2]

    # 5) Cap frequency
    mask = xf <= freq_cap
    return {'freqs': xf[mask], 'amplitudes': 2.0/N * np.abs(yf[:N//2][mask])}
