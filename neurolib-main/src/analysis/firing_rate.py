import numpy as np

def compute_firing_rates(df: 'pd.DataFrame', threshold: float, bin_size_s: float) -> dict:
    """
    Compute firing rates from neural data by detecting spikes and binning.

    1) Detect spikes using amplitude threshold
    2) Compute inter-spike intervals (ISI)
    3) Bin firing rate per time interval
    4) Calculate time edges for bins
    5) Convert spike times to seconds from start
    6) Count spikes in each bin
    7) Calculate firing rate by dividing counts by bin size
    8) Return spike times, ISIs, firing rates, and bin edges

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter threshold: float
                    :parameter bin_size_s: float

                Returns:
                    :return result: dict
    """

    if df.empty:
        return {
            'spike_times': np.array([]),
            'isi': np.array([]),
            'firing_rate': np.array([]),
            'bin_edges': np.array([])
        }

    # 1) Detect spikes via threshold
    df_spikes = df[df['Amplitude'] > threshold]
    spike_times = df_spikes['Time'].values

    # Error hyandling
    if len(spike_times) < 2:
        return {
            'spike_times': spike_times,
            'isi': np.array([]),
            'firing_rate': np.array([]),
            'bin_edges': np.array([])
        }

    # 2) Compute ISIs
    isi = np.diff(spike_times).astype('timedelta64[ns]').astype(float)/1e9

    # 3) Bin firing rate per interval
    total_time = (df['Time'].max() - df['Time'].min()).total_seconds()

    if np.isnan(total_time) or total_time <= 0:
        return {
            'spike_times': spike_times,
            'isi': isi,
            'firing_rate': np.array([]),
            'bin_edges': np.array([])
        }

    n_bins = int(np.ceil(total_time / bin_size_s))
    bin_edges = np.linspace(0, n_bins*bin_size_s, n_bins+1)
    # Fix --> Ensure both types are np.araray
    t0 = df['Time'].min().to_datetime64()
    spike_seconds = (spike_times.astype('datetime64[ns]') - t0).astype('timedelta64[ns]').astype(float) / 1e9
    counts, edges = np.histogram(spike_seconds, bins=bin_edges)
    fr = counts / bin_size_s
    return {'spike_times': spike_times, 'isi': isi, 'firing_rate': fr, 'bin_edges': edges}
