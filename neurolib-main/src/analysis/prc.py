# src/analysis/prc.py

import numpy as np
import pandas as pd

def calculate_prc_times(spike_times: np.ndarray,
                        frequency: float,
                        num_bins: int = 50) -> np.ndarray:
    """
    Calculate phase response curve (PRC) from spike times and stimulus frequency.

    1) Validate frequency is positive
    2) Calculate period from frequency
    3) Convert spike times to phases (0-1)
    4) Create phase bins
    5) Initialise PRC array and bin count array
    6) For each bin, calculate phase-dependent changes in inter-spike intervals
    7) Print diagnostic information about bin counts
    8) Apply Gaussian smoothing to the PRC
    9) Return the computed PRC

                Parameters:
                    :parameter spike_times: np.ndarray
                    :parameter frequency: float
                    :parameter num_bins: int

                Returns:
                    :return prc: np.ndarray
    """
    if frequency <= 0:
        raise ValueError("frequency must be > 0")
    period = 1.0 / frequency
    phases = (spike_times % period) / period
    edges = np.linspace(0, 1, num_bins+1)
    prc = np.zeros(num_bins, dtype=float)

    bin_counts = np.zeros(num_bins, dtype=int)

    for i in range(num_bins):
        mask = (phases >= edges[i]) & (phases < edges[i+1])
        ts = np.sort(spike_times[mask])
        bin_counts[i] = ts.size
        if ts.size > 1:
            isis = np.diff(ts)
            mu = isis.mean()
            if mu > 0:
                prc[i] = np.mean((isis - mu) / mu)

    print(f"Bin counts: {bin_counts}")
    print(f"Total spikes: {len(spike_times)}")
    print(f"Bins with enough spikes: {np.sum(bin_counts > 1)}/{num_bins}")

    try:
        from scipy.ndimage import gaussian_filter1d
        prc = gaussian_filter1d(prc, sigma=1)
    except ImportError:
        print("Warning: scipy.ndimage not available, skipping smoothing")

    return prc


def compute_prc(df: pd.DataFrame,
                channel: int,
                *,
                # for STDP:
                train_timestamps: pd.DatetimeIndex = None,
                train_duration_ms: int = None,
                stim_frequency: float = None,
                # for Phase-Encoding:
                phase_frequencies: dict = None,
                num_bins: int = 50
               ) -> dict:
    """
    Compute phase response curves (PRCs) for neural data in STDP or PSE mode.

    1) Init result dictionary with channel and phase bins
    2) Filter data to include only the specified channel
    3) Check if there are spikes for the channel
    4) Ensure Time column is in datetime format
    5) Determine which mode to use (STDP or PSE) based on parameters
    6) For STDP mode:
       a. Process each train timestamp
       b. Extract spikes within the train duration
       c. Calculate PRC for each train
       d. Average PRCs across trains
    7) For PSE mode:
       a. Process each condition with its frequency
       b. Filter spikes by condition
       c. Check if there are enough spikes across phases
       d. Calculate PRC for each condition
       e. Average PRCs across conditions
    8) Handle error cases and return results

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter channel: int
                    :parameter train_timestamps: pd.DatetimeIndex
                    :parameter train_duration_ms: int
                    :parameter stim_frequency: float
                    :parameter phase_frequencies: dict
                    :parameter num_bins: int

                Returns:
                    :return result: dict
    """
    result = {'channel': channel, 'phase_bins': np.linspace(0,1,num_bins)}
    # filter to this channel
    cdf = df[df['channel'] == channel].copy()
    if cdf.empty:
        print(f"No spikes found for channel {channel}")
        return {**result, 'error': f'no spikes for channel {channel}'}

    # Ensure Time column is in the correct format
    if not pd.api.types.is_datetime64_any_dtype(cdf['Time']):
        cdf['Time'] = pd.to_datetime(cdf['Time'], format='mixed', utc=True).dt.tz_convert(None)

    # STDP Mode
    if train_timestamps is not None and train_duration_ms is not None and stim_frequency is not None:
        prcs = []
        for i, t0 in enumerate(train_timestamps):
            t1 = t0 + pd.Timedelta(milliseconds=train_duration_ms)
            seg = cdf[(cdf['Time'] >= t0) & (cdf['Time'] < t1)]
            if seg.empty:
                print(f"No spikes found for train {i} in time range {t0} to {t1}")
                continue
            rel = (seg['Time'] - t0).dt.total_seconds().values
            prc_i = calculate_prc_times(rel, stim_frequency, num_bins)
            result[f'train_{i}'] = prc_i
            prcs.append(prc_i)

        # Fallback
        if not prcs:
            print(f"No valid PRCs computed for any train")
            return {**result, 'error': 'no valid PRCs computed'}

        # average across trains
        result['prc'] = np.mean(np.stack(prcs, axis=0), axis=0)
        return result

    # else, Phase-Encoding mode
    if phase_frequencies:
        valid_prcs = False
        for cond, freq in phase_frequencies.items():
            if freq is None:
                continue

            # Filter by condition
            seg = df[(df['channel'] == channel) & (df['condition'] == cond)].copy()
            if seg.empty:
                print(f"No spikes found for condition '{cond}' on channel {channel}")
                continue

            print(f"Found {len(seg)} spikes for condition '{cond}' on channel {channel}")

            # Check if we have enough spikes in different phases to compute a meaningful PRC
            t0 = seg['Time'].min()
            rel = (seg['Time'] - t0).dt.total_seconds().values

            if len(rel) < 2:
                print(f"Not enough spikes ({len(rel)}) for condition '{cond}' to compute PRC")
                continue

            # Calculate period and phases for checking distribution
            period = 1.0 / freq
            phases = (rel % period) / period

            # Check if spikes are distributed across different phases
            edges = np.linspace(0, 1, num_bins+1)
            bin_counts = np.zeros(num_bins, dtype=int)
            for i in range(num_bins):
                mask = (phases >= edges[i]) & (phases < edges[i+1])
                bin_counts[i] = np.sum(mask)

            bins_with_spikes = np.sum(bin_counts > 0)
            print(f"Bins with at least one spike: {bins_with_spikes}/{num_bins}")
            if bins_with_spikes < num_bins * 0.1:
                print(f"Spikes not well distributed across phases for condition '{cond}'")
                print(f"WARNING: PRC may not be meaningful for condition '{cond}'")

            prc_c = calculate_prc_times(rel, freq, num_bins)
            result[f'prc_{cond}'] = prc_c
            valid_prcs = True

        prc_list = [v for k,v in result.items() if k.startswith('prc_')]
        if prc_list:
            result['prc'] = np.mean(np.stack(prc_list,axis=0),axis=0)
            return result
        elif valid_prcs:
            return {**result, 'error': 'PRCs computed but not added to result'}
        else:
            return {**result, 'error': 'no valid PRCs computed for any condition'}

    return {**result,
            'error': 'must supply either (train_timestamps, train_duration_ms, stim_frequency) or phase_frequencies'}
