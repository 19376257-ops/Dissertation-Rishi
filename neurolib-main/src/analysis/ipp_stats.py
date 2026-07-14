import pandas as pd
import numpy as np
from scipy.stats import zscore, pearsonr, ks_2samp
from statsmodels.tsa.stattools import grangercausalitytests


def compute_firing_rate_zscore(df, channel=None, bin_size_s=1.0):
    """
    Compute firing rate z-scores for neural activity data with binning.

    1) Check if Time column is in datetime format, convert if needed
    2) Filter data by channel if specified
    3) Check if data is empty and return error if so
    4) Get start and end times from data
    5) Validate time range for NaT values
    6) Calculate duration in seconds
    7) Validate duration is positive
    8) Calculate number of bins based on bin size
    9) Validate number of bins is sufficient
    10) Create time bins using date_range
    11) Count spikes in each bin
    12) Calculate firing rates by dividing counts by bin size
    13) Compute z-scores of firing rates
    14) Return results including spike counts, firing rates, and z-scores

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter channel: int
                    :parameter bin_size_s: float

                Returns:
                    :return result: dict
    """
    if not pd.api.types.is_datetime64_any_dtype(df['Time']):
        df['Time'] = pd.to_datetime(df['Time'])

    data = df[df['channel'] == channel] if channel is not None else df

    if data.empty:
        return {'channel': channel, 'error': 'No data available for this channel'}

    start, end = data['Time'].min(), data['Time'].max()

    if pd.isna(start) or pd.isna(end):
        return {'channel': channel, 'error': 'Invalid time range (NaT values detected)'}

    duration = (end - start).total_seconds()

    if np.isnan(duration) or duration <= 0:
        return {'channel': channel, 'error': f'Invalid duration: {duration}'}

    num_bins = int(duration / bin_size_s)

    if num_bins < 2:
        return {'channel': channel, 'error': f'Duration too short for bin size {bin_size_s}s'}

    bins = pd.date_range(start, end, periods=num_bins + 1)
    spike_counts = np.histogram(data['Time'], bins=bins)[0]
    firing_rates = spike_counts / bin_size_s

    return {
        'channel': channel,
        'spike_counts': spike_counts,
        'firing_rates': firing_rates,
        'z_scores': zscore(firing_rates),
        'bin_centers': [(bins[i] + (bins[i + 1] - bins[i]) / 2).timestamp() - start.timestamp() for i in
                        range(num_bins)]
    }


def compute_kernel_density(df, channel=None, bin_size_s=1.0):
    """
    Compute kernel density estimation of firing rates for neural activity data.

    1) Call compute_firing_rate_zscore to get firing rate data
    2) Check if there was an error in computing firing rates
    3) Extract channel and firing rates information
    4) Return simplified result with channel and firing rates

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter channel: int
                    :parameter bin_size_s: float

                Returns:
                    :return result: dict
    """
    fr = compute_firing_rate_zscore(df, channel, bin_size_s)

    if 'error' in fr:
        return fr

    return {'channel': fr.get('channel'), 'firing_rates': fr['firing_rates']}


def compute_descriptive_statistics(df: pd.DataFrame, bin_size_s: float = 1.0) -> dict:
    """
    Compute descriptive statistics for firing rates across channels.

    1) Init result dictionary
    2) Check if Time column is in datetime format, convert if needed
    3) Get start and end times from data
    4) Calculate time range in seconds
    5) Calculate number of bins based on bin size
    6) Validate number of bins is sufficient
    7) Get unique channels from data
    8) Init data structures for channel data and collective spike counts
    9) For each channel, bin the data and count spikes in each bin
    10) Accumulate spike counts across all channels
    11) Calculate overall statistics (mean, median, std, var, max, min)
    12) Calculate channel-specific statistics for each channel
    13) Return comprehensive statistical results

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter bin_size_s: float

                Returns:
                    :return result: dict
    """
    result = {}

    if not pd.api.types.is_datetime64_any_dtype(df['Time']):
        df['Time'] = pd.to_datetime(df['Time'])

    start_time = df['Time'].min()
    end_time = df['Time'].max()

    time_range_s = (end_time - start_time).total_seconds()
    num_bins = int(time_range_s / bin_size_s)

    if num_bins < 2:
        return {
            'error': f"Time range too small ({time_range_s:.2f}s) for bin size {bin_size_s}s"
        }

    channels = df['channel'].unique()

    all_channel_data = {}
    collective_spike_counts = np.zeros(num_bins)

    for channel in channels:
        channel_data = df[df['channel'] == channel]

        bins = pd.date_range(start=start_time, end=end_time, periods=num_bins + 1)
        spike_counts = np.zeros(num_bins)

        for i in range(num_bins):
            bin_start = bins[i]
            bin_end = bins[i + 1]
            spike_counts[i] = len(channel_data[(channel_data['Time'] >= bin_start) &
                                               (channel_data['Time'] < bin_end)])

        all_channel_data[channel] = spike_counts

        collective_spike_counts += spike_counts

    # Run calculations
    result['mean_firing_rate'] = np.mean(collective_spike_counts) / bin_size_s
    result['median_firing_rate'] = np.median(collective_spike_counts) / bin_size_s
    result['std_firing_rate'] = np.std(collective_spike_counts) / bin_size_s
    result['var_firing_rate'] = np.var(collective_spike_counts) / (bin_size_s ** 2)
    result['max_firing_rate'] = np.max(collective_spike_counts) / bin_size_s
    result['min_firing_rate'] = np.min(collective_spike_counts) / bin_size_s

    # Store channel-specific stats
    channel_stats = {}
    for channel in channels:
        channel_stats[str(channel)] = {
            'mean': np.mean(all_channel_data[channel]) / bin_size_s,
            'median': np.median(all_channel_data[channel]) / bin_size_s,
            'std': np.std(all_channel_data[channel]) / bin_size_s,
            'max': np.max(all_channel_data[channel]) / bin_size_s,
            'min': np.min(all_channel_data[channel]) / bin_size_s
        }

    result['channel_stats'] = channel_stats

    return result


def compute_inferential_statistics(df: pd.DataFrame, bin_size_s: float = 1.0) -> dict:
    """
    Compute inferential statistics between pairs of channels to analyse relationships.

    1) Check if Time column is in datetime format, convert if needed
    2) Get start and end times from data
    3) Calculate duration and number of bins
    4) Validate number of bins is sufficient
    5) Get unique channels from data
    6) Validate at least 2 channels exist
    7) Create time bins and count spikes for each channel
    8) Init list for pairwise test results
    9) For each pair of channels:
       a. Perform Kolmogorov-Smirnov test to compare distributions
       b. Calculate Pearson correlation between channels
       c. Perform Granger causality tests in both directions
       d. Store results for each pair
    10) Return all pairwise test results

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter bin_size_s: float

                Returns:
                    :return result: dict
    """
    if not pd.api.types.is_datetime64_any_dtype(df['Time']):
        df['Time'] = pd.to_datetime(df['Time'])

    start, end = df['Time'].min(), df['Time'].max()
    duration = (end - start).total_seconds()
    num_bins = int(duration / bin_size_s)

    if num_bins < 2:
        return {'error': f'Time range too small ({duration:.2f}s) for bin size {bin_size_s}s'}

    channels = df['channel'].unique()
    if len(channels) < 2:
        return {'error': f'Need at least 2 channels, found {len(channels)}'}

    bins = pd.date_range(start, end, periods=num_bins + 1)
    all_data = {
        ch: np.histogram(df[df['channel'] == ch]['Time'], bins=bins)[0]
        for ch in channels
    }

    pairwise_tests = []
    for i, ch1 in enumerate(channels):
        for ch2 in channels[i + 1:]:
            result = {'channel1': int(ch1), 'channel2': int(ch2)}

            # KS Test
            try:
                stat, p = ks_2samp(all_data[ch1], all_data[ch2])
                result['ks_test'] = {'statistic': float(stat), 'p_value': float(p)}
            except Exception as e:
                result['ks_test'] = {'error': str(e)}

            # Pearson Correlation
            try:
                r, p = pearsonr(all_data[ch1], all_data[ch2])
                result['correlation'] = {'r': float(r), 'p_value': float(p)}
            except Exception as e:
                result['correlation'] = {'error': str(e)}

            # Granger Causality (both directions)
            try:
                maxlag = min(10, num_bins // 10)
                if maxlag >= 1:
                    def run_gc(x, y):
                        data = np.column_stack((x, y))
                        out = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                        return [float(out[lag][0]['ssr_ftest'][1]) for lag in range(1, maxlag + 1)]

                    gc12 = run_gc(all_data[ch1], all_data[ch2])
                    gc21 = run_gc(all_data[ch2], all_data[ch1])

                    result['granger_causality'] = {
                        'ch1_causes_ch2': {'p_values': gc12, 'min_p_value': min(gc12)},
                        'ch2_causes_ch1': {'p_values': gc21, 'min_p_value': min(gc21)}
                    }
                else:
                    result['granger_causality'] = {'error': f'Not enough bins for Granger test (need ≥ {maxlag})'}
            except Exception as e:
                result['granger_causality'] = {'error': str(e)}

            pairwise_tests.append(result)

    return {'pairwise_tests': pairwise_tests}
