import pytest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import patch, MagicMock

from src.analysis.ipp_stats import (
    compute_firing_rate_zscore,
    compute_kernel_density,
    compute_descriptive_statistics,
    compute_inferential_statistics
)

def test_compute_firing_rate_zscore_basic(sample_spike_data):
    """Test compute_firing_rate_zscore with basic valid data."""

    channel = sample_spike_data['channel'].iloc[0]

    result = compute_firing_rate_zscore(
        df=sample_spike_data,
        channel=channel,
        bin_size_s=1.0
    )

    # Check
    assert 'channel' in result
    assert 'spike_counts' in result
    assert 'firing_rates' in result
    assert 'z_scores' in result
    assert 'bin_centers' in result
    assert result['channel'] == channel
    assert len(result['spike_counts']) > 0
    assert len(result['firing_rates']) > 0
    assert len(result['z_scores']) > 0
    assert len(result['bin_centers']) > 0
    assert len(result['spike_counts']) == len(result['firing_rates'])
    assert len(result['firing_rates']) == len(result['z_scores'])
    assert len(result['z_scores']) == len(result['bin_centers'])

def test_compute_firing_rate_zscore_empty():
    """Test compute_firing_rate_zscore with empty DataFrame."""

    empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])

    result = compute_firing_rate_zscore(
        df=empty_df,
        channel=1,
        bin_size_s=1.0
    )

    # Check
    assert 'channel' in result
    assert 'error' in result
    assert 'No data available' in result['error']

def test_compute_firing_rate_zscore_nonexistent_channel(sample_spike_data):
    """Test compute_firing_rate_zscore with a channel that doesn't exist in the data."""

    max_channel = sample_spike_data['channel'].max()
    nonexistent_channel = max_channel + 1

    result = compute_firing_rate_zscore(
        df=sample_spike_data,
        channel=nonexistent_channel,
        bin_size_s=1.0
    )

    # Check
    assert 'channel' in result
    assert 'error' in result
    assert result['channel'] == nonexistent_channel
    assert 'No data available' in result['error']

def test_compute_firing_rate_zscore_short_duration():
    """Test compute_firing_rate_zscore with a duration that's too short for the bin size."""

    df = pd.DataFrame({
        'Time': [datetime.datetime(2020, 1, 1, 12, 0, 0), 
                 datetime.datetime(2020, 1, 1, 12, 0, 1)],
        'channel': [1, 1],
        'Amplitude': [-5.0, -6.0]
    })

    result = compute_firing_rate_zscore(
        df=df,
        channel=1,
        bin_size_s=1.0
    )


    # Check
    assert 'channel' in result
    assert 'error' in result
    assert 'Duration too short' in result['error']

def test_compute_kernel_density_basic(sample_spike_data):
    """Test compute_kernel_density with basic valid data"""
    channel = sample_spike_data['channel'].iloc[0]

    # Mock
    with patch('src.analysis.ipp_stats.compute_firing_rate_zscore') as mock_zscore:
        mock_zscore.return_value = {
            'channel': channel,
            'spike_counts': np.array([1, 2, 3]),
            'firing_rates': np.array([1.0, 2.0, 3.0]),
            'z_scores': np.array([0.0, 0.5, 1.0]),
            'bin_centers': np.array([0.5, 1.5, 2.5])
        }

        result = compute_kernel_density(
            df=sample_spike_data,
            channel=channel,
            bin_size_s=1.0
        )

        # Check
        assert 'channel' in result
        assert 'firing_rates' in result
        assert result['channel'] == channel
        assert np.array_equal(result['firing_rates'], np.array([1.0, 2.0, 3.0]))

def test_compute_kernel_density_error(sample_spike_data):
    """Test compute_kernel_density when compute_firing_rate_zscore returns an error."""

    channel = sample_spike_data['channel'].iloc[0]

    # Mock
    with patch('src.analysis.ipp_stats.compute_firing_rate_zscore') as mock_zscore:
        mock_zscore.return_value = {
            'channel': channel,
            'error': 'Test error'
        }

        result = compute_kernel_density(
            df=sample_spike_data,
            channel=channel,
            bin_size_s=1.0
        )

        # Check
        assert 'channel' in result
        assert 'error' in result
        assert result['error'] == 'Test error'

def test_compute_descriptive_statistics_basic(sample_spike_data):
    """Test compute_descriptive_statistics with basic valid data"""

    result = compute_descriptive_statistics(
        df=sample_spike_data,
        bin_size_s=1.0
    )

    # Check
    assert 'mean_firing_rate' in result
    assert 'median_firing_rate' in result
    assert 'std_firing_rate' in result
    assert 'var_firing_rate' in result
    assert 'max_firing_rate' in result
    assert 'min_firing_rate' in result
    assert 'channel_stats' in result

    # Check that channel_stats contains entries for each channel
    channels = sample_spike_data['channel'].unique()
    for channel in channels:
        assert str(channel) in result['channel_stats']

        # Check that each channel's stats have the expected structure
        channel_stats = result['channel_stats'][str(channel)]
        assert 'mean' in channel_stats
        assert 'median' in channel_stats
        assert 'std' in channel_stats
        assert 'max' in channel_stats
        assert 'min' in channel_stats

def test_compute_descriptive_statistics_short_duration():
    """Test compute_descriptive_statistics with a duration that's too short for the bin size"""

    df = pd.DataFrame({
        'Time': [datetime.datetime(2020, 1, 1, 12, 0, 0), 
                 datetime.datetime(2020, 1, 1, 12, 0, 1)],
        'channel': [1, 1],
        'Amplitude': [-5.0, -6.0]
    })

    result = compute_descriptive_statistics(
        df=df,
        bin_size_s=1.0
    )

    # Check
    assert 'error' in result
    assert 'Time range too small' in result['error']

def test_compute_inferential_statistics_basic(sample_spike_data):
    """Test compute_inferential_statistics with basic valid data."""

    channels = sample_spike_data['channel'].unique()
    if len(channels) < 2:
        second_channel = max(channels) + 1
        additional_data = sample_spike_data.copy()
        additional_data['channel'] = second_channel
        sample_spike_data = pd.concat([sample_spike_data, additional_data], ignore_index=True)

    # Mock
    with patch('src.analysis.ipp_stats.ks_2samp') as mock_ks, \
         patch('src.analysis.ipp_stats.pearsonr') as mock_pearson, \
         patch('src.analysis.ipp_stats.grangercausalitytests') as mock_granger:

        mock_ks.return_value = (0.3, 0.7) # statistic, p-value
        mock_pearson.return_value = (0.5, 0.6) # correlation, p-value

        # Mock grangercausalitytests
        mock_granger_result = {}
        for lag in range(1, 11):
            mock_granger_result[lag] = [{
                'ssr_ftest': [0, 0.1] # statistic, p-value
            }]
        mock_granger.return_value = mock_granger_result

        result = compute_inferential_statistics(
            df=sample_spike_data,
            bin_size_s=0.1
        )

        # Check
        assert 'pairwise_tests' in result
        assert len(result['pairwise_tests']) > 0
        for test in result['pairwise_tests']:
            assert 'channel1' in test
            assert 'channel2' in test
            assert 'ks_test' in test
            assert 'correlation' in test
            assert 'granger_causality' in test
            assert 'statistic' in test['ks_test']
            assert 'p_value' in test['ks_test']
            assert 'r' in test['correlation']
            assert 'p_value' in test['correlation']
            assert 'ch1_causes_ch2' in test['granger_causality']
            assert 'ch2_causes_ch1' in test['granger_causality']
            assert 'p_values' in test['granger_causality']['ch1_causes_ch2']
            assert 'min_p_value' in test['granger_causality']['ch1_causes_ch2']
            assert 'p_values' in test['granger_causality']['ch2_causes_ch1']
            assert 'min_p_value' in test['granger_causality']['ch2_causes_ch1']

def test_compute_inferential_statistics_short_duration():
    """Test compute_inferential_statistics with a duration that's too short for the bin size"""

    df = pd.DataFrame({
        'Time': [datetime.datetime(2020, 1, 1, 12, 0, 0), 
                 datetime.datetime(2020, 1, 1, 12, 0, 1)],
        'channel': [1, 2],
        'Amplitude': [-5.0, -6.0]
    })

    result = compute_inferential_statistics(
        df=df,
        bin_size_s=1.0
    )

    # Check
    assert 'error' in result
    assert 'Time range too small' in result['error']

def test_compute_inferential_statistics_not_enough_channels():
    """Test compute_inferential_statistics with not enough channels"""

    df = pd.DataFrame({
        'Time': [datetime.datetime(2020, 1, 1, 12, 0, 0), 
                 datetime.datetime(2020, 1, 1, 12, 0, 1)],
        'channel': [1, 1],
        'Amplitude': [-5.0, -6.0]
    })

    result = compute_inferential_statistics(
        df=df,
        bin_size_s=0.1
    )

    # Check
    assert 'error' in result
    assert 'Need at least 2 channels' in result['error']

def test_compute_inferential_statistics_exception_handling(sample_spike_data):
    """Test compute_inferential_statistics with exceptions in statistical functions"""

    channels = sample_spike_data['channel'].unique()
    if len(channels) < 2:
        second_channel = max(channels) + 1
        additional_data = sample_spike_data.copy()
        additional_data['channel'] = second_channel
        sample_spike_data = pd.concat([sample_spike_data, additional_data], ignore_index=True)

    # Mock
    with patch('src.analysis.ipp_stats.ks_2samp') as mock_ks, \
         patch('src.analysis.ipp_stats.pearsonr') as mock_pearson, \
         patch('src.analysis.ipp_stats.grangercausalitytests') as mock_granger:

        mock_ks.side_effect = ValueError("KS test error")
        mock_pearson.side_effect = ValueError("Pearson correlation error")
        mock_granger.side_effect = ValueError("Granger causality error")

        result = compute_inferential_statistics(
            df=sample_spike_data,
            bin_size_s=0.1
        )

        # Check
        assert 'pairwise_tests' in result
        assert len(result['pairwise_tests']) > 0
        for test in result['pairwise_tests']:
            assert 'error' in test['ks_test']
            assert 'KS test error' in test['ks_test']['error']
            assert 'error' in test['correlation']
            assert 'Pearson correlation error' in test['correlation']['error']
            assert 'error' in test['granger_causality']
            assert 'Granger causality error' in test['granger_causality']['error']
