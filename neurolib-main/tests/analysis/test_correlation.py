import pytest
import pandas as pd
import numpy as np
import datetime
from src.analysis.correlation import compute_cross_correlation, compute_auto_correlation

def test_compute_cross_correlation_with_sample_data(sample_spike_data):
    """Test compute_cross_correlation with sample spike data"""

    channels = sample_spike_data['channel'].unique()
    if len(channels) < 2:
        pytest.skip("Need at least 2 channels for cross-correlation test")
    
    channel1 = channels[0]
    channel2 = channels[1]
    
    result = compute_cross_correlation(
        df=sample_spike_data,
        channel1=channel1,
        channel2=channel2,
        bin_size_ms=10
    )
    
    # Check
    assert 'lags' in result
    assert 'correlation' in result
    assert 'channel1' in result
    assert 'channel2' in result
    assert result['channel1'] == channel1
    assert result['channel2'] == channel2
    assert len(result['lags']) > 0
    assert len(result['correlation']) > 0
    assert len(result['lags']) == len(result['correlation'])

def test_compute_cross_correlation_with_empty_data():
    """Test compute_cross_correlation with empty DataFrame."""

    empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])
    
    result = compute_cross_correlation(
        df=empty_df,
        channel1=1,
        channel2=2,
        bin_size_ms=10
    )
    
    # Check
    assert 'error' in result
    assert 'data' in result
    assert 'channel1' in result['data']
    assert 'channel2' in result['data']

def test_compute_cross_correlation_with_nonexistent_channels(sample_spike_data):
    """Test compute_cross_correlation with channels that don't exist in the data"""

    max_channel = sample_spike_data['channel'].max()
    nonexistent_channel = max_channel + 1
    
    result1 = compute_cross_correlation(
        df=sample_spike_data,
        channel1=nonexistent_channel,
        channel2=sample_spike_data['channel'].iloc[0],
        bin_size_ms=10
    )
    
    # Check
    assert 'error' in result1
    assert 'data' in result1
    
    result2 = compute_cross_correlation(
        df=sample_spike_data,
        channel1=nonexistent_channel,
        channel2=nonexistent_channel + 1,
        bin_size_ms=10
    )
    
    # Check
    assert 'error' in result2
    assert 'data' in result2

def test_compute_cross_correlation_with_sample_mode(sample_spike_data):
    """Test compute_cross_correlation with sample mode"""

    channels = sample_spike_data['channel'].unique()
    if len(channels) < 2:
        pytest.skip("Need at least 2 channels for cross-correlation test")
    
    channel1 = channels[0]
    channel2 = channels[1]
    
    result = compute_cross_correlation(
        df=sample_spike_data,
        channel1=channel1,
        channel2=channel2,
        bin_size_ms=10,
        sample_mode=True
    )
    
    # Check
    assert 'lags' in result
    assert 'correlation' in result
    assert 'channel1' in result
    assert 'channel2' in result

def test_compute_auto_correlation_with_sample_data(sample_spike_data):
    """Test compute_auto_correlation with sample spike data"""

    channel = sample_spike_data['channel'].iloc[0]
    
    result = compute_auto_correlation(
        df=sample_spike_data,
        channel=channel,
        bin_size_ms=1
    )
    
    # Check
    assert 'lags' in result
    assert 'correlation' in result
    assert 'channel' in result
    assert result['channel'] == channel
    assert len(result['lags']) > 0
    assert len(result['correlation']) > 0
    assert len(result['lags']) == len(result['correlation'])

def test_compute_auto_correlation_with_empty_data():
    """Test compute_auto_correlation with empty DataFrame."""

    empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])
    
    result = compute_auto_correlation(
        df=empty_df,
        channel=1,
        bin_size_ms=1
    )
    
    # Check
    assert 'error' in result
    assert 'data' in result
    assert 'channel' in result['data']

def test_compute_auto_correlation_with_nonexistent_channel(sample_spike_data):
    """Test compute_auto_correlation with a channel that doesn't exist in the data"""

    max_channel = sample_spike_data['channel'].max()
    nonexistent_channel = max_channel + 1
    
    result = compute_auto_correlation(
        df=sample_spike_data,
        channel=nonexistent_channel,
        bin_size_ms=1
    )
    
    # Check
    assert 'error' in result
    assert 'data' in result
    assert 'channel' in result['data']
    assert result['data']['channel'] == nonexistent_channel

def test_compute_auto_correlation_with_sample_mode(sample_spike_data):
    """Test compute_auto_correlation with sample mode"""

    channel = sample_spike_data['channel'].iloc[0]
    
    result = compute_auto_correlation(
        df=sample_spike_data,
        channel=channel,
        bin_size_ms=1,
        sample_mode=True
    )
    
    # Check
    assert 'lags' in result
    assert 'correlation' in result
    assert 'channel' in result