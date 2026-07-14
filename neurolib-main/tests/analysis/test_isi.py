import pytest
import pandas as pd
import numpy as np
from src.analysis.isi import compute_isi

def test_compute_isi_with_sample_data(sample_spike_data):
    """Test compute_isi with sample spike data"""

    channel = sample_spike_data['channel'].iloc[0]
    
    result = compute_isi(
        df=sample_spike_data,
        channel=channel
    )
    
    # Check
    assert 'channel' in result
    assert 'isi' in result
    assert 'mean_isi' in result
    assert 'std_isi' in result
    assert result['channel'] == channel
    
    channel_data = sample_spike_data[sample_spike_data['channel'] == channel]

    if len(channel_data) > 1:
        assert len(result['isi']) > 0
        assert not np.isnan(result['mean_isi'])
        assert not np.isnan(result['std_isi'])

def test_compute_isi_without_channel(sample_spike_data):
    """Test compute_isi without specifying a channel"""

    result = compute_isi(
        df=sample_spike_data
    )
    
    # Check
    assert 'isi' in result
    assert 'mean_isi' in result
    assert 'std_isi' in result
    assert 'channel' not in result

    if len(sample_spike_data) > 1:
        assert len(result['isi']) > 0
        assert not np.isnan(result['mean_isi'])
        assert not np.isnan(result['std_isi'])

def test_compute_isi_with_empty_data():
    """Test compute_isi with empty DataFrame."""

    empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])
    
    result = compute_isi(
        df=empty_df,
        channel=1
    )
    
    # Check
    assert 'channel' in result
    assert 'isi' in result
    assert 'mean_isi' in result
    assert 'std_isi' in result
    assert len(result['isi']) == 0
    assert np.isnan(result['mean_isi'])
    assert np.isnan(result['std_isi'])

def test_compute_isi_with_nonexistent_channel(sample_spike_data):
    """Test compute_isi with a channel that doesn't exist in the data"""

    max_channel = sample_spike_data['channel'].max()
    nonexistent_channel = max_channel + 1
    
    result = compute_isi(
        df=sample_spike_data,
        channel=nonexistent_channel
    )
    
    # Check
    assert 'channel' in result
    assert 'isi' in result
    assert 'mean_isi' in result
    assert 'std_isi' in result
    assert result['channel'] == nonexistent_channel
    assert len(result['isi']) == 0
    assert np.isnan(result['mean_isi'])
    assert np.isnan(result['std_isi'])

def test_compute_isi_with_single_spike(sample_spike_data):
    """Test compute_isi with a single spike"""

    channel = sample_spike_data['channel'].iloc[0]
    single_spike_df = sample_spike_data[sample_spike_data['channel'] == channel].iloc[:1]
    
    result = compute_isi(
        df=single_spike_df,
        channel=channel
    )
    
    # Check
    assert 'channel' in result
    assert 'isi' in result
    assert 'mean_isi' in result
    assert 'std_isi' in result
    assert result['channel'] == channel
    assert len(result['isi']) == 0
    assert np.isnan(result['mean_isi'])
    assert np.isnan(result['std_isi'])