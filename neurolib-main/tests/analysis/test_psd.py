import pytest
import pandas as pd
import numpy as np
from src.analysis.psd import compute_power_spectral_density

def test_compute_power_spectral_density_basic(sample_spike_data):
    """Test compute_power_spectral_density with sample spike data"""

    channel = sample_spike_data['channel'].iloc[0]

    result = compute_power_spectral_density(
        df=sample_spike_data,
        channel=channel,
        fs=1000.0
    )

    # Check
    assert 'freqs' in result
    assert 'psd' in result
    assert len(result['freqs']) > 0
    assert len(result['psd']) > 0
    assert len(result['freqs']) == len(result['psd'])

def test_compute_power_spectral_density_empty():
    """Test compute_power_spectral_density with empty DataFrame."""

    empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])

    result = compute_power_spectral_density(
        df=empty_df,
        channel=1,
        fs=1000.0
    )

    # Check
    assert 'freqs' in result
    assert 'psd' in result
    assert len(result['freqs']) == 0
    assert len(result['psd']) == 0

def test_compute_power_spectral_density_nonexistent_channel(sample_spike_data):
    """Test compute_power_spectral_density with a channel that doesn't exist"""

    max_channel = sample_spike_data['channel'].max()
    nonexistent_channel = max_channel + 1

    result = compute_power_spectral_density(
        df=sample_spike_data,
        channel=nonexistent_channel,
        fs=1000.0
    )

    # Check
    assert 'freqs' in result
    assert 'psd' in result
    assert len(result['freqs']) == 0
    assert len(result['psd']) == 0

def test_compute_power_spectral_density_with_custom_params(sample_spike_data):
    """Test compute_power_spectral_density with custom parameters"""

    channel = sample_spike_data['channel'].iloc[0]

    df = sample_spike_data.copy()
    for i in range(10):
        new_rows = df.copy()
        new_rows['Time'] = new_rows['Time'] + pd.Timedelta(seconds=i+1)
        df = pd.concat([df, new_rows])

    result = compute_power_spectral_density(
        df=df,
        channel=channel,
        fs=1000.0,
        nperseg=8,
        noverlap=4
    )

    # Check
    assert 'freqs' in result
    assert 'psd' in result
    assert len(result['freqs']) > 0
    assert len(result['psd']) > 0
    assert len(result['freqs']) == len(result['psd'])

def test_compute_power_spectral_density_with_default_params(sample_spike_data):
    """Test compute_power_spectral_density with default parameter"""

    channel = sample_spike_data['channel'].iloc[0]

    result = compute_power_spectral_density(
        df=sample_spike_data,
        channel=channel,
        fs=1000.0,
        nperseg=None,
        noverlap=None
    )

    # Check
    assert 'freqs' in result
    assert 'psd' in result
    assert len(result['freqs']) > 0
    assert len(result['psd']) > 0
    assert len(result['freqs']) == len(result['psd'])
