import pytest
import pandas as pd
import numpy as np
from src.analysis.fft import compute_fft

def test_compute_fft_with_sample_data(sample_spike_data):
    """Test compute_fft with sample spike data"""

    channel = sample_spike_data['channel'].iloc[0]
    
    result = compute_fft(
        df=sample_spike_data,
        channel=channel,
        resample_interval=1,
        freq_cap=10.0
    )
    
    # Check
    assert 'freqs' in result
    assert 'amplitudes' in result
    assert len(result['freqs']) > 0
    assert len(result['amplitudes']) > 0
    assert len(result['freqs']) == len(result['amplitudes'])
    assert np.all(result['freqs'] <= 10.0)

def test_compute_fft_with_empty_data():
    """Test compute_fft with empty DataFrame"""

    empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])
    
    result = compute_fft(
        df=empty_df,
        channel=1,
        resample_interval=1,
        freq_cap=10.0
    )
    
    # Check
    assert 'freqs' in result
    assert 'amplitudes' in result
    assert len(result['freqs']) == 0
    assert len(result['amplitudes']) == 0

def test_compute_fft_with_nonexistent_channel(sample_spike_data):
    """Test compute_fft with a channel that doesn't exist in the data"""

    max_channel = sample_spike_data['channel'].max()
    nonexistent_channel = max_channel + 1
    
    result = compute_fft(
        df=sample_spike_data,
        channel=nonexistent_channel,
        resample_interval=1,
        freq_cap=10.0
    )
    
    # Check
    assert 'freqs' in result
    assert 'amplitudes' in result
    assert len(result['freqs']) == 0
    assert len(result['amplitudes']) == 0

def test_compute_fft_with_different_parameters(sample_spike_data):
    """Test compute_fft with different parameters"""

    channel = sample_spike_data['channel'].iloc[0]
    
    result1 = compute_fft(
        df=sample_spike_data,
        channel=channel,
        resample_interval=1,
        freq_cap=5.0
    )
    
    result2 = compute_fft(
        df=sample_spike_data,
        channel=channel,
        resample_interval=2,
        freq_cap=10.0
    )
    
    # Check
    assert not np.array_equal(result1['freqs'], result2['freqs'])
    assert not np.array_equal(result1['amplitudes'], result2['amplitudes'])
    assert np.all(result1['freqs'] <= 5.0)
    assert np.all(result2['freqs'] <= 10.0)