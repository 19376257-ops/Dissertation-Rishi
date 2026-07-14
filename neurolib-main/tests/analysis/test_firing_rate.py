import pytest
import pandas as pd
import numpy as np
import datetime
from src.analysis.firing_rate import compute_firing_rates

def test_compute_firing_rates_basic(sample_spike_data):
    """Test compute_firing_rates with sample spike data."""

    result = compute_firing_rates(
        df=sample_spike_data,
        threshold=-10.0,
        bin_size_s=1.0
    )

    # Check
    assert 'spike_times' in result
    assert 'isi' in result
    assert 'firing_rate' in result
    assert 'bin_edges' in result

    if len(sample_spike_data) > 0:
        assert len(result['spike_times']) > 0

        # If there are at least 2 spikes, ISI should not be empty
        if len(result['spike_times']) >= 2:
            assert len(result['isi']) > 0
            assert len(result['firing_rate']) > 0
            assert len(result['bin_edges']) > 0

def test_compute_firing_rates_empty():
    """Test compute_firing_rates with empty DataFrame."""

    empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])

    result = compute_firing_rates(
        df=empty_df,
        threshold=0.0,
        bin_size_s=1.0
    )

    # Check
    assert 'spike_times' in result
    assert 'isi' in result
    assert 'firing_rate' in result
    assert 'bin_edges' in result
    assert len(result['spike_times']) == 0
    assert len(result['isi']) == 0
    assert len(result['firing_rate']) == 0
    assert len(result['bin_edges']) == 0

def test_compute_firing_rates_no_spikes():
    """Test compute_firing_rates with no spikes above threshold."""

    df = pd.DataFrame({
        'Time': [datetime.datetime(2020, 1, 1, 12, 0, 0), 
                 datetime.datetime(2020, 1, 1, 12, 0, 1)],
        'channel': [1, 1],
        'Amplitude': [-5.0, -6.0]
    })

    result = compute_firing_rates(
        df=df,
        threshold=0.0,
        bin_size_s=1.0
    )

    # Check
    assert 'spike_times' in result
    assert 'isi' in result
    assert 'firing_rate' in result
    assert 'bin_edges' in result
    assert len(result['spike_times']) == 0
    assert len(result['isi']) == 0
    assert len(result['firing_rate']) == 0
    assert len(result['bin_edges']) == 0

def test_compute_firing_rates_single_spike():
    """Test compute_firing_rates with a single spike."""

    df = pd.DataFrame({
        'Time': [datetime.datetime(2020, 1, 1, 12, 0, 0)],
        'channel': [1],
        'Amplitude': [5.0]
    })

    result = compute_firing_rates(
        df=df,
        threshold=0.0,
        bin_size_s=1.0
    )

    # Check
    assert 'spike_times' in result
    assert 'isi' in result
    assert 'firing_rate' in result
    assert 'bin_edges' in result
    assert len(result['spike_times']) == 1
    assert len(result['isi']) == 0
    assert len(result['firing_rate']) == 0
    assert len(result['bin_edges']) == 0

def test_compute_firing_rates_multiple_spikes():
    """Test compute_firing_rates with multiple spikes."""

    df = pd.DataFrame({
        'Time': [datetime.datetime(2020, 1, 1, 12, 0, 0), 
                 datetime.datetime(2020, 1, 1, 12, 0, 1),
                 datetime.datetime(2020, 1, 1, 12, 0, 2),
                 datetime.datetime(2020, 1, 1, 12, 0, 3)],
        'channel': [1, 1, 1, 1],
        'Amplitude': [5.0, 6.0, 7.0, 8.0]
    })

    result = compute_firing_rates(
        df=df,
        threshold=0.0,
        bin_size_s=1.0
    )

    # Check
    assert 'spike_times' in result
    assert 'isi' in result
    assert 'firing_rate' in result
    assert 'bin_edges' in result
    assert len(result['spike_times']) == 4
    assert len(result['isi']) == 3
    assert len(result['firing_rate']) > 0
    assert len(result['bin_edges']) > 0
    assert len(result['bin_edges']) == len(result['firing_rate']) + 1

def test_compute_firing_rates_zero_duration():
    """Test compute_firing_rates with zero duration."""

    df = pd.DataFrame({
        'Time': [datetime.datetime(2020, 1, 1, 12, 0, 0), 
                 datetime.datetime(2020, 1, 1, 12, 0, 0)],
        'channel': [1, 1],
        'Amplitude': [5.0, 6.0]
    })

    result = compute_firing_rates(
        df=df,
        threshold=0.0,
        bin_size_s=1.0
    )

    # Check
    assert 'spike_times' in result
    assert 'isi' in result
    assert 'firing_rate' in result
    assert 'bin_edges' in result
    assert len(result['spike_times']) == 2
    assert len(result['isi']) == 1
    assert len(result['firing_rate']) == 0
    assert len(result['bin_edges']) == 0
