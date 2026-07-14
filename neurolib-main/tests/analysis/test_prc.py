import pytest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import patch, MagicMock

from src.analysis.prc import calculate_prc_times, compute_prc

def test_calculate_prc_times_basic():
    """Test calculate_prc_times with basic valid data"""

    spike_times = np.array([0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3])

    # Mock gaussian
    with patch('scipy.ndimage.gaussian_filter1d') as mock_filter:

        mock_filter.side_effect = lambda x, sigma: x

        result = calculate_prc_times(
            spike_times=spike_times,
            frequency=1.0, # 1 Hz, period = 1 second
            num_bins=10
        )

        # Check
        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        assert np.sum(result != 0) > 0
        mock_filter.assert_called_once()

def test_calculate_prc_times_invalid_frequency():
    """Test calculate_prc_times with invalid frequency."""

    spike_times = np.array([0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3])

    with pytest.raises(ValueError, match="frequency must be > 0"):
        calculate_prc_times(
            spike_times=spike_times,
            frequency=0.0,
            num_bins=10
        )

def test_calculate_prc_times_empty_data():
    """Test calculate_prc_times with empty data."""

    spike_times = np.array([])

    result = calculate_prc_times(
        spike_times=spike_times,
        frequency=1.0,
        num_bins=10
    )

    # Check
    assert isinstance(result, np.ndarray)
    assert len(result) == 10
    assert np.all(result == 0) # All bins  == 0

def test_calculate_prc_times_no_smoothing():
    """Test calculate_prc_times when scipy.ndimage is not available"""

    spike_times = np.array([0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3])

    # Mock
    with patch('scipy.ndimage.gaussian_filter1d', side_effect=ImportError("No scipy")):

        result = calculate_prc_times(
            spike_times=spike_times,
            frequency=1.0,
            num_bins=10
        )

        # Check
        assert isinstance(result, np.ndarray)
        assert len(result) == 10

def test_compute_prc_stdp_mode(sample_spike_data):
    """Test compute_prc in STDP mode"""

    # Mock
    with patch('src.analysis.prc.calculate_prc_times') as mock_calculate:

        mock_calculate.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        start_time = sample_spike_data['Time'].min()

        result = compute_prc(
            df=sample_spike_data,
            channel=sample_spike_data['channel'].iloc[0],
            train_timestamps=[start_time],
            train_duration_ms=10000, # 10 seconds to cover the entire sample data
            stim_frequency=10.0,
            num_bins=5
        )

        # Check
        assert 'channel' in result
        assert 'phase_bins' in result
        assert 'prc' in result
        assert 'train_0' in result
        assert len(result['phase_bins']) == 5
        assert np.array_equal(result['train_0'], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        assert np.array_equal(result['prc'], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

def test_compute_prc_stdp_mode_no_spikes():
    """Test compute_prc in STDP mode with no spikes in the time range"""

    df = pd.DataFrame({
        'Time': [pd.Timestamp('2020-01-01 12:00:10')],
        'channel': [1],
        'Amplitude': [-5.0]
    })

    result = compute_prc(
        df=df,
        channel=1,
        train_timestamps=[pd.Timestamp('2020-01-01 12:00:00')],
        train_duration_ms=1, # Very short duration
        stim_frequency=10.0,
        num_bins=5
    )

    # Check
    assert 'channel' in result
    assert 'phase_bins' in result
    assert 'error' in result
    assert 'no valid PRCs computed' in result['error']

def test_compute_prc_phase_encoding_mode(sample_spike_data):
    """Test compute_prc in phase-encoding mode."""

    sample_data_with_condition = sample_spike_data.copy()
    sample_data_with_condition['condition'] = 'test_condition'

    # Mock
    with patch('src.analysis.prc.calculate_prc_times') as mock_calculate:

        mock_calculate.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = compute_prc(
            df=sample_data_with_condition,
            channel=sample_data_with_condition['channel'].iloc[0],
            phase_frequencies={'test_condition': 10.0},
            num_bins=5
        )

        # Check
        assert 'channel' in result
        assert 'phase_bins' in result
        assert 'prc' in result
        assert 'prc_test_condition' in result
        assert len(result['phase_bins']) == 5
        assert np.array_equal(result['prc_test_condition'], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        assert np.array_equal(result['prc'], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

def test_compute_prc_phase_encoding_mode_no_spikes():
    """Test compute_prc in phase-encoding mode with no spikes for the condition"""

    df = pd.DataFrame({
        'Time': [pd.Timestamp('2020-01-01 12:00:00')],
        'channel': [1],
        'Amplitude': [-5.0],
        'condition': ['other_condition']
    })

    result = compute_prc(
        df=df,
        channel=1,
        phase_frequencies={'test_condition': 10.0},
        num_bins=5
    )

    # Check
    assert 'channel' in result
    assert 'phase_bins' in result
    assert 'error' in result

    # Check
    assert 'no valid PRCs computed for any condition' in result['error']

def test_compute_prc_phase_encoding_mode_not_enough_spikes():
    """Test compute_prc in phase-encoding mode with not enough spikes for PRC calculation"""

    df = pd.DataFrame({
        'Time': [pd.Timestamp('2020-01-01 12:00:00')],
        'channel': [1],
        'Amplitude': [-5.0],
        'condition': ['test_condition']
    })

    result = compute_prc(
        df=df,
        channel=1,
        phase_frequencies={'test_condition': 10.0},
        num_bins=5
    )

    # Check
    assert 'channel' in result
    assert 'phase_bins' in result
    assert 'error' in result
    assert 'no valid PRCs computed for any condition' in result['error']

def test_compute_prc_no_mode_specified():
    """Test compute_prc with no mode specified"""


    df = pd.DataFrame({
        'Time': [pd.Timestamp('2020-01-01 12:00:00')],
        'channel': [1],
        'Amplitude': [-5.0]
    })

    result = compute_prc(
        df=df,
        channel=1,
        num_bins=5
    )

    # Check
    assert 'channel' in result
    assert 'phase_bins' in result
    assert 'error' in result
    assert 'must supply either' in result['error']

def test_compute_prc_empty_data():
    """Test compute_prc with empty DataFrame"""

    empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])

    result = compute_prc(
        df=empty_df,
        channel=1,
        train_timestamps=[pd.Timestamp('2020-01-01 12:00:00')],
        train_duration_ms=100,
        stim_frequency=10.0,
        num_bins=5
    )

    # Check
    assert 'channel' in result
    assert 'phase_bins' in result
    assert 'error' in result
    assert 'no spikes for channel' in result['error']

def test_compute_prc_nonexistent_channel(sample_spike_data):
    """Test compute_prc with a channel that doesn't exist in the data"""

    max_channel = sample_spike_data['channel'].max()
    nonexistent_channel = max_channel + 1

    result = compute_prc(
        df=sample_spike_data,
        channel=nonexistent_channel,
        train_timestamps=[pd.Timestamp('2020-01-01 12:00:00')],
        train_duration_ms=100,
        stim_frequency=10.0,
        num_bins=5
    )

    # Check
    assert 'channel' in result
    assert 'phase_bins' in result
    assert 'error' in result
    assert result['channel'] == nonexistent_channel
    assert f'no spikes for channel {nonexistent_channel}' in result['error']

def test_compute_prc_time_conversion():
    """Test compute_prc with non-datetime Time column"""

    df = pd.DataFrame({
        'Time': ['2020-01-01 12:00:00'],
        'channel': [1],
        'Amplitude': [-5.0]
    })

    # Mock
    with patch('src.analysis.prc.calculate_prc_times') as mock_calculate:
        mock_calculate.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result = compute_prc(
            df=df,
            channel=1,
            train_timestamps=[pd.Timestamp('2020-01-01 12:00:00')], # Convert string to Timestamp
            train_duration_ms=100,
            stim_frequency=10.0,
            num_bins=5
        )

        # Check
        assert 'channel' in result
        assert 'phase_bins' in result
