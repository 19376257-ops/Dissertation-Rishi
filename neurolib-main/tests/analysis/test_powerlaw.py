import pytest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import patch, MagicMock

from src.analysis.powerlaw import compute_powerlaw

def test_compute_powerlaw_basic():
    """Test compute_powerlaw with basic valid data."""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = [start_time + datetime.timedelta(seconds=i) for i in range(20)]
    channels = [1] * 20
    amplitudes = np.random.rand(20) * 10

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.powerlaw.powerlaw.Fit') as mock_fit:
        mock_fit_instance = MagicMock()
        mock_fit_instance.alpha = 2.5
        mock_fit_instance.xmin = 0.1
        mock_fit_instance.D = 0.05
        mock_fit.return_value = mock_fit_instance

        result = compute_powerlaw(
            df=df,
            channel=1
        )

        # Check
        assert 'channel' in result
        assert 'fit' in result
        assert 'alpha' in result
        assert 'xmin' in result
        assert 'D' in result
        assert result['channel'] == 1
        assert result['alpha'] == 2.5
        assert result['xmin'] == 0.1
        assert result['D'] == 0.05

def test_compute_powerlaw_not_enough_data():
    """Test compute_powerlaw with not enough data points."""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = [start_time + datetime.timedelta(seconds=i) for i in range(5)]
    channels = [1] * 5
    amplitudes = np.random.rand(5) * 10

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    result = compute_powerlaw(
        df=df,
        channel=1
    )

    # Check
    assert 'channel' in result
    assert 'error' in result
    assert "Not enough data points" in result['error']
    assert 'fit' not in result
    assert 'alpha' not in result
    assert 'xmin' not in result
    assert 'D' not in result

def test_compute_powerlaw_exception():
    """Test compute_powerlaw with an exception during fitting"""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = [start_time + datetime.timedelta(seconds=i) for i in range(20)]
    channels = [1] * 20
    amplitudes = np.random.rand(20) * 10

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.powerlaw.powerlaw.Fit') as mock_fit:
        mock_fit.side_effect = ValueError("Test exception")

        result = compute_powerlaw(
            df=df,
            channel=1
        )

        # Check
        assert 'channel' in result
        assert 'error' in result
        assert "Test exception" in result['error']
        assert 'fit' not in result
        assert 'alpha' not in result
        assert 'xmin' not in result
        assert 'D' not in result

def test_compute_powerlaw_no_channel():
    """Test compute_powerlaw without specifying a channel."""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = [start_time + datetime.timedelta(seconds=i) for i in range(20)]
    channels = [1] * 20
    amplitudes = np.random.rand(20) * 10

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.powerlaw.powerlaw.Fit') as mock_fit:
        mock_fit_instance = MagicMock()
        mock_fit_instance.alpha = 2.5
        mock_fit_instance.xmin = 0.1
        mock_fit_instance.D = 0.05
        mock_fit.return_value = mock_fit_instance

        result = compute_powerlaw(
            df=df
        )

        # Check
        assert 'fit' in result
        assert 'alpha' in result
        assert 'xmin' in result
        assert 'D' in result
        assert 'channel' not in result
        assert result['alpha'] == 2.5
        assert result['xmin'] == 0.1
        assert result['D'] == 0.05

def test_compute_powerlaw_time_conversion():
    """Test compute_powerlaw with non-datetime Time column"""

    times = [f"2020-01-01 12:00:{i:02d}" for i in range(20)]
    channels = [1] * 20
    amplitudes = np.random.rand(20) * 10

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    df['Time'] = pd.to_datetime(df['Time'])

    # Mock
    with patch('src.analysis.powerlaw.powerlaw.Fit') as mock_fit:
        mock_fit_instance = MagicMock()
        mock_fit_instance.alpha = 2.5
        mock_fit_instance.xmin = 0.1
        mock_fit_instance.D = 0.05
        mock_fit.return_value = mock_fit_instance

        result = compute_powerlaw(
            df=df,
            channel=1
        )

        # Check
        assert 'channel' in result
        assert 'fit' in result
        assert 'alpha' in result
        assert 'xmin' in result
        assert 'D' in result
        assert result['channel'] == 1
        assert result['alpha'] == 2.5
        assert result['xmin'] == 0.1
        assert result['D'] == 0.05
