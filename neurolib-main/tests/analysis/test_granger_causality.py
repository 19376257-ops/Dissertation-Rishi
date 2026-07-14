import pytest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import patch, MagicMock

from src.analysis.granger_causality import compute_granger_causality

def test_compute_granger_causality_basic():
    """Test compute_granger_causality with basic valid data."""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = [start_time + datetime.timedelta(seconds=i) for i in range(20)]
    channels = [1, 2] * 10
    amplitudes = np.random.rand(20) * 10

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.granger_causality.grangercausalitytests') as mock_granger:
        mock_result = {}
        for lag in range(1, 6):
            mock_result[lag] = [{
                'ssr_chi2test': [0, 0.04 if lag == 2 else 0.1] # Make lag 2 significant
            }]
        mock_granger.return_value = mock_result

        result = compute_granger_causality(
            df=df,
            pre_channel=1,
            post_channel=2,
            resample_interval='1s',
            max_lag=5
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        assert 'p_values' in result
        assert 'significant_lags' in result
        assert 'resampled_data' in result
        assert result['pre_channel'] == 1
        assert result['post_channel'] == 2
        assert len(result['p_values']) == 5
        for lag in range(1, 6):
            assert lag in result['p_values']
        assert 2 in result['significant_lags']
        assert len(result['significant_lags']) == 1

def test_compute_granger_causality_missing_channels():
    """Test compute_granger_causality with missing channels"""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = [start_time + datetime.timedelta(seconds=i) for i in range(10)]
    channels = [1] * 10
    amplitudes = np.random.rand(10) * 10

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    result = compute_granger_causality(
        df=df,
        pre_channel=1,
        post_channel=2,
        resample_interval='1s',
        max_lag=5
    )

    # Check
    assert 'pre_channel' in result
    assert 'post_channel' in result
    assert 'error' in result
    assert 'p_values' in result
    assert 'significant_lags' in result
    assert 'resampled_data' in result
    assert "post_channel 2" in result['error']
    assert result['p_values'] == {}
    assert result['significant_lags'] == []

def test_compute_granger_causality_not_enough_data():
    """Test compute_granger_causality with not enough data points."""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = [start_time + datetime.timedelta(seconds=i) for i in range(4)]
    channels = [1, 2] * 2
    amplitudes = np.random.rand(4) * 10

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    result = compute_granger_causality(
        df=df,
        pre_channel=1,
        post_channel=2,
        resample_interval='1s',
        max_lag=5  # This is > than the # of data points
    )

    # Check
    assert 'pre_channel' in result
    assert 'post_channel' in result
    assert 'error' in result
    assert 'p_values' in result
    assert 'significant_lags' in result
    assert 'resampled_data' in result
    assert "Not enough data points" in result['error']
    assert result['p_values'] == {}
    assert result['significant_lags'] == []

def test_compute_granger_causality_exception():
    """Test compute_granger_causality with an exception during computation."""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = [start_time + datetime.timedelta(seconds=i) for i in range(20)]
    channels = [1, 2] * 10
    amplitudes = np.random.rand(20) * 10

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.granger_causality.grangercausalitytests') as mock_granger:
        mock_granger.side_effect = ValueError("Test exception")

        result = compute_granger_causality(
            df=df,
            pre_channel=1,
            post_channel=2,
            resample_interval='1s',
            max_lag=5
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        assert 'error' in result
        assert 'p_values' in result
        assert 'significant_lags' in result
        assert 'resampled_data' in result
        assert "Test exception" in result['error']
        assert result['p_values'] == {}
        assert result['significant_lags'] == []

def test_compute_granger_causality_sample_mode():
    """Test compute_granger_causality with sample mode enabled."""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = [start_time + datetime.timedelta(seconds=i) for i in range(300)]
    channels = [1, 2] * 150
    amplitudes = np.random.rand(300) * 10

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.granger_causality.grangercausalitytests') as mock_granger:
        mock_result = {}
        for lag in range(1, 6):
            mock_result[lag] = [{
                'ssr_chi2test': [0, 0.1]
            }]
        mock_granger.return_value = mock_result

        result = compute_granger_causality(
            df=df,
            pre_channel=1,
            post_channel=2,
            resample_interval='1s',
            max_lag=5,
            sample_mode=True
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        assert 'p_values' in result
        assert 'significant_lags' in result
        assert 'resampled_data' in result
        assert result['pre_channel'] == 1
        assert result['post_channel'] == 2
        assert len(result['p_values']) == 5

def test_compute_granger_causality_time_conversion():
    """Test compute_granger_causality with non-datetime Time column."""

    times = [f"2020-01-01 12:00:{i:02d}" for i in range(20)]
    channels = [1, 2] * 10
    amplitudes = np.random.rand(20) * 10

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.granger_causality.grangercausalitytests') as mock_granger:
        mock_result = {}
        for lag in range(1, 6):
            mock_result[lag] = [{
                'ssr_chi2test': [0, 0.1]
            }]
        mock_granger.return_value = mock_result

        result = compute_granger_causality(
            df=df,
            pre_channel=1,
            post_channel=2,
            resample_interval='1s',
            max_lag=5
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        assert 'p_values' in result
        assert 'significant_lags' in result
        assert 'resampled_data' in result
        assert result['pre_channel'] == 1
        assert result['post_channel'] == 2
