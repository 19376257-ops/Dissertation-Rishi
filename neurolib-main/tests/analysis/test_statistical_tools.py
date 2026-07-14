import pytest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import patch, MagicMock

from src.analysis.statistical_tools import compute_statistical_tests

def test_compute_statistical_tests_basic():
    """Test compute_statistical_tests with basic valid data"""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = []
    channels = []
    amplitudes = []

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks, \
         patch('src.analysis.statistical_tools.pearsonr') as mock_pearson, \
         patch('src.analysis.statistical_tools.shapiro') as mock_shapiro, \
         patch('src.analysis.statistical_tools.ttest_ind') as mock_ttest, \
         patch('src.analysis.statistical_tools.mannwhitneyu') as mock_mw:

        mock_ks.return_value = (0.3, 0.7) # statistic, p-value
        mock_pearson.return_value = (0.5, 0.6) # correlation, p-value
        mock_shapiro.return_value = (0.9, 0.8) # statistic, p-value
        mock_ttest.return_value = (0.2, 0.9) # statistic, p-value
        mock_mw.return_value = (150, 0.4) # statistic, p-value

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        assert 'ks_test' in result
        assert 'pearson' in result
        assert 'shapiro' in result
        assert 't_test' in result
        assert 'mann_whitney' in result
        assert result['pre_channel'] == 1
        assert result['post_channel'] == 2
        assert result['ks_test']['statistic'] == 0.3
        assert result['ks_test']['p_value'] == 0.7
        if 'error' in result['pearson']:
            assert "Cannot compute correlation" in result['pearson']['error']
        else:
            assert result['pearson']['correlation'] == 0.5 or np.isnan(result['pearson']['correlation'])
            assert result['pearson']['p_value'] == 0.6 or np.isnan(result['pearson']['p_value'])
        assert result['shapiro']['pre_channel']['statistic'] == 0.9
        assert result['shapiro']['pre_channel']['p_value'] == 0.8
        assert result['shapiro']['post_channel']['statistic'] == 0.9
        assert result['shapiro']['post_channel']['p_value'] == 0.8
        assert result['t_test']['statistic'] == 0.2
        assert result['t_test']['p_value'] == 0.9
        assert result['mann_whitney']['statistic'] == 150
        assert result['mann_whitney']['p_value'] == 0.4

def test_compute_statistical_tests_not_enough_data():
    """Test compute_statistical_tests with not enough data points"""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = []
    channels = []
    amplitudes = []

    times.append(start_time)
    channels.append(1)
    amplitudes.append(-np.random.rand() * 10)

    times.append(start_time)
    channels.append(2)
    amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    result = compute_statistical_tests(
        df=df,
        pre_channel=1,
        post_channel=2
    )

    # Check
    assert 'pre_channel' in result
    assert 'post_channel' in result
    assert 'error' in result
    assert "Not enough data points" in result['error']

def test_compute_statistical_tests_with_condition():
    """Test compute_statistical_tests with a specified condition."""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = []
    channels = []
    amplitudes = []
    conditions = []

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)
        conditions.append('test_condition' if i < 10 else 'other_condition')

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)
        conditions.append('test_condition' if i < 10 else 'other_condition')

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes,
        'condition': conditions
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks, \
         patch('src.analysis.statistical_tools.pearsonr') as mock_pearson, \
         patch('src.analysis.statistical_tools.shapiro') as mock_shapiro, \
         patch('src.analysis.statistical_tools.ttest_ind') as mock_ttest, \
         patch('src.analysis.statistical_tools.mannwhitneyu') as mock_mw:

        mock_ks.return_value = (0.3, 0.7)
        mock_pearson.return_value = (0.5, 0.6)
        mock_shapiro.return_value = (0.9, 0.8)
        mock_ttest.return_value = (0.2, 0.9)
        mock_mw.return_value = (150, 0.4)

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2,
            condition='test_condition'
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        assert 'condition' in result
        assert 'ks_test' in result
        assert 'pearson' in result
        assert 'shapiro' in result
        assert 't_test' in result
        assert 'mann_whitney' in result
        assert result['condition'] == 'test_condition'

def test_compute_statistical_tests_sample_mode():
    """Test compute_statistical_tests with sample mode enabled"""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = []
    channels = []
    amplitudes = []

    for i in range(1500):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)

    for i in range(1500):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks, \
         patch('src.analysis.statistical_tools.pearsonr') as mock_pearson, \
         patch('src.analysis.statistical_tools.shapiro') as mock_shapiro, \
         patch('src.analysis.statistical_tools.ttest_ind') as mock_ttest, \
         patch('src.analysis.statistical_tools.mannwhitneyu') as mock_mw:

        mock_ks.return_value = (0.3, 0.7)
        mock_pearson.return_value = (0.5, 0.6)
        mock_shapiro.return_value = (0.9, 0.8)
        mock_ttest.return_value = (0.2, 0.9)
        mock_mw.return_value = (150, 0.4)

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2,
            sample_mode=True
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        assert 'ks_test' in result
        assert 'pearson' in result
        assert 'shapiro' in result
        assert 't_test' in result
        assert 'mann_whitney' in result

def test_compute_statistical_tests_ks_test_exception():
    """Test compute_statistical_tests with an exception in the K-S test"""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = []
    channels = []
    amplitudes = []

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks, \
         patch('src.analysis.statistical_tools.pearsonr') as mock_pearson, \
         patch('src.analysis.statistical_tools.shapiro') as mock_shapiro, \
         patch('src.analysis.statistical_tools.ttest_ind') as mock_ttest, \
         patch('src.analysis.statistical_tools.mannwhitneyu') as mock_mw:

        mock_ks.side_effect = ValueError("Test exception")
        mock_pearson.return_value = (0.5, 0.6)
        mock_shapiro.return_value = (0.9, 0.8)
        mock_ttest.return_value = (0.2, 0.9)
        mock_mw.return_value = (150, 0.4)

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        assert 'ks_test' in result
        assert 'error' in result['ks_test']
        assert "Test exception" in result['ks_test']['error']

def test_compute_statistical_tests_pearson_constant_arrays():
    """Test compute_statistical_tests with constant arrays for Pearson correlation"""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = []
    channels = []
    amplitudes = []

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks, \
         patch('src.analysis.statistical_tools.pearsonr') as mock_pearson, \
         patch('src.analysis.statistical_tools.shapiro') as mock_shapiro, \
         patch('src.analysis.statistical_tools.ttest_ind') as mock_ttest, \
         patch('src.analysis.statistical_tools.mannwhitneyu') as mock_mw, \
         patch('src.analysis.statistical_tools.np.unique') as mock_unique:

        mock_ks.return_value = (0.3, 0.7)
        mock_pearson.return_value = (0.5, 0.6)
        mock_shapiro.return_value = (0.9, 0.8)
        mock_ttest.return_value = (0.2, 0.9)
        mock_mw.return_value = (150, 0.4)
        mock_unique.side_effect = lambda x: np.array([1]) if len(x) > 0 and x[0] == 1 else np.array([1, 2, 3])

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        assert 'pearson' in result
        assert 'error' in result['pearson']
        assert "Cannot compute correlation" in result['pearson']['error']

def test_compute_statistical_tests_shapiro_downsampling():
    """Test compute_statistical_tests with downsampling for Shapiro-Wilk test"""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = []
    channels = []
    amplitudes = []

    for i in range(6000):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)

    for i in range(6000):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks, \
         patch('src.analysis.statistical_tools.pearsonr') as mock_pearson, \
         patch('src.analysis.statistical_tools.shapiro') as mock_shapiro, \
         patch('src.analysis.statistical_tools.ttest_ind') as mock_ttest, \
         patch('src.analysis.statistical_tools.mannwhitneyu') as mock_mw:

        mock_ks.return_value = (0.3, 0.7)
        mock_pearson.return_value = (0.5, 0.6)
        mock_shapiro.return_value = (0.9, 0.8)
        mock_ttest.return_value = (0.2, 0.9)
        mock_mw.return_value = (150, 0.4)

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        assert 'shapiro' in result
        assert 'pre_channel' in result['shapiro']
        assert 'post_channel' in result['shapiro']
        assert result['shapiro']['pre_channel']['downsampled'] == True
        assert result['shapiro']['post_channel']['downsampled'] == True

def test_compute_statistical_tests_non_datetime():
    """Test compute_statistical_tests with non-datetime Time"""
    times = []
    channels = []
    amplitudes = []

    for i in range(20):
        times.append(float(i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)

    # Create data for channel 2
    for i in range(20):
        times.append(float(i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks, \
         patch('src.analysis.statistical_tools.pearsonr') as mock_pearson, \
         patch('src.analysis.statistical_tools.shapiro') as mock_shapiro, \
         patch('src.analysis.statistical_tools.ttest_ind') as mock_ttest, \
         patch('src.analysis.statistical_tools.mannwhitneyu') as mock_mw:

        mock_ks.return_value = (0.3, 0.7)
        mock_pearson.return_value = (0.5, 0.6)
        mock_shapiro.return_value = (0.9, 0.8)
        mock_ttest.return_value = (0.2, 0.9)
        mock_mw.return_value = (150, 0.4)

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        assert 'ks_test' in result
        assert 'pearson' in result
        assert 'shapiro' in result
        assert 't_test' in result
        assert 'mann_whitney' in result
        assert result['ks_test']['statistic'] == 0.3
        assert result['ks_test']['p_value'] == 0.7

def test_compute_statistical_tests_minimum_samples():
    """Test compute_statistical_tests with minimum sample sizes for each test"""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = []
    channels = []
    amplitudes = []

    for i in range(3):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)

    for i in range(3):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    result = compute_statistical_tests(
        df=df,
        pre_channel=1,
        post_channel=2
    )

    # Check
    assert 'pre_channel' in result
    assert 'post_channel' in result
    assert 'ks_test' in result
    assert 'pearson' in result
    assert 'shapiro' in result
    assert 't_test' in result
    assert 'mann_whitney' in result
    if 'small_sample' in result['shapiro']['pre_channel']:
        assert isinstance(result['shapiro']['pre_channel']['small_sample'], bool)
        assert isinstance(result['shapiro']['post_channel']['small_sample'], bool)

def test_compute_statistical_tests_ks_test_minimum_samples():
    """Test compute_statistical_tests with minimum sample sizes for KS test."""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = [start_time + datetime.timedelta(seconds=i) for i in range(3)]
    channels = [1] * 3
    amplitudes = [-np.random.rand() * 10 for _ in range(3)]
    times.extend([start_time, start_time + datetime.timedelta(seconds=1)])
    channels.extend([2, 2])
    amplitudes.extend([-np.random.rand() * 10, -np.random.rand() * 10])

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks:
        mock_ks.return_value = (0.3, 0.7)

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        if 'error' in result:
            assert "Not enough data points" in result['error']
        else:
            assert 'ks_test' in result
            assert 'pearson' in result
            assert 'shapiro' in result
            assert 't_test' in result
            assert 'mann_whitney' in result

def test_compute_statistical_tests_pearson_minimum_samples():
    """Test compute_statistical_tests with minimum sample sizes for Pearson correlation."""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = []
    channels = []
    amplitudes = []

    for i in range(2):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)

    for i in range(2):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks, \
         patch('src.analysis.statistical_tools.pearsonr') as mock_pearson, \
         patch('src.analysis.statistical_tools.shapiro') as mock_shapiro, \
         patch('src.analysis.statistical_tools.ttest_ind') as mock_ttest, \
         patch('src.analysis.statistical_tools.mannwhitneyu') as mock_mw:

        mock_ks.return_value = (0.3, 0.7)
        mock_pearson.return_value = (0.5, 0.6)
        mock_shapiro.return_value = (0.9, 0.8)
        mock_ttest.return_value = (0.2, 0.9)
        mock_mw.return_value = (150, 0.4)

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        if 'error' in result:
            assert "Not enough data points" in result['error']
        else:
            assert 'ks_test' in result
            assert 'pearson' in result
            assert 'shapiro' in result
            assert 't_test' in result
            assert 'mann_whitney' in result

def test_compute_statistical_tests_t_test_minimum_samples():
    """Test compute_statistical_tests with minimum sample sizes for t-test"""
    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)

    times = []
    channels = []
    amplitudes = []

    for i in range(2):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)

    for i in range(2):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks, \
         patch('src.analysis.statistical_tools.pearsonr') as mock_pearson, \
         patch('src.analysis.statistical_tools.shapiro') as mock_shapiro, \
         patch('src.analysis.statistical_tools.ttest_ind') as mock_ttest, \
         patch('src.analysis.statistical_tools.mannwhitneyu') as mock_mw:

        mock_ks.return_value = (0.3, 0.7)
        mock_pearson.return_value = (0.5, 0.6)
        mock_shapiro.return_value = (0.9, 0.8)
        mock_ttest.return_value = (0.2, 0.9)
        mock_mw.return_value = (150, 0.4)

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result

        if 'error' in result:
            assert "Not enough data points" in result['error']
        else:
            assert 'ks_test' in result
            assert 'pearson' in result
            assert 'shapiro' in result
            assert 't_test' in result
            assert 'mann_whitney' in result

def test_compute_statistical_tests_mann_whitney_minimum_samples():
    """Test compute_statistical_tests with minimum sample sizes for Mann-Whitney U test"""
    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)

    times = []
    channels = []
    amplitudes = []

    for i in range(2):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)

    for i in range(2):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks, \
         patch('src.analysis.statistical_tools.pearsonr') as mock_pearson, \
         patch('src.analysis.statistical_tools.shapiro') as mock_shapiro, \
         patch('src.analysis.statistical_tools.ttest_ind') as mock_ttest, \
         patch('src.analysis.statistical_tools.mannwhitneyu') as mock_mw:

        mock_ks.return_value = (0.3, 0.7)
        mock_pearson.return_value = (0.5, 0.6)
        mock_shapiro.return_value = (0.9, 0.8)
        mock_ttest.return_value = (0.2, 0.9)
        mock_mw.return_value = (150, 0.4)

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        if 'error' in result:
            assert "Not enough data points" in result['error']
        else:
            assert 'ks_test' in result
            assert 'pearson' in result
            assert 'shapiro' in result
            assert 't_test' in result
            assert 'mann_whitney' in result

def test_compute_statistical_tests_pearson_exception():
    """Test compute_statistical_tests with an exception in the Pearson correlation"""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = []
    channels = []
    amplitudes = []

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks, \
         patch('src.analysis.statistical_tools.pearsonr') as mock_pearson, \
         patch('src.analysis.statistical_tools.shapiro') as mock_shapiro, \
         patch('src.analysis.statistical_tools.ttest_ind') as mock_ttest, \
         patch('src.analysis.statistical_tools.mannwhitneyu') as mock_mw:

        mock_pearson.side_effect = ValueError("Cannot compute correlation: one or both input arrays are constant.")
        mock_ks.return_value = (0.3, 0.7)
        mock_shapiro.return_value = (0.9, 0.8)
        mock_ttest.return_value = (0.2, 0.9)
        mock_mw.return_value = (150, 0.4)

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        if 'error' in result:
            assert "Not enough data points" in result['error']
        else:
            assert 'pearson' in result
            assert 'error' in result['pearson']
            assert "Cannot compute correlation" in result['pearson']['error']


def test_compute_statistical_tests_shapiro_exception():
    """Test compute_statistical_tests with an exception in the Shapiro-Wilk test"""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = []
    channels = []
    amplitudes = []

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks, \
         patch('src.analysis.statistical_tools.pearsonr') as mock_pearson, \
         patch('src.analysis.statistical_tools.shapiro') as mock_shapiro, \
         patch('src.analysis.statistical_tools.ttest_ind') as mock_ttest, \
         patch('src.analysis.statistical_tools.mannwhitneyu') as mock_mw:

        mock_shapiro.side_effect = ValueError("Test shapiro exception")
        mock_ks.return_value = (0.3, 0.7)
        mock_pearson.return_value = (0.5, 0.6)
        mock_ttest.return_value = (0.2, 0.9)
        mock_mw.return_value = (150, 0.4)

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        assert 'shapiro' in result
        assert 'error' in result['shapiro']
        assert "Test shapiro exception" in result['shapiro']['error']

def test_compute_statistical_tests_ttest_exception():
    """Test compute_statistical_tests with an exception in the t-test"""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = []
    channels = []
    amplitudes = []

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks, \
         patch('src.analysis.statistical_tools.pearsonr') as mock_pearson, \
         patch('src.analysis.statistical_tools.shapiro') as mock_shapiro, \
         patch('src.analysis.statistical_tools.ttest_ind') as mock_ttest, \
         patch('src.analysis.statistical_tools.mannwhitneyu') as mock_mw:

        mock_ttest.side_effect = ValueError("Test ttest exception")
        mock_ks.return_value = (0.3, 0.7)
        mock_pearson.return_value = (0.5, 0.6)
        mock_shapiro.return_value = (0.9, 0.8)
        mock_mw.return_value = (150, 0.4)

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        assert 't_test' in result
        assert 'error' in result['t_test']
        assert "Test ttest exception" in result['t_test']['error']

def test_compute_statistical_tests_mannwhitney_exception():
    """Test compute_statistical_tests with an exception in the Mann-Whitney U test"""

    start_time = datetime.datetime(2020, 1, 1, 12, 0, 0)
    times = []
    channels = []
    amplitudes = []

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(1)
        amplitudes.append(-np.random.rand() * 10)

    for i in range(20):
        times.append(start_time + datetime.timedelta(seconds=i))
        channels.append(2)
        amplitudes.append(-np.random.rand() * 10)

    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })

    # Mock
    with patch('src.analysis.statistical_tools.ks_2samp') as mock_ks, \
         patch('src.analysis.statistical_tools.pearsonr') as mock_pearson, \
         patch('src.analysis.statistical_tools.shapiro') as mock_shapiro, \
         patch('src.analysis.statistical_tools.ttest_ind') as mock_ttest, \
         patch('src.analysis.statistical_tools.mannwhitneyu') as mock_mw:

        mock_mw.side_effect = ValueError("Test mannwhitney exception")
        mock_ks.return_value = (0.3, 0.7)
        mock_pearson.return_value = (0.5, 0.6)
        mock_shapiro.return_value = (0.9, 0.8)
        mock_ttest.return_value = (0.2, 0.9)

        result = compute_statistical_tests(
            df=df,
            pre_channel=1,
            post_channel=2
        )

        # Check
        assert 'pre_channel' in result
        assert 'post_channel' in result
        assert 'mann_whitney' in result
        assert 'error' in result['mann_whitney']
        assert "Test mannwhitney exception" in result['mann_whitney']['error']
