import pytest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import patch, MagicMock

from src.analysis.utils import (
    map_channel_to_mea,
    map_channels_to_mea,
    assign_conditions_to_bins,
    group_conditions_by_pre_post,
    mask_stim_windows,
    compute_stim_artifact_template,
    apply_template_subtraction
)

def test_map_channel_to_mea_valid_range():
    """Test map_channel_to_mea with valid channel numbers in the 1-32 range."""

    result = map_channel_to_mea(1)
    assert result == 97 # 96 + 1

    result = map_channel_to_mea(32)
    assert result == 128 # 96 + 32

def test_map_channel_to_mea_already_mapped():
    """Test map_channel_to_mea with channel numbers already in the 1-128 range."""

    result = map_channel_to_mea(97)
    assert result == 97

    result = map_channel_to_mea(128)
    assert result == 128

def test_map_channel_to_mea_invalid_range():
    """Test map_channel_to_mea with invalid channel numbers."""

    with pytest.raises(ValueError, match="Channel 0 is out of range"):
        map_channel_to_mea(0)

    with pytest.raises(ValueError, match="Channel 129 is out of range"):
        map_channel_to_mea(129)

def test_map_channels_to_mea_valid():
    """Test map_channels_to_mea with valid channel numbers."""

    result = map_channels_to_mea([1, 2, 3])
    assert result == [97, 98, 99] # 96 + each channel

def test_map_channels_to_mea_empty():
    """Test map_channels_to_mea with an empty list."""

    result = map_channels_to_mea([])
    assert result == []

def test_assign_conditions_to_bins_basic():
    """Test assign_conditions_to_bins with basic valid data."""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00'),
            pd.Timestamp('2023-01-01 12:00:01'),
            pd.Timestamp('2023-01-01 12:00:02'),
            pd.Timestamp('2023-01-01 12:00:03')
        ],
        'channel': [1, 1, 1, 1],
        'Amplitude': [-5.0, -6.0, -7.0, -8.0],
        'condition': ['baseline', 'baseline', 'stim', 'stim']
    })

    start_time = pd.Timestamp('2023-01-01 12:00:00')
    bin_edges = [
        pd.Timestamp('2023-01-01 12:00:00'),
        pd.Timestamp('2023-01-01 12:00:01'),
        pd.Timestamp('2023-01-01 12:00:02'),
        pd.Timestamp('2023-01-01 12:00:03'),
        pd.Timestamp('2023-01-01 12:00:04')
    ]

    # Call
    result = assign_conditions_to_bins(df, start_time, bin_edges)

    # Check
    assert isinstance(result, np.ndarray)
    assert len(result) == 4  # 4 bins
    assert result[0] == 'baseline'
    assert result[1] == 'baseline'
    assert result[2] == 'stim'
    assert result[3] == 'stim'

def test_assign_conditions_to_bins_milliseconds():
    """Test assign_conditions_to_bins with bin edges in milliseconds."""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00'),
            pd.Timestamp('2023-01-01 12:00:01'),
            pd.Timestamp('2023-01-01 12:00:02'),
            pd.Timestamp('2023-01-01 12:00:03')
        ],
        'channel': [1, 1, 1, 1],
        'Amplitude': [-5.0, -6.0, -7.0, -8.0],
        'condition': ['baseline', 'baseline', 'stim', 'stim']
    })

    start_time = pd.Timestamp('2023-01-01 12:00:00')
    bin_edges = [0, 1000, 2000, 3000, 4000] # milliseconds

    # Call
    result = assign_conditions_to_bins(df, start_time, bin_edges)

    # Check
    assert isinstance(result, np.ndarray)
    assert len(result) == 4  # 4 bins
    assert result[0] == 'baseline'
    assert result[1] == 'baseline'
    assert result[2] == 'stim'
    assert result[3] == 'stim'

def test_assign_conditions_to_bins_empty_bin():
    """Test assign_conditions_to_bins with an empty bin"""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00'),
            pd.Timestamp('2023-01-01 12:00:02'),
            pd.Timestamp('2023-01-01 12:00:03')
        ],
        'channel': [1, 1, 1],
        'Amplitude': [-5.0, -7.0, -8.0],
        'condition': ['baseline', 'stim', 'stim']
    })

    start_time = pd.Timestamp('2023-01-01 12:00:00')
    bin_edges = [
        pd.Timestamp('2023-01-01 12:00:00'),
        pd.Timestamp('2023-01-01 12:00:01'),
        pd.Timestamp('2023-01-01 12:00:02'),
        pd.Timestamp('2023-01-01 12:00:03'),
        pd.Timestamp('2023-01-01 12:00:04')
    ]

    # Call
    result = assign_conditions_to_bins(df, start_time, bin_edges)

    assert isinstance(result, np.ndarray)
    assert len(result) == 4
    assert result[0] == 'baseline'
    assert result[1] == 'unknown'
    assert result[2] == 'stim'
    assert result[3] == 'stim'

def test_group_conditions_by_pre_post():
    """Test group_conditions_by_pre_post with various conditions"""

    conditions = np.array([
        'train_1_pre', 'train_1_post', 'train_2_pre', 'train_2_post',
        'no_stim', 'unknown', 'other_condition'
    ])

    # Call
    result = group_conditions_by_pre_post(conditions)

    # Check
    assert isinstance(result, np.ndarray)
    assert len(result) == len(conditions)
    assert result[0] == 'pre'
    assert result[1] == 'post'
    assert result[2] == 'pre'
    assert result[3] == 'post'
    assert result[4] == 'no_stim'
    assert result[5] == 'no_stim'
    assert result[6] == 'no_stim'

def test_mask_stim_windows_basic():
    """Test mask_stim_windows with basic valid data"""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00'),
            pd.Timestamp('2023-01-01 12:00:01'),
            pd.Timestamp('2023-01-01 12:00:02'),
            pd.Timestamp('2023-01-01 12:00:03')
        ],
        'channel': [1, 1, 1, 1],
        'Amplitude': [-5.0, -6.0, -7.0, -8.0]
    })

    pulse_times = [pd.Timestamp('2023-01-01 12:00:01')]
    result = mask_stim_windows(df, pulse_times, window_ms=1000)

    # Check
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1 # 1 spike should remain (at 12:00:03)
    assert pd.Timestamp('2023-01-01 12:00:03') in result['Time'].values
    assert pd.Timestamp('2023-01-01 12:00:00') not in result['Time'].values
    assert pd.Timestamp('2023-01-01 12:00:01') not in result['Time'].values
    assert pd.Timestamp('2023-01-01 12:00:02') not in result['Time'].values

def test_mask_stim_windows_string_timestamps():
    """Test mask_stim_windows with string timestamps"""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00'),
            pd.Timestamp('2023-01-01 12:00:01'),
            pd.Timestamp('2023-01-01 12:00:02'),
            pd.Timestamp('2023-01-01 12:00:03')
        ],
        'channel': [1, 1, 1, 1],
        'Amplitude': [-5.0, -6.0, -7.0, -8.0]
    })

    pulse_times = ['2023-01-01 12:00:01']
    result = mask_stim_windows(df, pulse_times, window_ms=1000)  # 1 second window

    # Check
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1 # 1 spike should remain (at 12:00:03)
    assert pd.Timestamp('2023-01-01 12:00:03') in result['Time'].values
    assert pd.Timestamp('2023-01-01 12:00:00') not in result['Time'].values
    assert pd.Timestamp('2023-01-01 12:00:01') not in result['Time'].values
    assert pd.Timestamp('2023-01-01 12:00:02') not in result['Time'].values

def test_mask_stim_windows_empty_data():
    """Test mask_stim_windows with empty DataFrame."""

    empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])
    pulse_times = [pd.Timestamp('2023-01-01 12:00:01')]

    # Call
    result = mask_stim_windows(empty_df, pulse_times, window_ms=1000)

    # Check
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_mask_stim_windows_no_pulse_times():
    """Test mask_stim_windows with no pulse times"""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00'),
            pd.Timestamp('2023-01-01 12:00:01')
        ],
        'channel': [1, 1],
        'Amplitude': [-5.0, -6.0]
    })

    # Call
    result = mask_stim_windows(df, [], window_ms=1000)

    # Check
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert pd.Timestamp('2023-01-01 12:00:00') in result['Time'].values
    assert pd.Timestamp('2023-01-01 12:00:01') in result['Time'].values

def test_compute_stim_artifact_template_basic():
    """Test compute_stim_artifact_template with basic valid data"""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00.000'),
            pd.Timestamp('2023-01-01 12:00:00.001'),
            pd.Timestamp('2023-01-01 12:00:00.002'),
            pd.Timestamp('2023-01-01 12:00:01.000'),
            pd.Timestamp('2023-01-01 12:00:01.001'),
            pd.Timestamp('2023-01-01 12:00:01.002')
        ],
        'channel': [1, 1, 1, 1, 1, 1],
        'Amplitude': [-5.0, -6.0, -7.0, -5.0, -6.0, -7.0]
    })

    pulse_times = [
        pd.Timestamp('2023-01-01 12:00:00'),
        pd.Timestamp('2023-01-01 12:00:01')
    ]

    # Call
    result = compute_stim_artifact_template(df, pulse_times, window_ms=5.0, channels=[1])

    # Check
    assert isinstance(result, dict)
    assert 1 in result # Channel 1 should be in the result
    assert isinstance(result[1], pd.DataFrame)
    assert 'relative_time_ms' in result[1].columns
    assert 'Amplitude' in result[1].columns
    assert len(result[1]) == 6

def test_compute_stim_artifact_template_string_timestamps():
    """Test compute_stim_artifact_template with string timestamps"""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00.000'),
            pd.Timestamp('2023-01-01 12:00:00.001'),
            pd.Timestamp('2023-01-01 12:00:01.000'),
            pd.Timestamp('2023-01-01 12:00:01.001')
        ],
        'channel': [1, 1, 1, 1],
        'Amplitude': [-5.0, -6.0, -5.0, -6.0]
    })

    pulse_times = [
        '2023-01-01 12:00:00',
        '2023-01-01 12:00:01'
    ]

    # Call
    result = compute_stim_artifact_template(df, pulse_times, window_ms=5.0)

    # Check
    assert isinstance(result, dict)
    assert 1 in result # Channel 1 should be in the result
    assert isinstance(result[1], pd.DataFrame)
    assert 'relative_time_ms' in result[1].columns
    assert len(result[1]) == 4

def test_compute_stim_artifact_template_empty_data():
    """Test compute_stim_artifact_template with empty DataFrame"""

    empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])
    pulse_times = [pd.Timestamp('2023-01-01 12:00:00')]
    result = compute_stim_artifact_template(empty_df, pulse_times, window_ms=5.0)

    # Check
    assert isinstance(result, dict)
    assert len(result) == 0

def test_compute_stim_artifact_template_no_pulse_times():
    """Test compute_stim_artifact_template with no pulse times"""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00'),
            pd.Timestamp('2023-01-01 12:00:01')
        ],
        'channel': [1, 1],
        'Amplitude': [-5.0, -6.0]
    })

    # Call
    result = compute_stim_artifact_template(df, [], window_ms=5.0)

    # Check
    assert isinstance(result, dict)
    assert len(result) == 0

def test_compute_stim_artifact_template_specific_channels():
    """Test compute_stim_artifact_template with specific channels"""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00'),
            pd.Timestamp('2023-01-01 12:00:00'),
            pd.Timestamp('2023-01-01 12:00:01'),
            pd.Timestamp('2023-01-01 12:00:01')
        ],
        'channel': [1, 2, 1, 2],
        'Amplitude': [-5.0, -6.0, -7.0, -8.0]
    })

    pulse_times = [
        pd.Timestamp('2023-01-01 12:00:00'),
        pd.Timestamp('2023-01-01 12:00:01')
    ]

    # Call
    result = compute_stim_artifact_template(df, pulse_times, window_ms=5.0, channels=[1])

    # Check
    assert isinstance(result, dict)
    assert 1 in result # Channel 1 should be in the result
    assert 2 not in result # Channel 2 should not be in the result
    assert isinstance(result[1], pd.DataFrame)
    assert len(result[1]) == 2

def test_apply_template_subtraction_basic():
    """Test apply_template_subtraction with basic valid data"""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00.000'),
            pd.Timestamp('2023-01-01 12:00:00.001'),
            pd.Timestamp('2023-01-01 12:00:01.000'),
            pd.Timestamp('2023-01-01 12:00:01.001')
        ],
        'channel': [1, 1, 1, 1],
        'Amplitude': [-5.0, -6.0, -7.0, -8.0]
    })

    pulse_times = [
        pd.Timestamp('2023-01-01 12:00:00'),
        pd.Timestamp('2023-01-01 12:00:01')
    ]

    templates = {
        1: pd.DataFrame({
            'relative_time_ms': [0.0, 1.0, 0.0, 1.0],
            'Amplitude': [-1.0, -2.0, -3.0, -4.0]
        })
    }

    # Call
    result = apply_template_subtraction(df, pulse_times, templates, window_ms=5.0)

    # Check
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert result['Amplitude'].iloc[0] == -4.0 # -5.0 - (-1.0) = -4.0
    assert result['Amplitude'].iloc[1] == -4.0 # -6.0 - (-2.0) = -4.0
    assert result['Amplitude'].iloc[2] == -6.0 # -7.0 - (-1.0) = -6.0
    assert result['Amplitude'].iloc[3] == -6.0 # -8.0 - (-2.0) = -6.0

def test_apply_template_subtraction_string_timestamps():
    """Test apply_template_subtraction with string timestamps"""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00.000'),
            pd.Timestamp('2023-01-01 12:00:00.001')
        ],
        'channel': [1, 1],
        'Amplitude': [-5.0, -6.0]
    })

    pulse_times = ['2023-01-01 12:00:00']
    templates = {
        1: pd.DataFrame({
            'relative_time_ms': [0.0, 1.0],
            'Amplitude': [-1.0, -2.0]
        })
    }

    # Call
    result = apply_template_subtraction(df, pulse_times, templates, window_ms=5.0)

    # Check
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert result['Amplitude'].iloc[0] == -4.0 # -5.0 - (-1.0)
    assert result['Amplitude'].iloc[1] == -4.0 # -6.0 - (-2.0)

def test_apply_template_subtraction_empty_data():
    """Test apply_template_subtraction with empty DataFrame"""

    empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])
    pulse_times = [pd.Timestamp('2023-01-01 12:00:00')]

    templates = {
        1: pd.DataFrame({
            'relative_time_ms': [0.0, 1.0],
            'Amplitude': [-1.0, -2.0]
        })
    }

    # Call
    result = apply_template_subtraction(empty_df, pulse_times, templates, window_ms=5.0)

    # Check
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_apply_template_subtraction_no_pulse_times():
    """Test apply_template_subtraction with no pulse times"""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00'),
            pd.Timestamp('2023-01-01 12:00:01')
        ],
        'channel': [1, 1],
        'Amplitude': [-5.0, -6.0]
    })
    templates = {
        1: pd.DataFrame({
            'relative_time_ms': [0.0, 1.0],
            'Amplitude': [-1.0, -2.0]
        })
    }

    result = apply_template_subtraction(df, [], templates, window_ms=5.0)

    # Check
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert result['Amplitude'].iloc[0] == -5.0
    assert result['Amplitude'].iloc[1] == -6.0

def test_apply_template_subtraction_no_templates():
    """Test apply_template_subtraction with no templates"""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00'),
            pd.Timestamp('2023-01-01 12:00:01')
        ],
        'channel': [1, 1],
        'Amplitude': [-5.0, -6.0]
    })

    pulse_times = [
        pd.Timestamp('2023-01-01 12:00:00'),
        pd.Timestamp('2023-01-01 12:00:01')
    ]

    result = apply_template_subtraction(df, pulse_times, {}, window_ms=5.0)

    # Check
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert result['Amplitude'].iloc[0] == -5.0
    assert result['Amplitude'].iloc[1] == -6.0

def test_apply_template_subtraction_multiple_channels():
    """Test apply_template_subtraction with multiple channels"""

    df = pd.DataFrame({
        'Time': [
            pd.Timestamp('2023-01-01 12:00:00'),
            pd.Timestamp('2023-01-01 12:00:00'),
            pd.Timestamp('2023-01-01 12:00:01'),
            pd.Timestamp('2023-01-01 12:00:01')
        ],
        'channel': [1, 2, 1, 2],
        'Amplitude': [-5.0, -6.0, -7.0, -8.0]
    })

    pulse_times = [
        pd.Timestamp('2023-01-01 12:00:00'),
        pd.Timestamp('2023-01-01 12:00:01')
    ]

    templates = {
        1: pd.DataFrame({
            'relative_time_ms': [0.0, 0.0],
            'Amplitude': [-1.0, -3.0]
        }),
        2: pd.DataFrame({
            'relative_time_ms': [0.0, 0.0],
            'Amplitude': [-2.0, -4.0]
        })
    }

    # Call
    result = apply_template_subtraction(df, pulse_times, templates, window_ms=5.0)

    # Check
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert result.loc[result['channel'] == 1, 'Amplitude'].iloc[0] == -4.0 # -5.0 - (-1.0) = -4.0
    assert result.loc[result['channel'] == 2, 'Amplitude'].iloc[0] == -4.0 # -6.0 - (-2.0) = -4.0
    assert result.loc[result['channel'] == 1, 'Amplitude'].iloc[1] == -6.0 # -7.0 - (-1.0) = -6.0
    assert result.loc[result['channel'] == 2, 'Amplitude'].iloc[1] == -6.0 # -8.0 - (-2.0) = -6.0
