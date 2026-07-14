import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import datetime

@pytest.fixture
def sample_spike_data():
    """Generate a sample DataFrame with spike data for testing"""

    np.random.seed(42) # For reproducibility
    start_time = datetime.datetime(2023, 1, 1, 12, 0, 0)
    times = [start_time + datetime.timedelta(milliseconds=int(t)) 
             for t in np.sort(np.random.randint(0, 10000, 100))]
    channels = np.random.randint(1, 33, 100)
    amplitudes = -np.random.rand(100) * 10
    df = pd.DataFrame({
        'Time': times,
        'channel': channels,
        'Amplitude': amplitudes
    })
    
    return df

@pytest.fixture
def sample_config():
    """Create a sample cfg dict"""
    return {
        'file_pattern': '*.csv',
        'raster_threshold': None,
        'fft_resample_ms': 1,
        'fft_freq_cap': 10.0,
        'avalanche_threshold_s': 0.1,
        'firing_rate_threshold': 0.0,
        'firing_rate_bin_size_s': 1.0,
        'psd_nperseg': None,
        'psd_noverlap': None,
        'gc_resample_interval': '1s',
        'gc_max_lag': 5,
        'bin_size_s': 1.0,
        'artefact_thresh': 8.0,
        'experiment_params': {
            'phase_timestamps': [
                '2023-01-01T12:00:05',
                '2023-01-01T12:00:10'
            ],
            'phase_names': ['baseline', 'stim']
        }
    }

@pytest.fixture
def sample_paths():
    """Create smpl paths dicts"""
    return {
        'raw': 'tests/data/raw',
        'results': 'tests/data/results'
    }

@pytest.fixture
def ensure_test_dirs(sample_paths):
    """Ensure test dict exist"""
    for path_key, path_val in sample_paths.items():
        Path(path_val).mkdir(parents=True, exist_ok=True)
    yield
