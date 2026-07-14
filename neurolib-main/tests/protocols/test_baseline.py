import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.protocols.baseline import BaselineProtocol

class TestBaselineProtocol:
    """Tests for the BaselineProtocol class."""

    def test_init(self, sample_config, sample_paths):
        """Test init of BaselineProtocol."""
        protocol = BaselineProtocol(sample_config, sample_paths)

        assert protocol.cfg == sample_config
        assert protocol.raw_dir == Path(sample_paths['raw'])
        assert protocol.results_dir == Path(sample_paths['results'])

    def test_preprocess(self, sample_spike_data):
        """Test preprocess method."""
        protocol = BaselineProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        # Call
        result = protocol.preprocess(sample_spike_data)

        # Check
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_spike_data)
        assert 'Time' in result.columns
        assert 'channel' in result.columns
        assert 'Amplitude' in result.columns
        assert pd.api.types.is_datetime64_dtype(result['Time'])

    def test_preprocess_empty(self):
        """Test pre-process method with empty DataFrame"""

        protocol = BaselineProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])

        # Call
        result = protocol.preprocess(empty_df)

        # Check
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_analyse(self, sample_spike_data):
        """Test analyse method."""
        protocol = BaselineProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        # Mockn
        with patch('src.protocols.baseline.compute_fft', return_value={'freqs': np.array([]), 'amplitudes': np.array([])}), \
             patch('src.protocols.baseline.compute_isi', return_value={'isi': np.array([]), 'mean_isi': 0, 'std_isi': 0}), \
             patch('src.protocols.baseline.compute_avalanche_distribution', return_value=np.array([])), \
             patch('src.protocols.baseline.compute_powerlaw', return_value={}), \
             patch('src.protocols.baseline.compute_firing_rates', return_value={}), \
             patch('src.protocols.baseline.compute_power_spectral_density', return_value={'freqs': np.array([]), 'psd': np.array([])}), \
             patch('src.protocols.baseline.compute_auto_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}), \
             patch('src.protocols.baseline.compute_granger_causality', return_value={}), \
             patch('src.protocols.baseline.compute_statistical_tests', return_value={}), \
             patch('src.protocols.baseline.compute_cross_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}):

            # Call
            results = protocol.analyse(sample_spike_data)

            # Check
            assert isinstance(results, list)
            channels = sample_spike_data['channel'].unique()
            assert len(results) >= len(channels)

    def test_analyse_empty(self):
        """Test analyse method with empty DataFrame"""

        protocol = BaselineProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])

        # Call
        results = protocol.analyse(empty_df)

        # Check
        assert isinstance(results, list)
        assert len(results) == 0

    def test_run_integration(self, sample_config, sample_paths, ensure_test_dirs):
        """Test run method integration"""
        protocol = BaselineProtocol(sample_config, sample_paths)

        test_csv = Path(sample_paths['raw']) / 'test.csv'
        df = pd.DataFrame({
            'Time': ['2023-01-01 12:00:00', '2023-01-01 12:00:01'],
            'channel': [1, 2],
            'Amplitude': [-5.0, -6.0]
        })
        df.to_csv(test_csv, index=False)

        # Mock
        with patch('src.protocols.baseline.compute_fft', return_value={'freqs': np.array([]), 'amplitudes': np.array([])}), \
             patch('src.protocols.baseline.compute_isi', return_value={'isi': np.array([]), 'mean_isi': 0, 'std_isi': 0}), \
             patch('src.protocols.baseline.compute_avalanche_distribution', return_value=np.array([])), \
             patch('src.protocols.baseline.compute_powerlaw', return_value={}), \
             patch('src.protocols.baseline.compute_firing_rates', return_value={}), \
             patch('src.protocols.baseline.compute_power_spectral_density', return_value={'freqs': np.array([]), 'psd': np.array([])}), \
             patch('src.protocols.baseline.compute_auto_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}), \
             patch('src.protocols.baseline.compute_granger_causality', return_value={}), \
             patch('src.protocols.baseline.compute_statistical_tests', return_value={}), \
             patch('src.protocols.baseline.compute_cross_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}):

            with patch.object(protocol, 'visualise'):
                # Call
                results = protocol.run()

                # Check
                assert isinstance(results, list)
