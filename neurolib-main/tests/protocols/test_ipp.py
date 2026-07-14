import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.protocols.ipp import IPPProtocol

class TestIPPProtocol:
    """Tests for the IPPProtocol class"""

    def test_init(self, sample_config, sample_paths):
        """Test init of IPPProtocol"""

        protocol = IPPProtocol(sample_config, sample_paths)

        # Check
        assert protocol.cfg == sample_config
        assert protocol.raw_dir == Path(sample_paths['raw'])
        assert protocol.results_dir == Path(sample_paths['results'])

    def test_preprocess(self, sample_spike_data):
        """Test preprocess method"""

        protocol = IPPProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

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

        protocol = IPPProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])

        # Call
        result = protocol.preprocess(empty_df)

        # Check
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_analyse(self, sample_spike_data):
        """Test analyse method"""

        protocol = IPPProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        sample_data_with_original = sample_spike_data.copy()
        sample_data_with_original['original_channel'] = sample_data_with_original['channel']

        # Mock
        with patch('src.protocols.ipp.compute_fft', return_value={'freqs': np.array([]), 'amplitudes': np.array([])}), \
             patch('src.protocols.ipp.compute_isi', return_value={'isi': np.array([]), 'mean_isi': 0, 'std_isi': 0}), \
             patch('src.protocols.ipp.compute_avalanche_distribution', return_value=np.array([])), \
             patch('src.protocols.ipp.compute_powerlaw', return_value={}), \
             patch('src.protocols.ipp.compute_firing_rates', return_value={}), \
             patch('src.protocols.ipp.compute_power_spectral_density', return_value={'freqs': np.array([]), 'psd': np.array([])}), \
             patch('src.protocols.ipp.compute_granger_causality', return_value={}), \
             patch('src.protocols.ipp.compute_statistical_tests', return_value={}), \
             patch('src.protocols.ipp.compute_cross_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}), \
             patch('src.protocols.ipp.compute_firing_rate_zscore', return_value={}), \
             patch('src.protocols.ipp.compute_kernel_density', return_value={}), \
             patch('src.protocols.ipp.compute_inferential_statistics', return_value={}), \
             patch('src.protocols.ipp.mask_stim_windows', return_value=sample_data_with_original):

            # Call
            results = protocol.analyse(sample_data_with_original)

            # Check
            assert isinstance(results, list)
            assert len(results) > 0

    def test_analyse_empty(self):
        """Test analyse method with empty DataFrame"""

        protocol = IPPProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])

        # Call
        results = protocol.analyse(empty_df)

        # Check
        assert isinstance(results, list)
        assert len(results) == 0

    def test_run_integration(self, sample_config, sample_paths, ensure_test_dirs):
        """Test run method integration"""

        protocol = IPPProtocol(sample_config, sample_paths)
        test_csv = Path(sample_paths['raw']) / 'test.csv'
        df = pd.DataFrame({
            'Time': ['2023-01-01 12:00:00', '2023-01-01 12:00:01'],
            'channel': [1, 2],
            'Amplitude': [-5.0, -6.0]
        })
        df.to_csv(test_csv, index=False)

        # Mock
        with patch.object(protocol, 'load', return_value=df):
            with patch('src.protocols.ipp.compute_fft', return_value={'freqs': np.array([]), 'amplitudes': np.array([])}), \
                 patch('src.protocols.ipp.compute_isi', return_value={'isi': np.array([]), 'mean_isi': 0, 'std_isi': 0}), \
                 patch('src.protocols.ipp.compute_avalanche_distribution', return_value=np.array([])), \
                 patch('src.protocols.ipp.compute_powerlaw', return_value={}), \
                 patch('src.protocols.ipp.compute_firing_rates', return_value={}), \
                 patch('src.protocols.ipp.compute_power_spectral_density', return_value={'freqs': np.array([]), 'psd': np.array([])}), \
                 patch('src.protocols.ipp.compute_granger_causality', return_value={}), \
                 patch('src.protocols.ipp.compute_statistical_tests', return_value={}), \
                 patch('src.protocols.ipp.compute_cross_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}), \
                 patch('src.protocols.ipp.compute_firing_rate_zscore', return_value={}), \
                 patch('src.protocols.ipp.compute_kernel_density', return_value={}), \
                 patch('src.protocols.ipp.compute_inferential_statistics', return_value={}), \
                 patch('src.protocols.ipp.mask_stim_windows', return_value=df):
                with patch.object(protocol, 'visualise'):
                    # Call
                    results = protocol.run()

                    # Check
                    assert isinstance(results, list)
