import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.protocols.pse import PSEProtocol

class TestPSEProtocol:
    """Tests for the PSEProtocol class"""

    def test_init(self, sample_config, sample_paths):
        """Test init of PSEProtocol"""
        protocol = PSEProtocol(sample_config, sample_paths)

        # Check
        assert protocol.cfg == sample_config
        assert protocol.raw_dir == Path(sample_paths['raw'])
        assert protocol.results_dir == Path(sample_paths['results'])

    def test_preprocess(self, sample_spike_data, sample_config):
        """Test pre-process method"""
        protocol = PSEProtocol(sample_config, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        # Call
        result = protocol.preprocess(sample_spike_data)

        # Check
        assert isinstance(result, pd.DataFrame)
        assert 'Time' in result.columns
        assert 'channel' in result.columns
        assert 'Amplitude' in result.columns
        assert pd.api.types.is_datetime64_dtype(result['Time'])

    def test_preprocess_empty(self):
        """Test pre-process method with empty DataFrame"""

        protocol = PSEProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])

        # Call
        result = protocol.preprocess(empty_df)

        # Check
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_analyse(self, sample_spike_data, sample_config):
        """Test analyse method"""
        protocol = PSEProtocol(sample_config, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        sample_data_with_condition = sample_spike_data.copy()
        sample_data_with_condition['condition'] = 'baseline'

        # Mock
        with patch('src.protocols.pse.compute_fft', return_value={'freqs': np.array([]), 'amplitudes': np.array([])}), \
             patch('src.protocols.pse.compute_isi', return_value={'isi': np.array([]), 'mean_isi': 0, 'std_isi': 0}), \
             patch('src.protocols.pse.compute_avalanche_distribution', return_value=np.array([])), \
             patch('src.protocols.pse.compute_powerlaw', return_value={}), \
             patch('src.protocols.pse.compute_firing_rates', return_value={}), \
             patch('src.protocols.pse.compute_power_spectral_density', return_value={'freqs': np.array([]), 'psd': np.array([])}), \
             patch('src.protocols.pse.compute_auto_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}), \
             patch('src.protocols.pse.compute_granger_causality', return_value={}), \
             patch('src.protocols.pse.compute_statistical_tests', return_value={}), \
             patch('src.protocols.pse.compute_cross_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}), \
             patch('src.protocols.pse.compute_firing_rate_zscore', return_value={}), \
             patch('src.protocols.pse.compute_kernel_density', return_value={}), \
             patch('src.protocols.pse.compute_umap_embedding', return_value={}), \
             patch('src.protocols.pse.compute_pca_embedding', return_value={}), \
             patch('src.protocols.pse.mask_stim_windows', return_value=sample_data_with_condition), \
             patch.object(protocol, '_run_analysis_for_data', return_value=[]):

            # Call
            results = protocol.analyse(sample_data_with_condition)

            # Check
            assert isinstance(results, list)

    def test_analyse_empty(self, sample_config):
        """Test analyse method with empty DataFrame"""
        protocol = PSEProtocol(sample_config, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])

        # Call
        results = protocol.analyse(empty_df)

        # Check
        assert isinstance(results, list)
        assert len(results) == 0

    def test_analyse_uses_phase_timestamps_as_interval_boundaries(self):
        """Test adapter-style phase timestamps are treated as interval boundaries."""
        cfg = {
            'experiment_params': {
                'phase_names': ['baseline', 'train', 'post_train'],
                'phase_timestamps': [
                    '2024-05-02T09:00:00',
                    '2024-05-02T09:05:00',
                    '2024-05-02T09:10:00',
                    '2024-05-02T09:15:00',
                ],
            }
        }
        protocol = PSEProtocol(cfg, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        df = pd.DataFrame({
            'Time': pd.to_datetime([
                '2024-05-02T09:00:10',
                '2024-05-02T09:05:10',
                '2024-05-02T09:10:10',
            ]),
            'channel': [101, 101, 101],
            'Amplitude': [-1.0, -2.0, -3.0],
        })

        with patch.object(protocol, '_run_analysis_for_data', return_value=[]) as mock_run, \
             patch('src.protocols.pse.compute_umap_embedding', return_value={'error': 'skip'}), \
             patch('src.protocols.pse.compute_pca_embedding', return_value={'error': 'skip'}):
            results = protocol.analyse(df)

        called_phases = [call.args[2] for call in mock_run.call_args_list]
        assert called_phases == ['baseline', 'train', 'post_train']
        assert set(df['condition']) == {'baseline', 'train', 'post_train'}
        assert results[0]['type'] == 'pse_summary'

    def test_run_analysis_for_data(self, sample_spike_data, sample_config):
        """Test _run_analysis_for_data method"""

        protocol = PSEProtocol(sample_config, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        sample_data_with_condition = sample_spike_data.copy()
        sample_data_with_condition['condition'] = 'baseline'

        # Mock
        with patch('src.protocols.pse.compute_fft', return_value={'freqs': np.array([]), 'amplitudes': np.array([])}), \
             patch('src.protocols.pse.compute_isi', return_value={'isi': np.array([]), 'mean_isi': 0, 'std_isi': 0}), \
             patch('src.protocols.pse.compute_avalanche_distribution', return_value=np.array([])), \
             patch('src.protocols.pse.compute_powerlaw', return_value={}), \
             patch('src.protocols.pse.compute_firing_rates', return_value={}), \
             patch('src.protocols.pse.compute_power_spectral_density', return_value={'freqs': np.array([]), 'psd': np.array([])}), \
             patch('src.protocols.pse.compute_auto_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}), \
             patch('src.protocols.pse.compute_granger_causality', return_value={}), \
             patch('src.protocols.pse.compute_statistical_tests', return_value={}), \
             patch('src.protocols.pse.compute_cross_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}):

            # Call
            channels = sample_data_with_condition['channel'].unique()
            results = protocol._run_analysis_for_data(sample_data_with_condition, channels, 'test_phase')

            # Check
            assert isinstance(results, list)

    def test_run_integration(self, sample_config, sample_paths, ensure_test_dirs):
        """Test run method integration"""

        protocol = PSEProtocol(sample_config, sample_paths)

        test_csv = Path(sample_paths['raw']) / 'test.csv'
        df = pd.DataFrame({
            'Time': ['2023-01-01 12:00:00', '2023-01-01 12:00:01'],
            'channel': [1, 2],
            'Amplitude': [-5.0, -6.0]
        })
        df.to_csv(test_csv, index=False)

        # Mock
        with patch('src.protocols.pse.compute_fft', return_value={'freqs': np.array([]), 'amplitudes': np.array([])}), \
             patch('src.protocols.pse.compute_isi', return_value={'isi': np.array([]), 'mean_isi': 0, 'std_isi': 0}), \
             patch('src.protocols.pse.compute_avalanche_distribution', return_value=np.array([])), \
             patch('src.protocols.pse.compute_powerlaw', return_value={}), \
             patch('src.protocols.pse.compute_firing_rates', return_value={}), \
             patch('src.protocols.pse.compute_power_spectral_density', return_value={'freqs': np.array([]), 'psd': np.array([])}), \
             patch('src.protocols.pse.compute_auto_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}), \
             patch('src.protocols.pse.compute_granger_causality', return_value={}), \
             patch('src.protocols.pse.compute_statistical_tests', return_value={}), \
             patch('src.protocols.pse.compute_cross_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}), \
             patch('src.protocols.pse.compute_umap_embedding', return_value={}), \
             patch('src.protocols.pse.compute_pca_embedding', return_value={}), \
             patch('src.protocols.pse.compute_firing_rate_zscore', return_value={}), \
             patch('src.protocols.pse.compute_kernel_density', return_value={}), \
             patch('src.protocols.pse.mask_stim_windows', return_value=df):

            with patch.object(protocol, 'visualise'):
                # Call
                results = protocol.run()

                # Check
                assert isinstance(results, list)
