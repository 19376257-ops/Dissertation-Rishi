import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.protocols.pse import PSEProtocol

class TestPSEProtocolExtended:
    """Extended tests for the PSEProtocol focusing on phase-specific analysis"""

    def test_preprocess_with_phase_specific_analysis(self, sample_config, sample_spike_data):
        """Test pre-process method with phase-specific analysis"""

        protocol = PSEProtocol(sample_config, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        df = sample_spike_data.copy()
        df['condition'] = 'entrain_phase1'

        # Mock
        protocol.cfg['experiment_params'] = {
            'mask_stim': True,
            'stim_pulse_timestamps': ['2023-01-01T12:00:05'],
            'mask_window_ms': 3.0,
            'stim_params': [
                {'index': 1}, # pre electrode for organoid 1
                {'index': 2}, # pre electrode for organoid 2
                {'index': 3}, # pre electrode for organoid 3
                {'index': 4}, # post electrode for organoid 1
                {'index': 5}, # post electrode for organoid 2
                {'index': 6}  # post electrode for organoid 3
            ]
        }

        # Mock
        with patch('src.protocols.pse.mask_stim_windows', return_value=df) as mock_mask:
            # Call
            result = protocol.preprocess(df)

            # Check
            assert isinstance(result, pd.DataFrame)

    def test_preprocess_with_train_phase(self, sample_config, sample_spike_data):
        """Test pre-process method with train phase"""

        protocol = PSEProtocol(sample_config, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        df = sample_spike_data.copy()
        df['condition'] = 'train_phase1'

        # Mock
        protocol.cfg['experiment_params'] = {
            'mask_stim': True,
            'stim_pulse_timestamps': ['2023-01-01T12:00:05'],
            'mask_window_ms': 3.0,
            'stim_params': [
                {'index': 1}, # pre electrode for organoid 1
                {'index': 2}, # pre electrode for organoid 2
                {'index': 3}, # pre electrode for organoid 3
                {'index': 4}, # post electrode for organoid 1
                {'index': 5}, # post electrode for organoid 2
                {'index': 6}  # post electrode for organoid 3
            ]
        }

        # Mock
        with patch('src.protocols.pse.mask_stim_windows', return_value=df) as mock_mask:
            # Call
            result = protocol.preprocess(df)

            # Check
            assert isinstance(result, pd.DataFrame)

    def test_preprocess_with_shift_phase(self, sample_config, sample_spike_data):
        """Test pre-process method with shift phase"""

        protocol = PSEProtocol(sample_config, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        df = sample_spike_data.copy()
        df['condition'] = 'shift_phase1'

        # Mock
        protocol.cfg['experiment_params'] = {
            'mask_stim': True,
            'stim_pulse_timestamps': ['2023-01-01T12:00:05'],
            'mask_window_ms': 3.0,
            'stim_params': [
                {'index': 1}, # pre electrode for organoid 1
                {'index': 2}, # pre electrode for organoid 2
                {'index': 3}, # pre electrode for organoid 3
                {'index': 4}, # post electrode for organoid 1
                {'index': 5}, # post electrode for organoid 2
                {'index': 6}  # post electrode for organoid 3
            ]
        }

        # Mock
        with patch('src.protocols.pse.mask_stim_windows', return_value=df) as mock_mask:
            # Call
            result = protocol.preprocess(df)

            # Check
            assert isinstance(result, pd.DataFrame)

    def test_run_analysis_for_data_with_condition_filtering(self, sample_config, sample_spike_data):
        """Test _run_analysis_for_data method with condition filtering"""
        protocol = PSEProtocol(sample_config, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        df = sample_spike_data.copy()
        df['condition'] = 'baseline'

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
             patch('src.protocols.pse.compute_kernel_density', return_value={}):

            # Call
            channels = df['channel'].unique()
            results = protocol._run_analysis_for_data(df, channels, 'baseline')

            # Check
            assert isinstance(results, list)
            for result in results:
                if 'data' in result and isinstance(result['data'], dict) and 'phase' in result['data']:
                    assert result['data']['phase'] == 'baseline'

    def test_analyse_with_embeddings(self, sample_config, sample_spike_data):
        """Test analyse method with embeddings"""

        protocol = PSEProtocol(sample_config, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        df = sample_spike_data.copy()
        df['condition'] = 'baseline'
        large_df = pd.concat([df] * 20, ignore_index=True)
        protocol.cfg['experiment_params'] = {
            'phase_groups': {
                'all_phases': ['baseline']
            },
            'phase_names': ['baseline']
        }

        sample_umap_result = {
            'embedding': np.random.random((10, 3)),
            'n_components': 3,
            'pca_components': 12,
            'pca_explained_variance_ratio': np.array([0.5, 0.3, 0.2]),
            'bin_size_ms': 10,
            'binned_data': pd.DataFrame(),
            'conditions': np.array(['baseline'] * 10)
        }

        sample_pca_result = {
            'embedding': np.random.random((10, 3)),
            'n_components': 3,
            'explained_variance_ratio': np.array([0.5, 0.3, 0.2]),
            'bin_size_ms': 10,
            'binned_data': pd.DataFrame(),
            'conditions': np.array(['baseline'] * 10)
        }

        # Mock
        with patch('src.protocols.pse.compute_umap_embedding', return_value=sample_umap_result) as mock_umap, \
             patch('src.protocols.pse.compute_pca_embedding', return_value=sample_pca_result) as mock_pca, \
             patch.object(protocol, '_run_analysis_for_data', return_value=[]), \
             patch.object(protocol, 'visualise'):

            # Call
            results = protocol.analyse(large_df)

            # Check
            assert isinstance(results, list)
