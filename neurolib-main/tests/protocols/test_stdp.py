import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.protocols.stdp import STDPProtocol
from src.analysis.utils import mask_stim_windows, compute_stim_artifact_template, apply_template_subtraction

class TestSTDPProtocol:
    """Tests for the STDPProtoco"""

    def test_init(self, sample_config, sample_paths):
        """Test init of STDPProtocol"""
        protocol = STDPProtocol(sample_config, sample_paths)

        # Check
        assert protocol.cfg == sample_config
        assert protocol.raw_dir == Path(sample_paths['raw'])
        assert protocol.results_dir == Path(sample_paths['results'])

    def test_load(self, sample_paths, ensure_test_dirs):
        """Test load method"""

        for file in Path(sample_paths['raw']).glob('*.csv'):
            file.unlink()

        protocol = STDPProtocol({}, sample_paths)
        test_csv = Path(sample_paths['raw']) / 'test.csv'
        df = pd.DataFrame({
            'Time': ['2023-01-01 12:00:00', '2023-01-01 12:00:01'],
            'channel': [1, 2],
            'Amplitude': [-5.0, -6.0]
        })
        df.to_csv(test_csv, index=False)

        # Call
        result = protocol.load()

        # Check
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'Time' in result.columns
        assert 'channel' in result.columns
        assert 'Amplitude' in result.columns

    def test_load_with_custom_pattern(self, sample_paths, ensure_test_dirs):
        """Test load method with custom file pattern"""

        for file in Path(sample_paths['raw']).glob('*.csv'):
            file.unlink()

        protocol = STDPProtocol({'file_pattern': 'custom_*.csv'}, sample_paths)

        test_csv = Path(sample_paths['raw']) / 'custom_test.csv'
        df = pd.DataFrame({
            'Time': ['2023-01-01 12:00:00', '2023-01-01 12:00:01'],
            'channel': [1, 2],
            'Amplitude': [-5.0, -6.0]
        })
        df.to_csv(test_csv, index=False)

        # Call
        result = protocol.load()

        # Check
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_load_no_files(self, sample_paths, ensure_test_dirs):
        """Test load method with no matching files."""

        for file in Path(sample_paths['raw']).glob('*.csv'):
            file.unlink()

        protocol = STDPProtocol({'file_pattern': 'nonexistent_*.csv'}, sample_paths)

        with pytest.raises(FileNotFoundError):
            protocol.load()

    def test_load_invalid_files(self, sample_paths, ensure_test_dirs):
        """Test load method with invalid files"""

        for file in Path(sample_paths['raw']).glob('*.csv'):
            file.unlink()

        protocol = STDPProtocol({}, sample_paths)
        test_csv = Path(sample_paths['raw']) / 'invalid.csv'
        df = pd.DataFrame({
            'InvalidColumn': [1, 2]
        })
        df.to_csv(test_csv, index=False)

        with pytest.raises(RuntimeError):
            protocol.load()

    def test_preprocess(self, sample_spike_data):
        """Test pre-process method."""
        protocol = STDPProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        # Call
        result = protocol.preprocess(sample_spike_data)

        # Check
        assert isinstance(result, pd.DataFrame)
        assert 'Time' in result.columns
        assert 'channel' in result.columns
        assert 'Amplitude' in result.columns
        assert pd.api.types.is_datetime64_dtype(result['Time'])

    def test_preprocess_empty(self):
        """Test pre-process method with emty DataFrame."""

        protocol = STDPProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])

        # Call
        result = protocol.preprocess(empty_df)

        # Check
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_preprocess_with_template_subtraction(self, sample_spike_data):
        """Test pre-process method with template subtraction"""

        config = {
            'experiment_params': {
                'mask_stim': True,
                'use_template_subtraction': True,
                'pre_electrodes': [1, 2],
                'post_electrodes': [3, 4],
                'stim_pulse_timestamps': ['2023-01-01T12:00:05', '2023-01-01T12:00:10'],
                'mask_window_ms': 3.0
            }
        }
        protocol = STDPProtocol(config, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        # Mock
        with patch('src.protocols.stdp.compute_stim_artifact_template', return_value={1: pd.DataFrame({'Time': [pd.Timestamp('2023-01-01 12:00:05')], 'Amplitude': [-5.0], 'relative_time_ms': [0.0]})}), \
             patch('src.protocols.stdp.apply_template_subtraction', return_value=sample_spike_data), \
             patch('src.protocols.stdp.mask_stim_windows', return_value=sample_spike_data):

            result = protocol.preprocess(sample_spike_data)

            # Check
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_spike_data)

    def test_preprocess_with_mask_stim_windows(self, sample_spike_data):
        """Test pre-process method with mask_stim_windows"""

        config = {
            'experiment_params': {
                'mask_stim': True,
                'use_template_subtraction': False,
                'pre_electrodes': [1, 2],
                'post_electrodes': [3, 4],
                'stim_pulse_timestamps': ['2023-01-01T12:00:05', '2023-01-01T12:00:10'],
                'mask_window_ms': 3.0
            }
        }
        protocol = STDPProtocol(config, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        # Mock
        with patch('src.protocols.stdp.mask_stim_windows', return_value=sample_spike_data) as mock_mask:

            # Call
            result = protocol.preprocess(sample_spike_data)

            # Check
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_spike_data)
            mock_mask.assert_called_once()

    def test_analyse(self, sample_spike_data, sample_config):
        """Test analyse method"""

        protocol = STDPProtocol(sample_config, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        sample_data_with_condition = sample_spike_data.copy()
        sample_data_with_condition['condition'] = 'pre'

        # Mock
        with patch.object(protocol, '_analyse_channel_metrics', return_value=None), \
             patch.object(protocol, '_analyse_channel_pair_metrics', return_value=None), \
             patch.object(protocol, '_analyse_embeddings', return_value=None), \
             patch.object(protocol, '_analyse_prcs', return_value=None), \
             patch('src.protocols.stdp.compute_fft', return_value={'freqs': np.array([]), 'amplitudes': np.array([])}), \
             patch('src.protocols.stdp.compute_isi', return_value={'isi': np.array([]), 'mean_isi': 0, 'std_isi': 0}), \
             patch('src.protocols.stdp.compute_avalanche_distribution', return_value=np.array([])), \
             patch('src.protocols.stdp.compute_powerlaw', return_value={}), \
             patch('src.protocols.stdp.compute_firing_rates', return_value={}), \
             patch('src.protocols.stdp.compute_power_spectral_density', return_value={'freqs': np.array([]), 'psd': np.array([])}), \
             patch('src.protocols.stdp.compute_auto_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}), \
             patch('src.protocols.stdp.compute_granger_causality', return_value={}), \
             patch('src.protocols.stdp.compute_statistical_tests', return_value={}), \
             patch('src.protocols.stdp.compute_cross_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}), \
             patch('src.protocols.stdp.compute_umap_embedding', return_value={}), \
             patch('src.protocols.stdp.compute_pca_embedding', return_value={}), \
             patch('src.protocols.stdp.compute_prc', return_value={}), \
             patch('src.protocols.stdp.group_conditions_by_pre_post', return_value={'pre': sample_data_with_condition, 'post': sample_data_with_condition}), \
             patch('src.protocols.stdp.mask_stim_windows', return_value=sample_data_with_condition):

            # Call
            results = protocol.analyse(sample_data_with_condition)

            # Check
            assert isinstance(results, list)

    def test_analyse_empty(self):
        """Test analyse method with empty DataFrame"""

        protocol = STDPProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])

        # Call
        results = protocol.analyse(empty_df)

        # Check
        assert isinstance(results, list)
        assert len(results) == 0

    def test_analyse_channel_metrics(self, sample_spike_data):
        """Test _analyse_channel_metrics method"""

        protocol = STDPProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        # Mock
        with patch('src.protocols.stdp.compute_fft', return_value={'freqs': np.array([]), 'amplitudes': np.array([])}), \
             patch('src.protocols.stdp.compute_isi', return_value={'isi': np.array([]), 'mean_isi': 0, 'std_isi': 0}), \
             patch('src.protocols.stdp.compute_avalanche_distribution', return_value=np.array([])), \
             patch('src.protocols.stdp.compute_powerlaw', return_value={}), \
             patch('src.protocols.stdp.compute_firing_rates', return_value={}), \
             patch('src.protocols.stdp.compute_power_spectral_density', return_value={'freqs': np.array([]), 'psd': np.array([])}), \
             patch('src.protocols.stdp.compute_auto_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}):

            # Call
            channels = sample_spike_data['channel'].unique()
            results = []
            protocol._analyse_channel_metrics(sample_spike_data, channels, results)

            # Check
            assert isinstance(results, list)

    def test_analyse_channel_metrics_with_avalanche(self, sample_spike_data):
        """Test _analyse_channel_metrics method with avalanche distribution computation."""
        protocol = STDPProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        sample_isi = {'isi': np.array([0.05, 0.2, 0.05, 0.05, 0.3]), 'mean_isi': 0.13, 'std_isi': 0.12}
        sample_avalanches = np.array([2, 1])

        # Mock
        with patch('src.protocols.stdp.compute_fft', return_value={'freqs': np.array([]), 'amplitudes': np.array([])}), \
             patch('src.protocols.stdp.compute_isi', return_value=sample_isi), \
             patch('src.protocols.stdp.compute_avalanche_distribution', return_value=sample_avalanches) as mock_avalanche, \
             patch('src.protocols.stdp.compute_powerlaw', return_value={}), \
             patch('src.protocols.stdp.compute_firing_rates', return_value={}), \
             patch('src.protocols.stdp.compute_power_spectral_density', return_value={'freqs': np.array([]), 'psd': np.array([])}), \
             patch('src.protocols.stdp.compute_auto_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}):

            # Call
            channels = sample_spike_data['channel'].unique()[:1]
            results = []
            protocol._analyse_channel_metrics(sample_spike_data, channels, results)

            # Check
            assert isinstance(results, list)
            mock_avalanche.assert_called_once_with(
                isi=sample_isi['isi'],
                threshold_s=protocol.cfg.get('avalanche_threshold_s', 0.1)
            )
            avalanche_results = [r for r in results if r['type'] == 'avalanche']
            assert len(avalanche_results) == 1
            assert avalanche_results[0]['data']['channel'] == channels[0]
            assert np.array_equal(avalanche_results[0]['data']['avalanches'], sample_avalanches)

    def test_analyse_channel_pair_metrics(self, sample_spike_data):
        """Test _analyse_channel_pair_metrics method"""

        protocol = STDPProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        # Mock
        with patch('src.protocols.stdp.compute_granger_causality', return_value={}), \
             patch('src.protocols.stdp.compute_cross_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}):

            # Call
            channels = sample_spike_data['channel'].unique()[:2]
            results = []
            protocol._analyse_channel_pair_metrics(sample_spike_data, channels, results)

            # Check
            assert isinstance(results, list)

    def test_analyse_prcs(self, sample_spike_data):
        """Test _analyse_prcs method with PRC analysis"""

        config = {
            'experiment_params': {
                'train_timestamps': {
                    'pre_times': ['2023-01-01T12:00:05', '2023-01-01T12:00:10'],
                    'post_times': ['2023-01-01T12:00:15', '2023-01-01T12:00:20']
                },
                'stdp_frequency_hz': 10.0
            },
            'prc_num_bins': 50
        }
        protocol = STDPProtocol(config, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        sample_data_with_condition = sample_spike_data.copy()
        sample_data_with_condition['condition'] = 'train_1_pre'
        sample_prc_result = {
            'channel': 1,
            'phase_bins': np.linspace(0, 1, 50),
            'prc_train_1': np.random.random(50),
            'prc_train_2': np.random.random(50)
        }

        # Mock
        with patch('src.protocols.stdp.compute_prc', return_value=sample_prc_result) as mock_prc:

            # Call
            channels = [1]
            results = []
            protocol._analyse_prcs(sample_data_with_condition, channels, results)

            # Check
            assert isinstance(results, list)
            assert mock_prc.call_count > 0

    def test_process_prc_results(self):
        """Test _process_prc_results method"""

        protocol = STDPProtocol({'prc_num_bins': 50}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        num_bins = 50
        pre_prc_res = {
            'channel': 1,
            'phase_bins': np.linspace(0, 1, num_bins),
            'prc_train_1': np.random.random(num_bins),
            'prc_train_2': np.random.random(num_bins),
            'prc_train_3': np.random.random(num_bins),
            'prc_train_4': np.random.random(num_bins),
            'prc_train_5': np.random.random(num_bins)
        }
        post_prc_res = {
            'channel': 1,
            'phase_bins': np.linspace(0, 1, num_bins),
            'prc_train_1': np.random.random(num_bins),
            'prc_train_2': np.random.random(num_bins),
            'prc_train_3': np.random.random(num_bins),
            'prc_train_4': np.random.random(num_bins),
            'prc_train_5': np.random.random(num_bins)
        }

        # Call
        results = []
        protocol._process_prc_results(pre_prc_res, post_prc_res, 1, results)

        # Check
        assert isinstance(results, list)
        assert len(results) == 1
        prc_result = results[0]
        assert prc_result['type'] == 'prc'
        assert 'data' in prc_result
        assert 'phase_bins' in prc_result['data']
        assert 'delta_phase' in prc_result['data']
        assert 'electrode' in prc_result['data']
        assert prc_result['data']['electrode'] == 1
        delta_phase = prc_result['data']['delta_phase']
        for i in range(5):
            assert f'train_{i}_pre' in delta_phase
            assert f'train_{i}_post' in delta_phase

    def test_process_prc_results_with_error(self):
        """Test _process_prc_results method with error in PRC results"""

        protocol = STDPProtocol({'prc_num_bins': 50}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        pre_prc_res = {'error': 'No valid PRCs computed'}
        post_prc_res = {'error': 'No valid PRCs computed'}

        # Call
        results = []
        protocol._process_prc_results(pre_prc_res, post_prc_res, 1, results)

        # Check
        assert isinstance(results, list)
        assert len(results) == 0

    def test_analyse_embeddings(self, sample_spike_data):
        """Test _analyse_embeddings method with UMAP and PCA"""

        protocol = STDPProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        large_df = pd.concat([sample_spike_data] * 2, ignore_index=True)
        large_df['condition'] = 'train_1_pre'
        sample_umap_result = {
            'embedding': np.random.random((10, 3)),
            'n_components': 3,
            'pca_components': 12,
            'pca_explained_variance_ratio': np.array([0.5, 0.3, 0.2]),
            'bin_size_ms': 10,
            'binned_data': pd.DataFrame(),
            'conditions': np.array(['train_1_pre'] * 10)
        }
        sample_pca_result = {
            'embedding': np.random.random((10, 3)),
            'n_components': 3,
            'explained_variance_ratio': np.array([0.5, 0.3, 0.2]),
            'bin_size_ms': 10,
            'binned_data': pd.DataFrame(),
            'conditions': np.array(['train_1_pre'] * 10)
        }

        # Mock
        with patch('src.protocols.stdp.compute_umap_embedding', return_value=sample_umap_result) as mock_umap, \
             patch('src.protocols.stdp.compute_pca_embedding', return_value=sample_pca_result) as mock_pca, \
             patch('src.protocols.stdp.group_conditions_by_pre_post', return_value=np.array(['pre'] * 10)):

            # Call
            results = []
            protocol._analyse_embeddings(large_df, results, sample_mode=True)

            # Check
            assert isinstance(results, list)
            mock_umap.assert_called_once()
            mock_pca.assert_called_once()
            umap_results = [r for r in results if r['type'] == 'umap']
            pca_results = [r for r in results if r['type'] == 'pca']
            assert len(umap_results) == 1
            assert len(pca_results) == 1

    def test_visualise(self):
        """Test visualise method"""

        protocol = STDPProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        results = [
            {
                'type': 'fft',
                'data': {
                    'channel': 1,
                    'freqs': np.array([1, 2, 3]),
                    'amplitudes': np.array([0.1, 0.2, 0.3])
                }
            },
            {
                'type': 'isi',
                'data': {
                    'channel': 1,
                    'isi': np.array([0.1, 0.2, 0.3]),
                    'mean_isi': 0.2,
                    'std_isi': 0.1
                }
            },
            {
                'type': 'avalanche',
                'data': {
                    'channel': 1,
                    'avalanches': np.array([2, 3, 1])
                }
            },
            {
                'type': 'powerlaw',
                'data': {
                    'channel': 1,
                    'alpha': 1.5,
                    'xmin': 1.0,
                    'xmax': 10.0
                }
            },
            {
                'type': 'firing_rate',
                'data': {
                    'channel': 1,
                    'times': np.array([1, 2, 3]),
                    'rates': np.array([0.1, 0.2, 0.3])
                }
            },
            {
                'type': 'psd',
                'data': {
                    'channel': 1,
                    'freqs': np.array([1, 2, 3]),
                    'psd': np.array([0.1, 0.2, 0.3])
                }
            },
            {
                'type': 'auto_correlation',
                'data': {
                    'channel': 1,
                    'lags': np.array([1, 2, 3]),
                    'correlation': np.array([0.1, 0.2, 0.3])
                }
            },
            {
                'type': 'cross_correlation',
                'data': {
                    'channel1': 1,
                    'channel2': 2,
                    'lags': np.array([1, 2, 3]),
                    'correlation': np.array([0.1, 0.2, 0.3])
                }
            },
            {
                'type': 'granger_causality',
                'data': {
                    'channel1': 1,
                    'channel2': 2,
                    'causality': 0.5
                }
            },
            {
                'type': 'statistical_tests',
                'data': {
                    'channel1': 1,
                    'channel2': 2,
                    'ks_test': {'statistic': 0.5, 'pvalue': 0.05}
                }
            },
            {
                'type': 'prc',
                'data': {
                    'phase_bins': np.linspace(0, 1, 10),
                    'delta_phase': {
                        'train_0_pre': np.random.random(10),
                        'train_0_post': np.random.random(10)
                    },
                    'electrode': 1
                }
            },
            {
                'type': 'umap',
                'data': {
                    'embedding': np.random.random((10, 3)),
                    'conditions': np.array(['train_1_pre'] * 10)
                }
            },
            {
                'type': 'pca',
                'data': {
                    'embedding': np.random.random((10, 3)),
                    'conditions': np.array(['train_1_pre'] * 10)
                }
            },
            {
                'type': 'raster',
                'data': {
                    'times': np.array([1, 2, 3]),
                    'channels': np.array([1, 2, 3]),
                    'phase': 'all'
                }
            },
            {
                'type': 'unknown',
                'data': {}
            }
        ]

        # Mock
        with patch('src.protocols.stdp.Plotter') as mock_plotter:
            mock_plotter_instance = mock_plotter.return_value
            protocol.visualise(results)

            # Check
            assert mock_plotter.call_count > 0
            assert mock_plotter_instance.plot.call_count > 0

    def test_run_integration(self, sample_config, sample_paths, ensure_test_dirs):
        """Test run method integration"""

        protocol = STDPProtocol(sample_config, sample_paths)
        test_csv = Path(sample_paths['raw']) / 'test.csv'
        df = pd.DataFrame({
            'Time': ['2023-01-01 12:00:00', '2023-01-01 12:00:01'],
            'channel': [1, 2],
            'Amplitude': [-5.0, -6.0]
        })
        df.to_csv(test_csv, index=False)

        # Mock
        with patch.object(protocol, '_analyse_channel_metrics'), \
             patch.object(protocol, '_analyse_channel_pair_metrics'), \
             patch.object(protocol, '_analyse_embeddings'), \
             patch.object(protocol, '_analyse_prcs'), \
             patch('src.protocols.stdp.compute_fft', return_value={'freqs': np.array([]), 'amplitudes': np.array([])}), \
             patch('src.protocols.stdp.compute_isi', return_value={'isi': np.array([]), 'mean_isi': 0, 'std_isi': 0}), \
             patch('src.protocols.stdp.compute_avalanche_distribution', return_value=np.array([])), \
             patch('src.protocols.stdp.compute_powerlaw', return_value={}), \
             patch('src.protocols.stdp.compute_firing_rates', return_value={}), \
             patch('src.protocols.stdp.compute_power_spectral_density', return_value={'freqs': np.array([]), 'psd': np.array([])}), \
             patch('src.protocols.stdp.compute_auto_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}), \
             patch('src.protocols.stdp.compute_granger_causality', return_value={}), \
             patch('src.protocols.stdp.compute_statistical_tests', return_value={}), \
             patch('src.protocols.stdp.compute_cross_correlation', return_value={'lags': np.array([]), 'correlation': np.array([])}), \
             patch('src.protocols.stdp.compute_umap_embedding', return_value={}), \
             patch('src.protocols.stdp.compute_pca_embedding', return_value={}), \
             patch('src.protocols.stdp.compute_prc', return_value={}), \
             patch('src.protocols.stdp.group_conditions_by_pre_post', return_value={'pre': df, 'post': df}), \
             patch('src.protocols.stdp.mask_stim_windows', return_value=df):

            with patch.object(protocol, 'visualise'):
                # Call
                results = protocol.run()

                # Check
                assert isinstance(results, list)
