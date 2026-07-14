import pytest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import patch, MagicMock

from src.analysis.pca_embed import compute_pca_embedding

def test_compute_pca_embedding_basic(sample_spike_data):
    """Test compute_pca_embedding with basic valid data"""

    # Mock
    with patch('src.analysis.pca_embed.StandardScaler') as mock_scaler, \
         patch('src.analysis.pca_embed.PCA') as mock_pca, \
         patch('src.analysis.pca_embed.assign_conditions_to_bins') as mock_assign:

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.rand(10, 5)
        mock_scaler.return_value = mock_scaler_instance

        # Mock
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(10, 3)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])
        mock_pca.return_value = mock_pca_instance
        mock_assign.return_value = np.array(['condition1', 'condition2'] * 5)

        result = compute_pca_embedding(
            df=sample_spike_data,
            n_components=3,
            bin_size_ms=10,
            random_state=42
        )

        # Check
        assert 'embedding' in result
        assert 'n_components' in result
        assert 'explained_variance_ratio' in result
        assert 'bin_size_ms' in result
        assert 'binned_data' in result
        assert 'conditions' in result
        assert result['n_components'] == 3
        assert result['bin_size_ms'] == 10
        assert np.array_equal(result['explained_variance_ratio'], np.array([0.5, 0.3, 0.2]))
        assert result['embedding'].shape[1] == 3 # 3 components

def test_compute_pca_embedding_sample_mode(sample_spike_data):
    """Test compute_pca_embedding with sample mode enabled"""
    # Mock
    with patch('src.analysis.pca_embed.StandardScaler') as mock_scaler, \
         patch('src.analysis.pca_embed.PCA') as mock_pca, \
         patch('src.analysis.pca_embed.assign_conditions_to_bins') as mock_assign:

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.rand(10, 5)
        mock_scaler.return_value = mock_scaler_instance
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(10, 3)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])
        mock_pca.return_value = mock_pca_instance
        mock_assign.return_value = np.array(['condition1', 'condition2'] * 5)

        result = compute_pca_embedding(
            df=sample_spike_data,
            n_components=3,
            bin_size_ms=10,
            random_state=42,
            sample_mode=True
        )

        # Check
        assert 'embedding' in result
        assert 'n_components' in result
        assert 'explained_variance_ratio' in result
        assert 'bin_size_ms' in result
        assert 'binned_data' in result
        assert 'conditions' in result

def test_compute_pca_embedding_exception_handling(sample_spike_data):
    """Test compute_pca_embedding with an exception during computation"""

    # Mock
    with patch('src.analysis.pca_embed.StandardScaler') as mock_scaler, \
         patch('src.analysis.pca_embed.PCA') as mock_pca, \
         patch('src.analysis.pca_embed.assign_conditions_to_bins') as mock_assign:

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.rand(10, 5)
        mock_scaler.return_value = mock_scaler_instance
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.side_effect = ValueError("PCA error")
        mock_pca.return_value = mock_pca_instance
        mock_assign.return_value = np.array(['condition1', 'condition2'] * 5)

        result = compute_pca_embedding(
            df=sample_spike_data,
            n_components=3,
            bin_size_ms=10,
            random_state=42
        )

        # Check
        assert 'error' in result
        assert 'n_components' in result
        assert 'explained_variance_ratio' in result
        assert 'bin_size_ms' in result
        assert 'binned_data' in result
        assert 'conditions' in result
        assert "PCA error" in result['error']

def test_compute_pca_embedding_empty_data():
    """Test compute_pca_embedding with empty DataFrame."""

    empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])

    # Mock
    with patch('src.analysis.pca_embed.StandardScaler') as mock_scaler, \
         patch('src.analysis.pca_embed.PCA') as mock_pca, \
         patch('src.analysis.pca_embed.assign_conditions_to_bins') as mock_assign:

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.side_effect = ValueError("Empty data")
        mock_scaler.return_value = mock_scaler_instance
        mock_assign.return_value = np.array([])
        result = compute_pca_embedding(
            df=empty_df,
            n_components=3,
            bin_size_ms=10,
            random_state=42
        )

        # Check
        assert 'error' in result
        assert "Empty data" in result['error']

def test_compute_pca_embedding_fewer_components_than_channels(sample_spike_data):
    """Test compute_pca_embedding with fewer components than channels"""

    # Mock
    with patch('src.analysis.pca_embed.StandardScaler') as mock_scaler, \
         patch('src.analysis.pca_embed.PCA') as mock_pca, \
         patch('src.analysis.pca_embed.assign_conditions_to_bins') as mock_assign:

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.rand(10, 5)
        mock_scaler.return_value = mock_scaler_instance
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(10, 2)  # Only 2 components
        mock_pca_instance.explained_variance_ratio_ = np.array([0.6, 0.4])
        mock_pca.return_value = mock_pca_instance
        mock_assign.return_value = np.array(['condition1', 'condition2'] * 5)

        result = compute_pca_embedding(
            df=sample_spike_data,
            n_components=10, # More than the 5 channels in the mock data
            bin_size_ms=10,
            random_state=42
        )

        assert 'embedding' in result
        assert 'n_components' in result
        assert 'explained_variance_ratio' in result
        assert result['n_components'] <= 5 # Should be limited by the number of channels

def test_compute_pca_embedding_time_conversion():
    """Test compute_pca_embedding with nondatetime Time column"""

    df = pd.DataFrame({
        'Time': ['2020-01-01 12:00:00', '2020-01-01 12:00:01'],
        'channel': [1, 2],
        'Amplitude': [-5.0, -6.0]
    })

    # Mock
    with patch('src.analysis.pca_embed.StandardScaler') as mock_scaler, \
         patch('src.analysis.pca_embed.PCA') as mock_pca, \
         patch('src.analysis.pca_embed.assign_conditions_to_bins') as mock_assign:

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.rand(10, 2)
        mock_scaler.return_value = mock_scaler_instance
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(10, 2)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.6, 0.4])
        mock_pca.return_value = mock_pca_instance
        mock_assign.return_value = np.array(['condition1', 'condition2'] * 5)

        result = compute_pca_embedding(
            df=df,
            n_components=2,
            bin_size_ms=10,
            random_state=42
        )

        # Check
        assert 'embedding' in result
        assert 'n_components' in result
        assert 'explained_variance_ratio' in result
        mock_assign.assert_called_once()
