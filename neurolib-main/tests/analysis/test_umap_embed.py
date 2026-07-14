import pytest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import patch, MagicMock

from src.analysis.umap_embed import compute_umap_embedding, DEFAULT_PCA_COMPONENTS

def test_compute_umap_embedding_basic(sample_spike_data):
    """Test compute_umap_embedding with basic valid data"""

    # Mock
    with patch('src.analysis.umap_embed.StandardScaler') as mock_scaler, \
         patch('src.analysis.umap_embed.PCA') as mock_pca, \
         patch('src.analysis.umap_embed.umap.UMAP') as mock_umap, \
         patch('src.analysis.umap_embed.assign_conditions_to_bins') as mock_assign:

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.rand(10, 5)
        mock_scaler.return_value = mock_scaler_instance
        mock_pca_instance = MagicMock()
        pca_n_components = 5
        mock_pca_instance.fit_transform.return_value = np.random.rand(10, pca_n_components)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.1] * pca_n_components)
        mock_pca.return_value = mock_pca_instance
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.random.rand(10, 3)
        mock_umap.return_value = mock_umap_instance
        mock_assign.return_value = np.array(['condition1', 'condition2'] * 5)

        result = compute_umap_embedding(
            df=sample_spike_data,
            n_components=3,
            bin_size_ms=10,
            random_state=42
        )

        # Check
        assert 'embedding' in result
        assert 'n_components' in result
        assert 'pca_components' in result
        assert 'pca_explained_variance_ratio' in result
        assert 'bin_size_ms' in result
        assert 'binned_data' in result
        assert 'conditions' in result
        assert result['n_components'] == 3
        assert result['pca_components'] == 5
        assert result['bin_size_ms'] == 10
        assert len(result['pca_explained_variance_ratio']) == result['pca_components']
        assert result['embedding'].shape[1] == 3

def test_compute_umap_embedding_sample_mode(sample_spike_data):
    """Test compute_umap_embedding with sample mode enabled"""

    # Mock
    with patch('src.analysis.umap_embed.StandardScaler') as mock_scaler, \
         patch('src.analysis.umap_embed.PCA') as mock_pca, \
         patch('src.analysis.umap_embed.umap.UMAP') as mock_umap, \
         patch('src.analysis.umap_embed.assign_conditions_to_bins') as mock_assign:

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.rand(10, 5)
        mock_scaler.return_value = mock_scaler_instance
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(10, DEFAULT_PCA_COMPONENTS)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.1] * DEFAULT_PCA_COMPONENTS)
        mock_pca.return_value = mock_pca_instance
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.random.rand(10, 3)
        mock_umap.return_value = mock_umap_instance
        mock_assign.return_value = np.array(['condition1', 'condition2'] * 5)

        result = compute_umap_embedding(
            df=sample_spike_data,
            n_components=3,
            bin_size_ms=10,
            random_state=42,
            sample_mode=True
        )

        # Check
        assert 'embedding' in result
        assert 'n_components' in result
        assert 'pca_components' in result
        assert 'pca_explained_variance_ratio' in result
        assert 'bin_size_ms' in result
        assert 'binned_data' in result
        assert 'conditions' in result

        mock_assign.assert_called_once()

def test_compute_umap_embedding_exception_handling(sample_spike_data):
    """Test compute_umap_embedding with an exception during compute"""

    # Mock
    with patch('src.analysis.umap_embed.StandardScaler') as mock_scaler, \
         patch('src.analysis.umap_embed.PCA') as mock_pca, \
         patch('src.analysis.umap_embed.umap.UMAP') as mock_umap, \
         patch('src.analysis.umap_embed.assign_conditions_to_bins') as mock_assign:

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.rand(10, 5)
        mock_scaler.return_value = mock_scaler_instance
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(10, DEFAULT_PCA_COMPONENTS)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.1] * DEFAULT_PCA_COMPONENTS)
        mock_pca.return_value = mock_pca_instance
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.side_effect = ValueError("UMAP error")
        mock_umap.return_value = mock_umap_instance
        mock_assign.return_value = np.array(['condition1', 'condition2'] * 5)

        result = compute_umap_embedding(
            df=sample_spike_data,
            n_components=3,
            bin_size_ms=10,
            random_state=42
        )

        # Check
        assert 'error' in result
        assert 'n_components' in result
        assert 'pca_components' in result
        assert 'pca_explained_variance_ratio' in result
        assert 'bin_size_ms' in result
        assert 'binned_data' in result
        assert 'conditions' in result
        assert "UMAP error" in result['error']

def test_compute_umap_embedding_empty_data():
    """Test compute_umap_embedding with empty DataFrame"""

    empty_df = pd.DataFrame(columns=['Time', 'channel', 'Amplitude'])

    # Mock
    with patch('src.analysis.umap_embed.StandardScaler') as mock_scaler, \
         patch('src.analysis.umap_embed.PCA') as mock_pca, \
         patch('src.analysis.umap_embed.umap.UMAP') as mock_umap, \
         patch('src.analysis.umap_embed.assign_conditions_to_bins') as mock_assign:
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.side_effect = ValueError("Empty data")
        mock_scaler.return_value = mock_scaler_instance
        mock_assign.return_value = np.array([])

        result = compute_umap_embedding(
            df=empty_df,
            n_components=3,
            bin_size_ms=10,
            random_state=42
        )

        # Check
        assert 'error' in result
        assert "Empty data" in result['error']

def test_compute_umap_embedding_pca_components_adjustment(sample_spike_data):
    """Test compute_umap_embedding with PCA components adjustment"""

    # Mock
    with patch('src.analysis.umap_embed.StandardScaler') as mock_scaler, \
         patch('src.analysis.umap_embed.PCA') as mock_pca, \
         patch('src.analysis.umap_embed.umap.UMAP') as mock_umap, \
         patch('src.analysis.umap_embed.assign_conditions_to_bins') as mock_assign:

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.rand(10, 3)
        mock_scaler.return_value = mock_scaler_instance
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(10, 3)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])
        mock_pca.return_value = mock_pca_instance
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.random.rand(10, 3)
        mock_umap.return_value = mock_umap_instance
        mock_assign.return_value = np.array(['condition1', 'condition2'] * 5)

        result = compute_umap_embedding(
            df=sample_spike_data,
            n_components=3,
            bin_size_ms=10,
            random_state=42
        )

        # Check
        assert 'embedding' in result
        assert 'pca_components' in result
        assert result['pca_components'] == 3
        mock_pca.assert_called_once()
        args, kwargs = mock_pca.call_args
        assert kwargs['n_components'] == 3

def test_compute_umap_embedding_time_conversion():
    """Test compute_umap_embedding with non-datetime Time column"""

    df = pd.DataFrame({
        'Time': ['2020-01-01 12:00:00', '2020-01-01 12:00:01'],
        'channel': [1, 2],
        'Amplitude': [-5.0, -6.0]
    })

    # Mock
    with patch('src.analysis.umap_embed.StandardScaler') as mock_scaler, \
         patch('src.analysis.umap_embed.PCA') as mock_pca, \
         patch('src.analysis.umap_embed.umap.UMAP') as mock_umap, \
         patch('src.analysis.umap_embed.assign_conditions_to_bins') as mock_assign:

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.rand(10, 5)
        mock_scaler.return_value = mock_scaler_instance
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(10, DEFAULT_PCA_COMPONENTS)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.1] * DEFAULT_PCA_COMPONENTS)
        mock_pca.return_value = mock_pca_instance
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.random.rand(10, 3)
        mock_umap.return_value = mock_umap_instance
        mock_assign.return_value = np.array(['condition1', 'condition2'] * 5)

        result = compute_umap_embedding(
            df=df,
            n_components=3,
            bin_size_ms=10,
            random_state=42
        )

        # Check
        assert 'embedding' in result
        assert 'n_components' in result
        assert 'pca_components' in result

        mock_assign.assert_called_once()

def test_compute_umap_embedding_umap_parameters(sample_spike_data):
    """Test compute_umap_embedding with specific UMAP parameters"""

    # Mock
    with patch('src.analysis.umap_embed.StandardScaler') as mock_scaler, \
         patch('src.analysis.umap_embed.PCA') as mock_pca, \
         patch('src.analysis.umap_embed.umap.UMAP') as mock_umap, \
         patch('src.analysis.umap_embed.assign_conditions_to_bins') as mock_assign:

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.rand(10, 5)
        mock_scaler.return_value = mock_scaler_instance
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(10, DEFAULT_PCA_COMPONENTS)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.1] * DEFAULT_PCA_COMPONENTS)
        mock_pca.return_value = mock_pca_instance
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.random.rand(10, 3)
        mock_umap.return_value = mock_umap_instance
        mock_assign.return_value = np.array(['condition1', 'condition2'] * 5)

        result = compute_umap_embedding(
            df=sample_spike_data,
            n_components=3,
            bin_size_ms=10,
            random_state=42
        )

        mock_umap.assert_called_once()
        args, kwargs = mock_umap.call_args
        assert kwargs['n_components'] == 3
        assert kwargs['random_state'] == 42
        assert kwargs['n_epochs'] == 50
        assert kwargs['metric'] == 'euclidean'
        assert kwargs['low_memory'] == True
