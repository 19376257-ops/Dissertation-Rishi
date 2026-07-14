import pytest
import numpy as np
from src.analysis.avalanche import compute_avalanche_distribution

def test_compute_avalanche_distribution_basic():
    """Test compute_avalanche_distribution with a basic example."""

    isi = np.array([0.1, 0.2, 0.3, 1.0, 0.1, 0.2, 0.5])
    result = compute_avalanche_distribution(isi, 0.5)

    # Check
    assert isinstance(result, np.ndarray)
    assert len(result) == 2
    assert result[0] == 3 # First avalanche: 0.1, 0.2, 0.3
    assert result[1] == 2 # Second avalanche: 0.1, 0.2

def test_compute_avalanche_distribution_empty():
    """Test compute_avalanche_distribution with an empty array."""

    isi = np.array([])
    result = compute_avalanche_distribution(isi, 0.5)
    
    # Check
    assert isinstance(result, np.ndarray)
    assert len(result) == 0

def test_compute_avalanche_distribution_no_avalanches():
    """Test compute_avalanche_distribution with no avalanches."""

    isi = np.array([1.0, 2.0, 3.0, 4.0])
    result = compute_avalanche_distribution(isi, 0.5)
    
    # Check
    assert isinstance(result, np.ndarray)
    assert len(result) == 0

def test_compute_avalanche_distribution_all_avalanches():
    """Test compute_avalanche_distribution with all intervals below threshold."""

    isi = np.array([0.1, 0.2, 0.3, 0.4])
    result = compute_avalanche_distribution(isi, 0.5)
    
    # Check
    assert isinstance(result, np.ndarray)
    assert len(result) == 1
    assert result[0] == 4 # One avalanche with all 4 intervals

def test_compute_avalanche_distribution_edge_case():
    """Test compute_avalanche_distribution with intervals at the threshold."""

    isi = np.array([0.5, 0.5, 0.5, 0.5])
    result = compute_avalanche_distribution(isi, 0.5)

    assert isinstance(result, np.ndarray)
    assert len(result) == 0

    result = compute_avalanche_distribution(isi, 0.51)
    
    # Check
    assert isinstance(result, np.ndarray)
    assert len(result) == 1
    assert result[0] == 4