import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.protocols.base import ProtocolBase

class TestProtocolBase:
    """Tests for ProtocolBase class."""

    class ConcreteProtocol(ProtocolBase):
        def preprocess(self, data):
            """Implement abstract method."""
            return data

        def analyse(self, preproc, sample_mode=False):
            """Implement abstract method."""
            return []

        def visualise(self, results):
            """Implement abstract method."""
            pass

    def test_init(self, sample_config, sample_paths):
        """Test init of ProtocolBase."""
        protocol = self.ConcreteProtocol(sample_config, sample_paths)

        assert protocol.cfg == sample_config
        assert protocol.raw_dir == Path(sample_paths['raw'])
        assert protocol.results_dir == Path(sample_paths['results'])

    def test_ensure_results_dir(self, sample_config, sample_paths, ensure_test_dirs):
        """Test _ensure_results_dir method."""
        protocol = self.ConcreteProtocol(sample_config, sample_paths)

        assert protocol.results_dir.exists()

    def test_run(self, sample_config, sample_paths, ensure_test_dirs):
        """Test run method."""
        protocol = self.ConcreteProtocol(sample_config, sample_paths)

        # Mock
        protocol.load = MagicMock(return_value=pd.DataFrame())
        protocol.preprocess = MagicMock(return_value=pd.DataFrame())
        protocol.analyse = MagicMock(return_value=[])
        protocol._generate_raster_data = MagicMock(return_value=None)
        protocol.visualise = MagicMock()

        results = protocol.run()

        # Check
        protocol.load.assert_called_once()
        protocol.preprocess.assert_called_once()
        protocol.analyse.assert_called_once()
        protocol._generate_raster_data.assert_called_once()
        protocol.visualise.assert_called_once()
        assert results == []

    def test_run_with_raster(self, sample_config, sample_paths, ensure_test_dirs):
        """Test run method with raster data."""
        protocol = self.ConcreteProtocol(sample_config, sample_paths)

        # Mock
        protocol.load = MagicMock(return_value=pd.DataFrame())
        protocol.preprocess = MagicMock(return_value=pd.DataFrame())
        protocol.analyse = MagicMock(return_value=[])
        raster_result = {'type': 'raster', 'data': {}}
        protocol._generate_raster_data = MagicMock(return_value=raster_result)

        protocol.visualise = MagicMock()

        # Call
        results = protocol.run()

        # Check
        protocol.load.assert_called_once()
        protocol.preprocess.assert_called_once()
        protocol.analyse.assert_called_once()
        protocol._generate_raster_data.assert_called_once()
        protocol.visualise.assert_called_once()
        assert results == [raster_result]

    def test_load(self, sample_config, sample_paths, ensure_test_dirs):
        """Test load method."""
        protocol = self.ConcreteProtocol(sample_config, sample_paths)

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

    def test_load_no_files(self, sample_config, ensure_test_dirs):
        """Test load method with no files."""

        empty_dir = Path('tests/data/empty')
        empty_dir.mkdir(parents=True, exist_ok=True)

        empty_paths = {
            'raw': str(empty_dir),
            'results': 'tests/data/results'
        }

        protocol = self.ConcreteProtocol(sample_config, empty_paths)

        # Call
        with pytest.raises(FileNotFoundError):
            protocol.load()

    def test_generate_raster_data(self, sample_spike_data):
        """Test _generate_raster_data method"""

        protocol = self.ConcreteProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        # Call
        result = protocol._generate_raster_data(sample_spike_data)

        # Check
        assert result['type'] == 'raster'
        assert 'data' in result
        assert 'times' in result['data']
        assert 'channels' in result['data']
        assert 'phase' in result['data']
        assert result['data']['phase'] == 'all'

    def test_generate_raster_data_with_phase(self, sample_spike_data):
        """Test _generate_raster_data method with phase"""

        protocol = self.ConcreteProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})
        df = sample_spike_data.copy()
        df['condition'] = 'test_phase'

        # Call
        result = protocol._generate_raster_data(df, phase='test_phase')

        # Check
        assert result['type'] == 'raster'
        assert 'data' in result
        assert 'times' in result['data']
        assert 'channels' in result['data']
        assert 'phase' in result['data']
        assert result['data']['phase'] == 'test_phase'

    def test_generate_raster_data_with_threshold(self, sample_spike_data):
        """Test _generate_raster_data method with threshold"""

        protocol = self.ConcreteProtocol({'raster_threshold': -5.0}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        # Call
        result = protocol._generate_raster_data(sample_spike_data)

        # Check
        assert result['type'] == 'raster'
        assert 'data' in result
        assert 'times' in result['data']
        assert 'channels' in result['data']
        assert all(sample_spike_data.loc[sample_spike_data['Amplitude'] < -5.0, 'Time'].isin(result['data']['times']))

    def test_generate_raster_data_empty(self):
        """Test _generate_raster_data method with empty DataFrame"""
        protocol = self.ConcreteProtocol({}, {'raw': 'tests/data/raw', 'results': 'tests/data/results'})

        # Call
        result = protocol._generate_raster_data(pd.DataFrame())

        # Check
        assert result is None
