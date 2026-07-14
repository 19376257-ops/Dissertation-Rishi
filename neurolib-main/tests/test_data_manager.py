import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from src.data_manager import DataManager

class TestDataManager:
    def test_init(self):
        """Test init of DataManager."""
        root_dir = Path("/test/root")
        dm = DataManager(root_dir)
        assert dm.root == root_dir

    def test_get_paths(self):
        """Test get_paths method."""
        root_dir = Path("/test/root")
        dm = DataManager(root_dir)
        
        iteration = "5"
        protocol = "IPP"
        date = "2024-01-01"
        round_ = "Round1"
        
        paths = dm.get_paths(iteration, protocol, date, round_)
        
        expected_base = root_dir / iteration / protocol / date / round_
        assert paths["raw"] == expected_base / "raw"
        assert paths["results"] == expected_base / "results"

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"param": "value"}')
    @patch('json.load')
    def test_load_experiment_params_local(self, mock_json_load, mock_file, mock_exists):
        """Test load_experiment_params when local file exists."""
        # 1) Mocks
        mock_exists.return_value = True
        mock_json_load.return_value = {"param": "value"}
        
        # 2) DataManager
        root_dir = Path("/test/root")
        dm = DataManager(root_dir)
        raw_dir = Path("/test/raw")
        
        # 3) Call method
        result = dm.load_experiment_params(raw_dir)
        
        # 4) Check
        expected_path = raw_dir.parent / "experiment_params.json"
        mock_file.assert_called_once_with(expected_path, "r")
        mock_json_load.assert_called_once()
        assert result == {"param": "value"}

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"param": "value"}')
    @patch('json.load')
    def test_load_experiment_params_global(self, mock_json_load, mock_file, mock_exists):
        """Test load_experiment_params when local file doesn't exist."""
        # 1) Mock
        mock_exists.return_value = False
        mock_json_load.return_value = {"param": "value"}
        
        # 2) DataManager
        root_dir = Path("/test/root")
        dm = DataManager(root_dir)
        raw_dir = Path("/test/raw")
        
        # 3) Call method
        result = dm.load_experiment_params(raw_dir)
        
        # 4) Check
        expected_path = root_dir / "experiment_params.json"
        mock_file.assert_called_once_with(expected_path, "r")
        mock_json_load.assert_called_once()
        assert result == {"param": "value"}