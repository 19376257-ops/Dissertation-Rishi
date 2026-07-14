import pytest
import argparse
from unittest.mock import patch, MagicMock

from src.cli import CaseInsensitiveChoicesAction, main

class TestCaseInsensitiveChoicesAction:
    def test_init(self):
        """Test init of CaseInsensitiveChoicesAction"""
        choices = ["OPTION1", "OPTION2", "OPTION3"]
        action = CaseInsensitiveChoicesAction(
            option_strings=["--option"],
            dest="option",
            choices=choices
        )

        assert action.choices_lower == {
            "option1": "OPTION1",
            "option2": "OPTION2",
            "option3": "OPTION3"
        }

    def test_call_valid_choice(self):
        """Test _call_ with a valid choice"""
        choices = ["OPTION1", "OPTION2", "OPTION3"]
        action = CaseInsensitiveChoicesAction(
            option_strings=["--option"],
            dest="option",
            choices=choices
        )

        namespace = argparse.Namespace()
        parser = MagicMock()
        action(parser, namespace, "option2", "--option")

        assert namespace.option == "OPTION2"

    def test_call_invalid_choice(self):
        """Test _call_ with an invalid choice"""
        choices = ["OPTION1", "OPTION2", "OPTION3"]
        action = CaseInsensitiveChoicesAction(
            option_strings=["--option"],
            dest="option",
            choices=choices
        )
        namespace = argparse.Namespace()
        parser = MagicMock()
        parser.error = MagicMock(side_effect=SystemExit)

        try:
            action(parser, namespace, "invalid", "--option")
        except SystemExit:
            pass

        parser.error.assert_called_once()

class TestMain:
    @patch('argparse.ArgumentParser.parse_args')
    @patch('src.data_manager.DataManager')
    @patch('src.protocols.baseline.BaselineProtocol')
    def test_main_basic(self, mock_protocol, mock_dm, mock_parse_args):
        """Basic test for main function with minimal mocking."""
        # 1) Mock return values
        mock_args = MagicMock()
        mock_args.protocol = "BASELINE"
        mock_args.iteration = "5"
        mock_args.date = "2024-01-01"
        mock_args.round = "Round1"
        mock_args.config = "configs/defaults.yaml"
        mock_args.sample = False
        mock_parse_args.return_value = mock_args

        # 2) Mock DataManager
        mock_dm_instance = MagicMock()
        mock_dm_instance.get_paths.return_value = {"raw": "raw_path", "results": "results_path"}
        mock_dm_instance.load_experiment_params.return_value = {}
        mock_dm.return_value = mock_dm_instance

        # 3) Mock Protocol
        mock_protocol_instance = MagicMock()
        mock_protocol_instance.run.return_value = []
        mock_protocol.return_value = mock_protocol_instance

        # 4) Mock object
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file

        # 5) Call main
        with patch('builtins.open', return_value=mock_file):
            with patch('yaml.safe_load', return_value={"data_root": "data"}):
                main()

        # 5) Check
        mock_protocol.assert_called_once()
        mock_protocol_instance.run.assert_called_once_with(sample_mode=False)

    @patch('argparse.ArgumentParser.parse_args')
    @patch('src.cli.adapt_pse_experiment')
    def test_main_adapt_pse(self, mock_adapt, mock_parse_args):
        """Test CLI adapter command delegates to PSE adapter."""
        mock_args = MagicMock()
        mock_args.command = "adapt-pse"
        mock_args.spikes = "raw_export.csv"
        mock_args.metadata = "metadata.json"
        mock_args.output_root = "data"
        mock_args.iteration = "Iteration1"
        mock_args.date = "2024-05-02"
        mock_args.round = "Round1"
        mock_args.column_map = '{"timestamp": "Time", "electrode": "channel", "amplitude_uv": "Amplitude"}'
        mock_parse_args.return_value = mock_args

        mock_adapt.return_value = {
            "spike_path": "data/Iteration1/PSE/2024-05-02/Round1/raw/spikes.csv",
            "experiment_params_path": "data/Iteration1/PSE/2024-05-02/Round1/experiment_params.json",
        }

        main()

        mock_adapt.assert_called_once_with(
            spike_source="raw_export.csv",
            metadata="metadata.json",
            output_root="data",
            iteration="Iteration1",
            date="2024-05-02",
            round_="Round1",
            column_map={
                "timestamp": "Time",
                "electrode": "channel",
                "amplitude_uv": "Amplitude",
            },
        )
