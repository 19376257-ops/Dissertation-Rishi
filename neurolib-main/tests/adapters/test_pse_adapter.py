import json

import pandas as pd
import pytest

from src.adapters.pse_adapter import PSEAdapterError, adapt_pse_experiment
from src.data_manager import DataManager
from src.protocols.pse import PSEProtocol


def test_adapt_pse_experiment_writes_pipeline_inputs(tmp_path):
    spikes = pd.DataFrame(
        {
            "timestamp": [
                "2024-05-02T09:00:01Z",
                "2024-05-02T09:00:00Z",
            ],
            "electrode": [104, 103],
            "amplitude_uv": [-5.2, -4.8],
            "extra": ["ignored", "ignored"],
        }
    )
    metadata = {
        "phase_names": ["baseline", "train", "post_train"],
        "phase_timestamps": [
            "2024-05-02T09:00:00Z",
            "2024-05-02T09:05:00Z",
            "2024-05-02T09:10:00Z",
            "2024-05-02T09:15:00Z",
        ],
        "training_frequency_hz": 8,
        "stim_location": 104,
        "stim_pulse_timestamps": ["2024-05-02T09:05:01Z"],
        "mask_stim": True,
    }

    paths = adapt_pse_experiment(
        spike_source=spikes,
        metadata=metadata,
        output_root=tmp_path,
        iteration="Iteration1",
        date="2024-05-02",
        round_="Round1",
        column_map={
            "timestamp": "Time",
            "electrode": "channel",
            "amplitude_uv": "Amplitude",
        },
    )

    assert paths["spike_path"].exists()
    assert paths["experiment_params_path"].exists()

    saved_spikes = pd.read_csv(paths["spike_path"])
    assert list(saved_spikes.columns) == ["Time", "channel", "Amplitude"]
    assert saved_spikes["channel"].tolist() == [103, 104]
    assert saved_spikes["Time"].str.endswith("+00:00").sum() == 0

    saved_params = json.loads(paths["experiment_params_path"].read_text())
    assert saved_params["phase_names"] == ["baseline", "train", "post_train"]
    assert saved_params["phase_timestamps"][0] == "2024-05-02T09:00:00.000000"
    assert saved_params["stim_pulse_timestamps"] == ["2024-05-02T09:05:01.000000"]
    assert saved_params["protocol"] == "PSE"
    assert saved_params["time_reference"] == "UTC"


def test_adapt_pse_experiment_rejects_missing_spike_columns(tmp_path):
    spikes = pd.DataFrame({"Time": ["2024-05-02T09:00:00Z"], "channel": [1]})
    metadata = {
        "phase_names": ["baseline"],
        "phase_timestamps": ["2024-05-02T09:00:00Z", "2024-05-02T09:01:00Z"],
    }

    with pytest.raises(PSEAdapterError, match="Amplitude"):
        adapt_pse_experiment(spikes, metadata, tmp_path, "Iteration1", "2024-05-02", "Round1")


def test_adapt_pse_experiment_rejects_bad_phase_contract(tmp_path):
    spikes = pd.DataFrame(
        {
            "Time": ["2024-05-02T09:00:00Z"],
            "channel": [1],
            "Amplitude": [-1.0],
        }
    )
    metadata = {
        "phase_names": ["baseline", "train"],
        "phase_timestamps": ["2024-05-02T09:00:00Z", "2024-05-02T09:01:00Z"],
    }

    with pytest.raises(PSEAdapterError, match="interval boundaries"):
        adapt_pse_experiment(spikes, metadata, tmp_path, "Iteration1", "2024-05-02", "Round1")


def test_adapter_output_loads_and_preprocesses_with_pse_protocol(tmp_path):
    spikes = pd.DataFrame(
        {
            "Time": [
                "2024-05-02T09:00:10Z",
                "2024-05-02T09:05:10Z",
                "2024-05-02T09:10:10Z",
            ],
            "channel": [101, 101, 102],
            "Amplitude": [-1.0, -2.0, -3.0],
        }
    )
    metadata = {
        "phase_names": ["baseline", "train", "post_train"],
        "phase_timestamps": [
            "2024-05-02T09:00:00Z",
            "2024-05-02T09:05:00Z",
            "2024-05-02T09:10:00Z",
            "2024-05-02T09:15:00Z",
        ],
        "training_frequency_hz": 8,
        "stim_location": 101,
    }

    adapt_pse_experiment(
        spike_source=spikes,
        metadata=metadata,
        output_root=tmp_path,
        iteration="Iteration1",
        date="2024-05-02",
        round_="Round1",
    )

    dm = DataManager(tmp_path)
    paths = dm.get_paths("Iteration1", "PSE", "2024-05-02", "Round1")
    cfg = {
        "experiment_params": dm.load_experiment_params(paths["raw"]),
        "artefact_thresh": 8.0,
    }
    protocol = PSEProtocol(cfg, paths)

    loaded = protocol.load()
    preprocessed = protocol.preprocess(loaded)

    assert list(loaded.columns) == ["Time", "channel", "Amplitude"]
    assert sorted(preprocessed["condition"].unique().tolist()) == ["baseline", "post_train", "train"]
    assert cfg["experiment_params"]["training_frequency_hz"] == 8
    assert cfg["experiment_params"]["stim_location"] == 101
