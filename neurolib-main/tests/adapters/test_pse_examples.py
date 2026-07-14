import json

import pandas as pd

from src.adapters.pse_adapter import adapt_pse_experiment


def test_pse_example_templates_are_usable(tmp_path):
    metadata_path = "examples/pse/metadata_template.json"
    column_map_path = "examples/pse/column_map_template.json"

    spikes = pd.DataFrame(
        {
            "timestamp": [
                "2024-05-02T09:00:10Z",
                "2024-05-02T09:05:10Z",
                "2024-05-02T09:10:10Z",
            ],
            "electrode": [101, 101, 102],
            "amplitude_uv": [-1.0, -2.0, -3.0],
        }
    )
    raw_export_path = tmp_path / "raw_spike_export.csv"
    spikes.to_csv(raw_export_path, index=False)

    with open(column_map_path, "r") as f:
        column_map = json.load(f)

    paths = adapt_pse_experiment(
        spike_source=raw_export_path,
        metadata=metadata_path,
        output_root=tmp_path / "data",
        iteration="Iteration1",
        date="2024-05-02",
        round_="Round1",
        column_map=column_map,
    )

    saved_params = json.loads(paths["experiment_params_path"].read_text())
    saved_spikes = pd.read_csv(paths["spike_path"])

    assert paths["spike_path"].exists()
    assert paths["experiment_params_path"].exists()
    assert list(saved_spikes.columns) == ["Time", "channel", "Amplitude"]
    assert len(saved_params["phase_timestamps"]) == len(saved_params["phase_names"]) + 1
    assert saved_params["training_frequency_hz"] == 8
