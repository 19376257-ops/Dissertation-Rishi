import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import pandas as pd

from src.data_manager import DataManager


REQUIRED_SPIKE_COLUMNS = ("Time", "channel", "Amplitude")
REQUIRED_METADATA_KEYS = ("phase_names", "phase_timestamps")


class PSEAdapterError(ValueError):
    """Raised when a PSE adapter input cannot be converted safely."""


def adapt_pse_experiment(
    spike_source: Union[str, Path, pd.DataFrame],
    metadata: Union[str, Path, Mapping[str, Any]],
    output_root: Union[str, Path],
    iteration: str,
    date: str,
    round_: str,
    column_map: Optional[Mapping[str, str]] = None,
    output_filename: str = "spikes.csv",
) -> dict:
    """
    Convert raw PSE experiment output into the folder/files expected by PSEProtocol.

    The generated structure is:
        output_root/iteration/PSE/date/round_/raw/spikes.csv
        output_root/iteration/PSE/date/round_/experiment_params.json

    Timestamps are normalized to UTC and written without timezone information
    because the current protocol code compares timezone-naive datetimes.
    """
    dm = DataManager(Path(output_root))
    paths = dm.get_paths(iteration, "PSE", date, round_)
    raw_dir = paths["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)

    spikes = _load_spikes(spike_source, column_map)
    params = _load_metadata(metadata)
    params = _normalise_metadata(params)

    spike_path = raw_dir / output_filename
    params_path = raw_dir.parent / "experiment_params.json"

    spikes.to_csv(spike_path, index=False)
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    return {
        "raw": raw_dir,
        "results": paths["results"],
        "spike_path": spike_path,
        "experiment_params_path": params_path,
    }


def _load_spikes(
    spike_source: Union[str, Path, pd.DataFrame],
    column_map: Optional[Mapping[str, str]],
) -> pd.DataFrame:
    if isinstance(spike_source, pd.DataFrame):
        df = spike_source.copy()
    else:
        df = pd.read_csv(Path(spike_source))

    if column_map:
        df = df.rename(columns=dict(column_map))

    missing = [col for col in REQUIRED_SPIKE_COLUMNS if col not in df.columns]
    if missing:
        raise PSEAdapterError(f"Missing required spike column(s): {', '.join(missing)}")

    df = df.loc[:, REQUIRED_SPIKE_COLUMNS].copy()
    df["Time"] = _normalise_datetime_series(df["Time"])
    df["channel"] = pd.to_numeric(df["channel"], errors="raise").astype(int)
    df["Amplitude"] = pd.to_numeric(df["Amplitude"], errors="raise")

    if df["Time"].isna().any():
        raise PSEAdapterError("Spike data contains invalid timestamp values")

    return df.sort_values(["Time", "channel"]).reset_index(drop=True)


def _load_metadata(metadata: Union[str, Path, Mapping[str, Any]]) -> dict:
    if isinstance(metadata, Mapping):
        return dict(metadata)

    with open(Path(metadata), "r") as f:
        return json.load(f)


def _normalise_metadata(metadata: Mapping[str, Any]) -> dict:
    params = dict(metadata)

    missing = [key for key in REQUIRED_METADATA_KEYS if key not in params]
    if missing:
        raise PSEAdapterError(f"Missing required metadata key(s): {', '.join(missing)}")

    phase_names = list(params["phase_names"])
    phase_timestamps = list(params["phase_timestamps"])
    if len(phase_timestamps) != len(phase_names) + 1:
        raise PSEAdapterError(
            "phase_timestamps must contain interval boundaries: "
            "len(phase_timestamps) == len(phase_names) + 1"
        )

    params["phase_names"] = phase_names
    params["phase_timestamps"] = _normalise_datetime_values(phase_timestamps)

    if "stim_pulse_timestamps" in params:
        params["stim_pulse_timestamps"] = _normalise_datetime_values(params["stim_pulse_timestamps"])

    params.setdefault("protocol", "PSE")
    params.setdefault("time_reference", "UTC")
    params.setdefault("mask_stim", False)

    return params


def _normalise_datetime_series(values) -> pd.Series:
    timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    return timestamps.dt.tz_convert(None).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")


def _normalise_datetime_values(values) -> list:
    timestamps = pd.to_datetime(list(values), utc=True, errors="raise")
    return [
        timestamp.tz_convert(None).strftime("%Y-%m-%dT%H:%M:%S.%f")
        for timestamp in timestamps
    ]
