# PSE Adapter Development Summary

## 1. Objective

The goal of this development work was to create an adapter for the Phase-Shift Encoding (PSE) workflow.

The adapter converts raw experiment outputs from notebooks or database exports into the structured input format expected by the existing neuronal analysis pipeline.

This is important because the current analysis pipeline expects a fixed folder structure, standardized spike data columns, and a valid `experiment_params.json` file. Raw experiment exports may not naturally match that format.

## 2. Background

The project analyzes neuronal spike/firing data using protocol-specific pipelines.

The PSE protocol is used to study how neural network behavior changes across experimental phases such as:

- baseline
- training
- post-training
- phase-shift or stimulation phases

The development discussion identified three important experiment requirements:

- compare network behavior after a single training session
- support different stimulation frequencies
- support different stimulation locations or electrodes

To support this, the project needs a reliable adapter layer between raw experiment collection and analysis.

## 3. What Was Built

A new PSE adapter module was added:

```text
src/adapters/pse_adapter.py
```

The adapter can take raw spike data from either:

- a CSV file
- a pandas DataFrame

It converts the spike data into the required pipeline columns:

```text
Time, channel, Amplitude
```

It then writes the output into the folder structure expected by the PSE analysis pipeline:

```text
data/{iteration}/PSE/{date}/{round}/raw/spikes.csv
data/{iteration}/PSE/{date}/{round}/experiment_params.json
data/{iteration}/PSE/{date}/{round}/results/
```

## 4. Metadata Support

The adapter also writes experiment metadata into `experiment_params.json`.

The metadata supports:

- phase names
- phase timestamps
- stimulation/training frequency
- stimulation location or electrode
- stimulation pulse timestamps
- stimulation masking settings
- timing reference

Example metadata fields:

```json
{
  "phase_names": ["baseline", "train", "post_train"],
  "phase_timestamps": [
    "2024-05-02T09:00:00Z",
    "2024-05-02T09:05:00Z",
    "2024-05-02T09:10:00Z",
    "2024-05-02T09:15:00Z"
  ],
  "training_frequency_hz": 8,
  "stim_location": 101,
  "stim_pulse_timestamps": [],
  "mask_stim": true,
  "mask_window_ms": 3.0,
  "time_reference": "UTC"
}
```

## 5. Timing Contract

A key part of this work was clarifying how timing should be represented.

The adapter and PSE protocol now use this contract:

```text
phase_timestamps = interval boundaries
phase_names = labels for the intervals between boundaries
```

Example:

```text
phase_timestamps:
09:00, 09:05, 09:10, 09:15

phase_names:
baseline, train, post_train
```

This creates the following phase intervals:

```text
09:00-09:05 = baseline
09:05-09:10 = train
09:10-09:15 = post_train
```

This is important because every spike is assigned to an experimental phase based on its timestamp. If timing is wrong, the downstream analysis may classify spikes into the wrong condition.

## 6. CLI Support

The adapter can now be run from the command line.

Example:

```bash
python -m src.cli adapt-pse \
  --spikes path/to/raw_spike_export.csv \
  --metadata examples/pse/metadata_template.json \
  --column-map examples/pse/column_map_template.json \
  --output-root data \
  --iteration Iteration1 \
  --date 2024-05-02 \
  --round Round1
```

After running the adapter, the PSE analysis can be run using:

```bash
python -m src.cli \
  --iteration Iteration1 \
  --protocol PSE \
  --date 2024-05-02 \
  --round Round1 \
  --sample
```

## 7. Example Templates

Example files were added to make usage easier:

```text
examples/pse/metadata_template.json
examples/pse/column_map_template.json
examples/pse/README.md
```

The column map template converts raw export column names into pipeline column names.

Example:

```json
{
  "timestamp": "Time",
  "electrode": "channel",
  "amplitude_uv": "Amplitude"
}
```

This allows the adapter to work with raw exports even when the original column names differ from the pipeline standard.

## 8. Additional Technical Improvements

UMAP loading was changed to lazy loading.

Previously, importing the CLI or PSE protocol could immediately import UMAP and numba, which caused environment-related errors during tests.

Now UMAP is imported only when a UMAP embedding is actually computed.

This makes CLI usage and adapter testing more reliable.

## 9. Testing Completed

Tests were added or updated for:

- adapter file generation
- spike column validation
- metadata validation
- timestamp normalization
- example template validity
- CLI adapter command behavior
- adapter output loading through `PSEProtocol`
- PSE phase boundary interpretation

Latest verified test result:

```text
22 passed, 6 warnings
```

The warnings are existing z-score warnings in protocol tests and are not caused by the adapter work.

## 10. Current Status

The PSE adapter is now:

- implemented
- accessible through CLI
- documented with examples
- tested
- compatible with the existing PSE preprocessing path

The project now has a working bridge from raw experiment exports to the PSE analysis pipeline.

## 11. Pending Work

The main pending tasks are:

- connect the adapter to real notebook or database exports
- confirm actual raw export column names
- fill real metadata from the PSE notebook
- decide the authoritative source for pulse timestamps
- decide whether timing should come from Python logs, trigger logs, or database records
- add analysis summaries grouped by stimulation frequency and location
- optionally add a combined command for adapter plus PSE analysis

## 12. Short Summary

We built a PSE adapter that converts raw experiment exports into the required analysis pipeline format.

It standardizes spike data, preserves timing information, records stimulation frequency and location, writes experiment metadata, and prepares the correct folder structure for PSE analysis.

The timing contract has also been clarified so that phase timestamps are treated as interval boundaries.

The adapter is now available through the CLI and is covered by tests.

