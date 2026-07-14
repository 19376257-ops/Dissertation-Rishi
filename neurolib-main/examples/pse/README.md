# PSE Adapter Example

Use these templates to convert a raw notebook or database export into the folder structure expected by `PSEProtocol`.

The raw spike export must provide three fields after mapping:

```text
Time, channel, Amplitude
```

If the export uses different names, edit `column_map_template.json`. The keys are the raw export column names, and the values are the pipeline column names.

Edit `metadata_template.json` with the actual experiment timing and stimulation setup:

- `phase_names`: labels for each interval.
- `phase_timestamps`: interval boundaries. This list must have one more item than `phase_names`.
- `training_frequency_hz`: stimulation/training frequency.
- `stim_location`: stimulation electrode or mapped MEA channel.
- `stim_pulse_timestamps`: optional exact pulse times from trigger logs.
- `time_reference`: use `UTC` unless the pipeline is explicitly changed.

Run:

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

Then run the PSE analysis:

```bash
python -m src.cli \
  --iteration Iteration1 \
  --protocol PSE \
  --date 2024-05-02 \
  --round Round1 \
  --sample
```
