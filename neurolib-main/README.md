
# Neuronal Firing-Rate Analysis Framework

## Overview

This repository provides a comprehensive, data-centric framework for analysing neuronal firing-rate experiments under various protocols:

- **Inhomogeneous Poisson Process (IPP)**: For analysing stochastic neuronal firing patterns
- **Spike-Timing-Dependent Plasticity (STDP)**: For studying synaptic strength changes based on timing
- **Phase-Shift Encoding (PSE)**: For investigating phase-based information encoding
- **Baseline**: For standard neuronal activity analysis without specific stimulation

The framework is designed with modularity and extensibility in mind, making it easy to add new analysis methods or protocols.

## Key Features

1. **Standardised Data Management**: Consistent folder structure for raw data and results
2. **Unified Pipeline Architecture**: Common workflow across all protocols
3. **Modular Analysis Tools**: Plug-and-play analysis modules for different metrics
4. **Comprehensive Visualisation**: Automatic plotting of analysis results
5. **Configurable Parameters**: YAML-based configuration for experiment settings
6. **Command-Line Interface**: Easy execution of analysis pipelines

## Project Structure

```
project-root/ 
├── data/ 
│   └── {iteration}/ 
│       └── {protocol}/ 
│           └── {YYYY-MM-DD}/ 
│               └── {round}/ 
│                   ├── raw/                  # Raw CSV data files
│                   ├── experiment_params.json # Experiment parameters
│                   └── results/              # Generated analysis outputs
├── configs/ 
│   ├── defaults.yaml                # Global default settings
│   └── protocols/                   # Protocol-specific configurations
│       ├── baseline.yaml
│       ├── ipp.yaml  
│       ├── stdp.yaml 
│       └── pse.yaml 
├── src/ 
│   ├── analysis/                    # Analysis modules
│   │   ├── avalanche.py             # Neuronal avalanche analysis
│   │   ├── correlation.py           # Auto and cross-correlation
│   │   ├── fft.py                   # Fourier transform analysis
│   │   ├── firing_rate.py           # Firing rate calculations
│   │   ├── granger_causality.py     # Causality analysis
│   │   ├── ipp_stats.py             # IPP-specific statistics
│   │   ├── isi.py                   # Inter-spike interval analysis
│   │   ├── pca_embed.py             # PCA dimensionality reduction
│   │   ├── powerlaw.py              # Power law distribution analysis
│   │   ├── prc.py                   # Phase response curve analysis
│   │   ├── psd.py                   # Power spectral density
│   │   ├── statistical_tools.py     # Statistical analysis utilities
│   │   ├── umap_embed.py            # UMAP dimensionality reduction
│   │   └── utils.py                 # Utility functions
│   ├── protocols/                   # Protocol implementations
│   │   ├── base.py                  # Abstract base protocol
│   │   ├── baseline.py              # Baseline protocol
│   │   ├── ipp.py                   # IPP protocol
│   │   ├── stdp.py                  # STDP protocol
│   │   └── pse.py                   # PSE protocol
│   ├── visualisation/               # Visualisation tools
│   │   └── plotters.py              # Central plotting dispatcher
│   ├── adapters/                    # Raw export to pipeline input adapters
│   ├── data_manager.py              # Data path resolution
│   └── cli.py                       # Command-line interface
├── examples/                        # Adapter templates and usage examples
├── stim_notebooks/                  # Jupyter notebooks for exploration
├── tests/                           # Test suite
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Core Architecture

### 1. Data Management

The `DataManager` class provides consistent path resolution for all protocols:

```python
dm = DataManager(Path("data"))
paths = dm.get_paths(iteration, protocol, date, round_)
# Returns: {'raw': Path('data/iteration/protocol/date/round/raw'),
#           'results': Path('data/iteration/protocol/date/round/results')}
```

This ensures that all protocols follow the same folder structure, making it easy to locate and manage data.

### 2. Protocol Base Class

All protocols inherit from `ProtocolBase`, which defines the standard pipeline:

```python
def run(self, sample_mode=False):
    # 1) LOAD raw data
    data = self.load()
    
    # 2) PREPROCESS data
    preproc = self.preprocess(data)
    
    # 3) ANALYSE preprocessed data
    results = self.analyse(preproc, sample_mode=sample_mode)
    
    # 4) Generate RASTER plot data
    raster = self._generate_raster_data(preproc)
    if raster:
        results.append(raster)
    
    # 5) VISUALISE results
    self.visualise(results)
    
    return results
```

This standardised workflow ensures consistency across all protocols while allowing for protocol-specific implementations of each stage.

### 3. Analysis Modules

Each analysis is implemented as a standalone function that takes input data and returns a dictionary with results:

```python
def compute_fft(df, channel, resample_interval, freq_cap):
    # Process data...
    return {'freqs': frequencies, 'amplitudes': amplitudes}
```

This modular approach makes it easy to add new analysis methods or modify existing ones without affecting the rest of the codebase.

### 4. Visualisation System

The `Plotter` class provides a unified interface for visualising all types of results:

```python
plotter = Plotter(results_dir)
for result in results:
    plotter.plot(result['type'], result['data'])
```

The plotter uses dynamic dispatch to route each result to the appropriate plotting method based on its type:

```python
def plot(self, result_type, data):
    fn = getattr(self, f'_plot_{result_type}', None)
    if fn is None:
        raise ValueError(f"No plotter for type {result_type}")
    fn(data)
```

This makes it easy to add new visualisations by simply adding a new `_plot_<type>` method to the `Plotter` class.

## Protocol-Specific Use Cases

### Baseline Protocol

The Baseline protocol is used for analysing standard neuronal activity without specific stimulation. It provides:

- Basic firing rate analysis
- Inter-spike interval (ISI) distributions
- Fourier transform analysis
- Power spectral density
- Avalanche size distributions
- Cross-correlation between channels
- Auto-correlation for individual channels
- Granger causality analysis

Use case: Establishing baseline neuronal activity patterns before applying stimulation protocols.

### IPP (Inhomogeneous Poisson Process) Protocol

The IPP protocol is designed for analysing stochastic neuronal firing patterns. It extends the baseline analysis with:

- Firing rate z-scores
- Kernel density estimation
- Inferential statistics
- IPP-specific statistical tests

Use case: Studying how neurons respond to random stimulation patterns and how this affects network dynamics.

### STDP (Spike-Timing-Dependent Plasticity) Protocol

The STDP protocol focuses on analysing how the timing of pre- and post-synaptic spikes affects synaptic strength. It includes:

- Phase response curve (PRC) analysis
- Channel pair metrics
- UMAP and PCA embeddings for dimensionality reduction
- Stimulus artifact removal
- Pre/post condition grouping

Use case: Investigating synaptic plasticity mechanisms and how they contribute to learning and memory formation.

### PSE (Phase-Shift Encoding) Protocol

The PSE protocol is used for studying phase-based information encoding in neuronal networks. It provides:

- Phase-specific analysis across different experimental phases
- Embedding visualisations for different phases
- Comparative analysis between phases

Use case: Understanding how phase shifts in neuronal activity encode information and how this encoding changes under different conditions.

## Running the Framework

### Command-Line Interface

The framework can be run from the command line using the following syntax:

```bash
python -m src.cli --iteration <iteration> --protocol <protocol> --date <YYYY-MM-DD> --round <round> [--config <config_path>] [--sample]
```

Arguments:
- `--iteration`: Iteration folder name (e.g., "6")
- `--protocol`: Protocol to run (BASELINE, IPP, STDP, or PSE)
- `--date`: Date of experiment (YYYY-MM-DD format)
- `--round`: Round identifier (e.g., "1")
- `--config`: Path to the global defaults YAML file (default: "configs/defaults.yaml")
- `--sample`: Run in sample mode with reduced channel set for quicker testing

Example:
```bash
python -m src.cli --iteration 6 --protocol STDP --date 2024-11-14 --round 2
```

### PSE Adapter

The PSE adapter converts notebook/database spike exports into the pipeline input layout.

It writes:

```text
data/{iteration}/PSE/{date}/{round}/raw/spikes.csv
data/{iteration}/PSE/{date}/{round}/experiment_params.json
```

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

The adapter expects spike data to map to `Time`, `channel`, and `Amplitude`. PSE phase timestamps are interval boundaries, so `phase_timestamps` must contain one more value than `phase_names`.

### Configuration

The framework uses YAML files for configuration:

1. Global defaults in `configs/defaults.yaml`
2. Protocol-specific settings in `configs/protocols/<protocol>.yaml`

These configurations are merged at runtime, with protocol-specific settings taking precedence.

## Extending the Framework

### Adding a New Analysis Module

1. Create a new file in the `src/analysis/` directory
2. Implement your analysis function that takes input data and returns a dictionary with results
3. Import and use your function in the appropriate protocol's `analyse()` method
4. Add a corresponding `_plot_<type>()` method to the `Plotter` class

### Adding a New Protocol

1. Create a new file in the `src/protocols/` directory
2. Subclass `ProtocolBase` and implement the required methods:
   - `load()`: Load raw data
   - `preprocess()`: Clean and prepare data
   - `analyse()`: Run analysis modules
   - `visualise()`: Visualise results
3. Create a protocol-specific configuration file in `configs/protocols/`
4. Add your protocol to the CLI options in `src.cli`

## Jupyter Notebooks

The `stim_notebooks/` directory contains Jupyter notebooks for exploring and demonstrating different aspects of the framework:

- `baseline.ipynb`: Basic neuronal activity analysis
- `(In)homogenous Poisson Process.ipynb`: IPP protocol demonstration
- `LOW FR (In)homogenous Poisson Process.ipynb`: IPP with low firing rates
- `stdp.ipynb`: STDP protocol demonstration
- `phase-shift-encoder.ipynb`: PSE protocol demonstration

These notebooks provide practical examples of how to use the framework for different types of neuronal data analysis.
