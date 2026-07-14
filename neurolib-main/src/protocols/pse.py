import pandas as pd
import numpy as np
from pathlib import Path
import logging

from src.protocols.base import ProtocolBase
from src.visualisation.plotters import Plotter

from src.analysis.fft import compute_fft
from src.analysis.isi import compute_isi
from src.analysis.powerlaw import compute_powerlaw
from src.analysis.granger_causality import compute_granger_causality
from src.analysis.statistical_tools import compute_statistical_tests
from src.analysis.avalanche import compute_avalanche_distribution
from src.analysis.firing_rate import compute_firing_rates
from src.analysis.umap_embed import compute_umap_embedding
from src.analysis.pca_embed import compute_pca_embedding
from src.analysis.psd import compute_power_spectral_density
from src.analysis.correlation import compute_auto_correlation, compute_cross_correlation
from src.analysis.ipp_stats import compute_firing_rate_zscore, compute_kernel_density
from src.analysis.utils import mask_stim_windows, map_channels_to_mea

logger = logging.getLogger(__name__)

class PSEProtocol(ProtocolBase):
    """
    Phase-Specific Encoding (PSE) protocol for analysing neuronal recordings with distinct experimental phases.

    1) Load raw neural recording data from CSV files
    2) Pre-process data with phase labeling and artifact removal:
       - Convert time to datetime format
       - Remove rows with missing values
       - Apply MAD-based noise reduction
       - Assign phase labels based on timestamps
       - Apply stimulus window masking for specific channels
    3) Analyse data for each phase separately:
       - Perform standard neural analyses (FFT, ISI, etc.)
       - Generate phase-specific embeddings (UMAP, PCA)
       - Compute statistical relationships between channels
    4) Visualise results with phase-specific plots and comparisons
    """

    def __init__(self, cfg, paths):
        super().__init__(cfg, paths)

    def load(self):
        """
        Load raw neuronal recording data from CSV files in the raw directory.

        1) Find all CSV files in the raw directory
        2) Validate that at least one CSV file exists
        3) Init an empty list to store DataFrames
        4) For each CSV file:
           a. Attempt to read the file with datetime parsing for the Time column
           b. Add the DataFrame to the collection
           c. Skip files with missing required columns
        5) Concatenate all DataFrames into a single DataFrame
        6) Return the combined DataFrame

                    Returns:
                        :return data: pd.DataFrame - Combined DataFrame with neural recording data
        """
        csvs = list(Path(self.raw_dir).glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSVs in {self.raw_dir}")

        dfs = []
        for p in csvs:
            try:
                df = pd.read_csv(p, parse_dates=["Time"])
                dfs.append(df)
            except ValueError:
                logger.warning(f"Skipping {p.name}: missing required columns")
        return pd.concat(dfs, ignore_index=True)

    def preprocess(self, data):
        """
        Pre-process raw neuronal recording data with phase labeling and artifact removal.

        1) Check if input data is empty and return early if so
        2) Create a copy of the input DataFrame to avoid modifying the original
        3) Convert Time column to datetime format with UTC normalisation
        4) Remove rows with missing values in essential columns
        5) Apply MAD-based noise reduction:
           a. Calculate median absolute deviation (MAD)
           b. Convert MAD to standard deviation equivalent
           c. Set threshold based on configuration
           d. Remove amplitude values exceeding the threshold
        6) Assign phase labels based on timestamps from experiment parameters
        7) Apply stimulus window masking if enabled:
           a. Extract pre- and post- electrodes from configuration
           b. Map electrode indices to MEA channels
           c. Get pulse times for stimulation
           d. Process each phase separately
           e. Determine which channels to mask based on phase type
           f. Apply masking to the appropriate channels
        8) Return the preprocessed DataFrame

                    Parameters:
                        :parameter data: pd.DataFrame - Raw neuronal recording data

                    Returns:
                        :return df: pd.DataFrame - Preprocessed DataFrame with phase labels and artifacts removed
        """
        if data.empty:
            return data

        df = data.copy()

        df['Time'] = pd.to_datetime(df['Time'], format='mixed', utc=True).dt.tz_convert(None)

        df = df.dropna(subset=["Time", "channel", "Amplitude"])

        # Apply MAD noise reduction
        mad = np.median(np.abs(df['Amplitude'] - np.median(df['Amplitude'])))
        sigma = mad / 0.6745
        thresh = self.cfg.get('artefact_thresh', 8.0) * sigma

        # Mask out big artefacts
        df = df[np.abs(df['Amplitude']) < thresh]

        # assign phase labels
        params = self.cfg["experiment_params"]
        ts = [pd.to_datetime(x) for x in params.get("phase_timestamps", [])]
        names = params.get("phase_names", [])

        df["condition"] = "unknown"
        if len(ts) == len(names) + 1:
            for i, name in enumerate(names):
                start, end = ts[i], ts[i + 1]
                mask = (df["Time"] >= start) & (df["Time"] < end)
                df.loc[mask, "condition"] = name

        # Apply stim window masking if enabled
        mask_stim = params.get('mask_stim', False)
        if mask_stim:
            stim_params = params.get('stim_params', [])
            if not stim_params:
                return df

            # Extract pre and post electrodes
            # First 3 (indices 0-2) are entrainment/pre electrodes for organoids 1, 2, 3
            # Second 3 (indices 3-5) are corresponding post training electrodes
            pre_electrodes = [param.get('index') for param in stim_params[:3] if 'index' in param]
            post_electrodes = [param.get('index') for param in stim_params[3:6] if 'index' in param]
            mapped_pre_electrodes = map_channels_to_mea(pre_electrodes) if pre_electrodes else []
            mapped_post_electrodes = map_channels_to_mea(post_electrodes) if post_electrodes else []

            # pulse times - optional
            pulse_times = params.get('stim_pulse_timestamps', [])
            window_ms = params.get('mask_window_ms', 3.0)

            # If no explicit pulse times, use phase timestamps as known stimulation windows
            if not pulse_times and 'phase_timestamps' in params:
                pulse_times = [pd.to_datetime(ts) for ts in params.get('phase_timestamps', [])]

            if not pulse_times:
                return df

            # Process each phase separately to apply the correct masking
            for phase_name in df['condition'].unique():
                if phase_name == 'unknown':
                    continue

                phase_mask = df['condition'] == phase_name
                phase_df = df[phase_mask]

                if phase_df.empty:
                    continue

                channels_to_mask = []

                # Entrainment phases: mask only the organoid-specific pre channel
                if 'entrain' in phase_name:
                    if 'phase1' in phase_name and mapped_pre_electrodes:
                        channels_to_mask = [mapped_pre_electrodes[0]] 
                    elif 'phase2' in phase_name and len(mapped_pre_electrodes) > 1:
                        channels_to_mask = [mapped_pre_electrodes[1]] 
                    elif 'phase3' in phase_name and len(mapped_pre_electrodes) > 2:
                        channels_to_mask = [mapped_pre_electrodes[2]]  

                # Training phases: mask both pre and post channels for the specific organoid
                elif 'train' in phase_name:
                    if 'phase1' in phase_name:
                        if mapped_pre_electrodes and mapped_post_electrodes:
                            channels_to_mask = [mapped_pre_electrodes[0], mapped_post_electrodes[0]]  
                    elif 'phase2' in phase_name:
                        if len(mapped_pre_electrodes) > 1 and len(mapped_post_electrodes) > 1:
                            channels_to_mask = [mapped_pre_electrodes[1], mapped_post_electrodes[1]]  
                    elif 'phase3' in phase_name:
                        if len(mapped_pre_electrodes) > 2 and len(mapped_post_electrodes) > 2:
                            channels_to_mask = [mapped_pre_electrodes[2], mapped_post_electrodes[2]] 

                # Shift phases: mask only the pre channel for the specific organoid
                elif 'shift' in phase_name:
                    if 'phase1' in phase_name and mapped_pre_electrodes:
                        channels_to_mask = [mapped_pre_electrodes[0]]  
                    elif 'phase2' in phase_name and len(mapped_pre_electrodes) > 1:
                        channels_to_mask = [mapped_pre_electrodes[1]] 
                    elif 'phase3' in phase_name and len(mapped_pre_electrodes) > 2:
                        channels_to_mask = [mapped_pre_electrodes[2]] 

                # Apply masking only to the relevant channels for this phase
                if channels_to_mask:
                    channels_df = phase_df[phase_df['channel'].isin(channels_to_mask)]

                    if not channels_df.empty:
                        masked_channels_df = mask_stim_windows(channels_df, pulse_times, window_ms)

                        # Replace the original data for these channels with the masked data
                        df = df.drop(channels_df.index)
                        df = pd.concat([df, masked_channels_df], ignore_index=True)

        return df

    def analyse(self, preproc, sample_mode=False):
        """
        Analyse preprocessed neural data with phase-specific processing and cross-phase comparisons.

        1) Init empty results list
        2) Extract experiment parameters and phase groups
        3) Check if pre-processed data is empty and return early if so
        4) Get unique channels from the dataset
        5) Add a summary result with basic information about the data
        6) Handle data processing based on phase timestamps:
           a. If no phase timestamps, analyse all data as a single phase
           b. If phase timestamps are available:
              i. Convert timestamps to datetime format
              ii. Generate phase names if not provided
              iii. Assign phase labels to data based on timestamps
              iv. Process each phase separately using _run_analysis_for_data
        7) Perform cross-phase analysis for each phase group:
           a. Create UMAP embeddings for the entire group
           b. Create PCA embeddings for the entire group
           c. Generate phase-specific embeddings for comparison
        8) Generate raster plot data
        9) Return the list of all analysis results

                    Parameters:
                        :parameter preproc: pd.DataFrame - Preprocessed neuronal data with phase labels
                        :parameter sample_mode: bool - If True, use reduced data for faster processing

                    Returns:
                        :return results: list - List of dictionaries containing analysis results
        """
        results = []
        params = self.cfg["experiment_params"]
        phase_groups = params.get("phase_groups", {})

        if not phase_groups and 'phase_names' in params:
            phase_groups['all_phases'] = params['phase_names']

        if preproc.empty:
            return results

        channels = preproc['channel'].unique()

        # Add a summary result
        results.append({
            'type': 'pse_summary',
            'data': {
                'total_records': len(preproc),
                'channels': channels.tolist(),
                'time_range': [preproc['Time'].min(), preproc['Time'].max()]
            }
        })

        params = self.cfg["experiment_params"]
        phase_timestamps = params.get('phase_timestamps', [])

        if not phase_timestamps:
            results.extend(self._run_analysis_for_data(preproc, channels, "all", sample_mode=sample_mode))
        else:
            phase_timestamps = [pd.to_datetime(ts) for ts in phase_timestamps]
            phase_names = list(params.get('phase_names', []))

            preproc['condition'] = None

            if len(phase_timestamps) == len(phase_names) + 1:
                phase_intervals = [
                    (phase_names[i], phase_timestamps[i], phase_timestamps[i + 1])
                    for i in range(len(phase_names))
                ]
            else:
                # Legacy support: older configs treated timestamps as switch points,
                # not interval boundaries.
                phase_names = phase_names or [f"phase_{i}" for i in range(len(phase_timestamps) + 1)]
                if len(phase_names) < len(phase_timestamps) + 1:
                    phase_names.extend([f"phase_{i}" for i in range(len(phase_names), len(phase_timestamps) + 1)])
                phase_names[0] = "baseline"

                phase_intervals = []
                for i in range(len(phase_timestamps) + 1):
                    if i == 0:
                        start_time = preproc['Time'].min()
                        end_time = phase_timestamps[0]
                    elif i == len(phase_timestamps):
                        start_time = phase_timestamps[-1]
                        end_time = preproc['Time'].max()
                    else:
                        start_time = phase_timestamps[i-1]
                        end_time = phase_timestamps[i]
                    phase_intervals.append((phase_names[i], start_time, end_time))

            for phase_name, start_time, end_time in phase_intervals:

                mask = (preproc['Time'] >= start_time) & (preproc['Time'] < end_time)
                preproc.loc[mask, 'condition'] = phase_name

                phase_data = preproc[mask].copy()

                if phase_data.empty:
                    continue

                phase_results = self._run_analysis_for_data(phase_data, channels, phase_name, sample_mode=sample_mode)
                results.extend(phase_results)

            # UMAP and PCA for cross-phase analysis
            for group_name, phases in phase_groups.items():
                # Ensure baseline is included in all phase groups
                if "baseline" not in phases:
                    phases = ["baseline"] + list(phases)

                subset = preproc[preproc['condition'].isin(phases)].copy()
                if subset.empty:
                    continue

                # UMAP embedding for the entire group
                umap_res = compute_umap_embedding(
                    df=subset,
                    n_components=self.cfg.get('umap_n_components', 3),
                    bin_size_ms=self.cfg.get('umap_bin_size_ms', 10),
                    random_state=self.cfg.get('umap_random_state', 42),
                    sample_mode=sample_mode
                )
                umap_res['group'] = group_name
                if 'error' not in umap_res:
                    results.append({
                        'type': 'umap_with_conditions',
                        'data': umap_res
                    })

                # PCA embedding for the entire group
                pca_res = compute_pca_embedding(
                    df=subset,
                    n_components=self.cfg.get('pca_n_components', 3),
                    bin_size_ms=self.cfg.get('pca_bin_size_ms', 10),
                    random_state=self.cfg.get('pca_random_state', 42),
                    sample_mode=sample_mode
                )
                pca_res['group'] = group_name
                if 'error' not in pca_res:
                    results.append({
                        'type': 'pca',
                        'data': pca_res
                    })

                # Process each phase in the group individually for phase group visualisation
                for phase in phases:
                    phase_subset = preproc[preproc['condition'] == phase].copy()
                    if phase_subset.empty:
                        continue

                    # UMAP embedding for this phase
                    phase_umap_res = compute_umap_embedding(
                        df=phase_subset,
                        n_components=self.cfg.get('umap_n_components', 3),
                        bin_size_ms=self.cfg.get('umap_bin_size_ms', 10),
                        random_state=self.cfg.get('umap_random_state', 42),
                        sample_mode=sample_mode
                    )

                    if 'error' not in phase_umap_res:
                        phase_umap_res['phase'] = phase
                        phase_umap_res['group'] = group_name
                        results.append({
                            'type': 'umap',
                            'data': phase_umap_res
                        })

                    # PCA embedding for this phase
                    phase_pca_res = compute_pca_embedding(
                        df=phase_subset,
                        n_components=self.cfg.get('pca_n_components', 3),
                        bin_size_ms=self.cfg.get('pca_bin_size_ms', 10),
                        random_state=self.cfg.get('pca_random_state', 42),
                        sample_mode=sample_mode
                    )

                    if 'error' not in phase_pca_res:
                        phase_pca_res['phase'] = phase
                        phase_pca_res['group'] = group_name
                        results.append({
                            'type': 'pca',
                            'data': phase_pca_res
                        })

            # Raster
            raster_data = self._generate_raster_data(preproc)
            if raster_data:
                results.append(raster_data)

        return results

    def _run_analysis_for_data(self, data, channels, phase_name, sample_mode=False):
        """
        Run analyses on a specific subset of data for a given phase.

        1) Init empty results list
        2) Set up channel pairs for analysis:
           a. Use configured channel pairs if available
           b. Generate sequential pairs if not configured
           c. Validate that channels exist in the dataset
        3) Generate raster plot data for the phase
        4) Return the list of analysis results

        Note: This method contains commented-out code for additional analyses that can be enabled depending
        on compututational requirements and available data. These analyses include::
        - FFT (Fast Fourier Transform)
        - ISI (Inter-Spike Interval)
        - Avalanche detection
        - Firing rate analysis
        - Power law fitting
        - PSD (Power Spectral Density)
        - Auto-correlation
        - Granger causality
        - Statistical tests
        - Cross-correlation

                    Parameters:
                        :parameter data: pd.DataFrame - Subset of neural data for a specific phase
                        :parameter channels: list - List of channels to analyse
                        :parameter phase_name: str - Name of the phase being analysed
                        :parameter sample_mode: bool - If True, use reduced data for faster processing

                    Returns:
                        :return results: list - List of dictionaries containing analysis results
        """
        results = []

        # FFT
        for channel in channels:
            fft_res = compute_fft(
                df=data,
                channel=channel,
                resample_interval=self.cfg.get('fft_resample_ms', 1),
                freq_cap=self.cfg.get('fft_freq_cap', 10.0)
            )

            results.append({
                'type': 'fft',
                'data': {
                    'channel': channel,
                    'freqs': fft_res['freqs'],
                    'amplitudes': fft_res['amplitudes'],
                    'phase': phase_name
                }
            })

        # ISI
        for channel in channels:
            isi_res = compute_isi(
                df=data,
                channel=channel
            )

            # Add phase name to the result
            if 'error' not in isi_res:
                isi_res['phase'] = phase_name

            results.append({
                'type': 'isi',
                'data': isi_res
            })

        # Avalanche
        for channel in channels:
            isi_res = compute_isi(
                df=data,
                channel=channel
            )

            if 'isi' in isi_res and len(isi_res['isi']) > 0:
                avalanche_res = compute_avalanche_distribution(
                    isi=isi_res['isi'],
                    threshold_s=self.cfg.get('avalanche_threshold_s', 0.1)
                )

                results.append({
                    'type': 'avalanche',
                    'data': {
                        'channel': channel,
                        'avalanches': avalanche_res,
                        'phase': phase_name
                    }
                })

        # Firing rates
        for channel in channels:
            firing_rate_res = compute_firing_rates(
                df=data[data['channel'] == channel],
                threshold=self.cfg.get('firing_rate_threshold', 0.0),
                bin_size_s=self.cfg.get('firing_rate_bin_size_s', 1.0)
            )

            if 'error' not in firing_rate_res:
                firing_rate_res['phase'] = phase_name

                # Calculate Z-scores for firing rates using ipp_stats
                zscore_res = compute_firing_rate_zscore(
                    df=data[data['channel'] == channel],
                    channel=channel,
                    bin_size_s=self.cfg.get('firing_rate_bin_size_s', 1.0)
                )

                if 'error' not in zscore_res:
                    zscore_res['phase'] = phase_name

                    # Add Z-score results
                    results.append({
                        'type': 'firing_rate_zscore',
                        'data': zscore_res
                    })

                # Calculate Kernel Density Estimation using ipp_stats
                kde_res = compute_kernel_density(
                    df=data[data['channel'] == channel],
                    channel=channel,
                    bin_size_s=self.cfg.get('firing_rate_bin_size_s', 1.0)
                )

                if 'error' not in kde_res:
                    kde_res['phase'] = phase_name

                    # Add KDE results
                    results.append({
                        'type': 'kernel_density',
                        'data': kde_res
                    })

            results.append({
                'type': 'firing_rate',
                'data': firing_rate_res
            })

        # Powerlaw
        for channel in channels:
            powerlaw_res = compute_powerlaw(
                df=data,
                channel=channel
            )

            if 'error' not in powerlaw_res:
                powerlaw_res['phase'] = phase_name

                results.append({
                    'type': 'powerlaw',
                    'data': powerlaw_res
                })

            # PSD
            psd_res = compute_power_spectral_density(
                df=data,
                channel=channel,
                fs=1e3,
                nperseg=self.cfg.get('psd_nperseg', None),
                noverlap=self.cfg.get('psd_noverlap', None)
            )

            if 'error' not in psd_res:
                results.append({
                    'type': 'psd',
                    'data': {
                        'channel': channel,
                        'freqs': psd_res['freqs'],
                        'psd': psd_res['psd'],
                        'phase': phase_name  # if in phased analysis
                    }
                })

            # Auto-correlation
            auto_corr_res = compute_auto_correlation(
                df=data,
                channel=channel,
                bin_size_ms=self.cfg.get('auto_correlation_bin_size_ms', 1),
                sample_mode=sample_mode
            )

            if 'error' not in auto_corr_res:
                auto_corr_res['phase'] = phase_name

                results.append({
                    'type': 'auto_correlation',
                    'data': auto_corr_res
                })

        # GC
        channel_pairs = self.cfg.get('channel_pairs', [])
        if not channel_pairs and len(channels) >= 2:
            channel_pairs = [(channels[i], channels[i+1]) for i in range(len(channels)-1)]

        valid_channel_pairs = []
        for ch1, ch2 in channel_pairs:
            if ch1 in channels and ch2 in channels:
                valid_channel_pairs.append((ch1, ch2))
            else:
                print(f"Skipping channel pair ({ch1}, {ch2}) as one or both channels are not in the dataset")

        channel_pairs = valid_channel_pairs

        for channel1, channel2 in channel_pairs:
            gc_res = compute_granger_causality(
                df=data,
                pre_channel=channel1,
                post_channel=channel2,
                resample_interval=self.cfg.get('gc_resample_interval', '1s'),
                max_lag=self.cfg.get('gc_max_lag', 5),
                sample_mode=sample_mode
            )

            if 'error' not in gc_res:
                gc_res['phase'] = phase_name

                results.append({
                    'type': 'granger',
                    'data': gc_res
                })

        # Stats
        for channel1, channel2 in channel_pairs:
            stats_res = compute_statistical_tests(
                df=data,
                pre_channel=channel1,
                post_channel=channel2,
                sample_mode=sample_mode
            )

            if 'error' not in stats_res:
                stats_res['phase'] = phase_name

                results.append({
                    'type': 'statistical_tests',
                    'data': stats_res
                })

            # Cross-correlation
            cross_corr_res = compute_cross_correlation(
                df=data,
                channel1=channel1,
                channel2=channel2,
                bin_size_ms=self.cfg.get('cross_correlation_bin_size_ms', 10),
                sample_mode=sample_mode
            )

            if 'error' not in cross_corr_res:
                cross_corr_res['phase'] = phase_name

                results.append({
                    'type': 'cross_correlation',
                    'data': cross_corr_res
                })

        # Raster
        raster_data = self._generate_raster_data(data, phase=phase_name)
        if raster_data:
            results.append(raster_data)

        return results

    def visualise(self, results):
        """
        Visualise analysis results with phase-specific and cross-phase plots.

        1) Check if there are any results to visualise
        2) Create a Plotter instance for visualisation
        3) Define result types for different categories:
           a. Per-channel results (auto_correlation, isi, firing_rate, etc.)
           b. Pairwise results (cross_correlation, granger, statistical_tests)
           c. Cross-phase results (umap, pca)
        4) Create embeddings directory for cross-phase analyses
        5) Process per-channel results with batch plotting
        6) Process pairwise results with batch plotting
        7) Group embedding results by type and phase group:
           a. Collect UMAP results by group
           b. Collect PCA results by group
        8) Create phase group plots for UMAP:
           a. Generate standard UMAP phase group plots
           b. Generate layered UMAP phase group plots
        9) Create phase group plots for PCA:
           a. Generate standard PCA phase group plots
           b. Generate layered PCA phase group plots
        10) Process special cases and other result types:
            a. Send cross-phase analyses to embeddings directory
            b. Send phase-specific results to phase directories
            c. Handle errors for each plotting operation

                    Parameters:
                        :parameter results: list - List of dictionaries containing analysis results

                    Returns:
                        :return: None
        """
        if not results:
            print("No results to visualize")
            return

        plotter = Plotter(self.results_dir)

        # Group plotting
        per_channel_types = ['auto_correlation', 'isi', 'firing_rate', 'fft', 'kernel_density', 'firing_rate_zscore', 'powerlaw', 'psd']
        pairwise_types = ['cross_correlation', 'granger', 'statistical_tests']
        cross_phase_types = ['umap', 'umap_with_conditions', 'pca']

        # Create embeddings directory
        embeddings_dir = self.results_dir / 'embeddings'
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        for result_type in per_channel_types:
            type_results = [r['data'] for r in results if r['type'] == result_type]
            if type_results:
                plotter.plot(result_type, type_results, batch=True)

        # Process pairwise results -> only in phase-specific directories
        for result_type in pairwise_types:
            type_results = [r['data'] for r in results if r['type'] == result_type]
            if type_results:
                plotter.plot(result_type, type_results, batch=True)

        # Process embeddings for phase groups
        # First, collect all embedding results by type and group
        umap_by_group = {}
        pca_by_group = {}

        for result in results:
            if result['type'] == 'umap' and 'phase' in result['data']:
                group = result['data'].get('group', 'all')
                if group not in umap_by_group:
                    umap_by_group[group] = []
                umap_by_group[group].append(result['data'])
            elif result['type'] == 'pca' and 'phase' in result['data']:
                group = result['data'].get('group', 'all')
                if group not in pca_by_group:
                    pca_by_group[group] = []
                pca_by_group[group].append(result['data'])

        # Create phase group plots for UMAP
        embeddings_plotter = Plotter(embeddings_dir)
        for group, group_data in umap_by_group.items():
            try:
                embeddings_plotter._plot_umap_phase_group(group_data)
                print(f"Successfully plotted UMAP phase group: {group}")

                embeddings_plotter._plot_umap_phase_group_layered(group_data)
                print(f"Successfully plotted layered UMAP phase group: {group}")
            except Exception as e:
                print(f"Error plotting UMAP phase group {group}: {e}")

        for group, group_data in pca_by_group.items():
            try:
                embeddings_plotter._plot_pca_phase_group(group_data)
                print(f"Successfully plotted PCA phase group: {group}")

                embeddings_plotter._plot_pca_phase_group_layered(group_data)
                print(f"Successfully plotted layered PCA phase group: {group}")
            except Exception as e:
                print(f"Error plotting PCA phase group {group}: {e}")

        # Process special cases (e.g., raster plots, UMAP, etc.)
        for result in results:
            try:
                # Cross-phase analysis goes in embeddings directory
                if result['type'] in cross_phase_types:
                    embeddings_plotter.plot(result['type'], result['data'])
                # Phase-specific results go in phase directories
                elif result['type'] not in per_channel_types + pairwise_types:
                    if isinstance(result['data'], dict) and 'phase' in result['data']:
                        phase = result['data']['phase']
                        plotter.plot(result['type'], result['data'], phase=phase)
                    else:
                        # Else goes in embeddings directory
                        embeddings_plotter.plot(result['type'], result['data'])
            except Exception as e:
                print(f"Error plotting {result['type']}: {e}")
