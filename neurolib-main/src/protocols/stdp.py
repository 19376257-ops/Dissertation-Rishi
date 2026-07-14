import pandas as pd
import numpy as np
from pathlib import Path
import logging
from itertools import combinations

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
from src.analysis.prc import compute_prc
from src.analysis.psd import compute_power_spectral_density
from src.analysis.correlation import compute_cross_correlation, compute_auto_correlation
from src.analysis.utils import map_channels_to_mea
from src.analysis.utils import group_conditions_by_pre_post
from src.analysis.utils import mask_stim_windows, compute_stim_artifact_template, apply_template_subtraction

logger = logging.getLogger(__name__)


class STDPProtocol(ProtocolBase):
    """
    Spike-Timing-Dependent Plasticity (STDP) protocol for analysing neural recordings with pre-post stimulation.

    1) Load raw neural recording data from CSV files
    2) Preprocess data with condition labeling and artifact removal:
       - Convert time to datetime format
       - Remove rows with missing values
       - Apply MAD-based noise reduction
       - Assign condition labels based on train timestamps
       - Apply stimulus window masking for specific channels
    3) Analyse data with multiple metrics:
       - Compute channel-specific metrics (FFT, ISI, avalanche, etc.)
       - Compute channel pair metrics (Granger causality, statistical tests, etc.)
       - Generate embeddings (UMAP, PCA) for visualisation
       - Calculate phase response curves (PRCs) for pre- and post- stimulation
    4) Visualise results with specialized plots for STDP analysis
    """

    def load(self) -> pd.DataFrame:
        """
        Load raw neural recording data from CSV files in the raw directory.

        1) Find all CSV files in the raw directory
        2) Validate that at least one CSV file exists
        3) Init an empty list to store DataFrames
        4) For each CSV file:
           a. Attempt to read the file with datetime parsing for the Time column
           b. Validate that required columns are present
           c. Add the DataFrame to the collection
           d. Skip files with missing required columns or other errors
        5) Validate that at least one valid CSV file was processed
        6) Concatenate all DataFrames into a single DataFrame
        7) Return the combined DataFrame

                    Returns:
                        :return data: pd.DataFrame - Combined DataFrame with neural recording data
        """
        # load all CSVs
        csvs = list(Path(self.raw_dir).glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSVs in {self.raw_dir}")

        frames = []
        for p in csvs:
            try:
                df = pd.read_csv(p, parse_dates=['Time'])
                for col in ('Time', 'channel'):
                    if col not in df.columns:
                        raise ValueError(f"Missing column {col}")
                frames.append(df)
            except Exception as e:
                logger.warning(f"Skipping {p.name}: {e}")
        if not frames:
            raise RuntimeError("No valid CSVs found.")
        return pd.concat(frames, ignore_index=True)

    def preprocess(self, data):
        """
        Preprocess raw neural recording data with condition labeling and artifact removal.

        1) Check if input data is empty and return early if so
        2) Create a copy of the input DataFrame to avoid modifying the original
        3) Convert Time column to datetime format with UTC normalisation
        4) Remove rows with missing values in essential columns
        5) Apply MAD-based noise reduction:
           a. Calculate median absolute deviation (MAD)
           b. Convert MAD to standard deviation equivalent
           c. Set threshold based on configuration
           d. Remove amplitude values exceeding the threshold
        6) Assign condition labels based on train timestamps:
           a. Extract train timestamps from experiment parameters
           b. Set default condition as "no_stim"
           c. Mark pre- and post- train periods with appropriate labels
        7) Identify pre and post electrodes:
           a. Extract electrode lists from parameters
           b. Map electrode indices to MEA channels
           c. Mark channels as pre or post in the DataFrame
        8) Apply stimulus window masking if enabled:
           a. Extract pulse times from train timestamps
           b. Apply template subtraction if configured
           c. Mask out stimulation artifacts
        9) Return the preprocessed DataFrame

                    Parameters:
                        :parameter data: pd.DataFrame - Raw neural recording data

                    Returns:
                        :return df: pd.DataFrame - Preprocessed DataFrame with condition labels and artifacts removed
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

        params = self.cfg.get('experiment_params', {})
        train_timestamps = params.get('train_timestamps', {})
        dur = pd.Timedelta(milliseconds=params.get('train_duration_ms', 600))

        df["condition"] = "no_stim"

        """
        Handles stimulation train marking
        """
        if isinstance(train_timestamps, dict) and 'pre_times' in train_timestamps and 'post_times' in train_timestamps:
            pre_times = train_timestamps.get('pre_times', [])
            print(f"DEBUG: pre_times: {pre_times}")
            post_times = train_timestamps.get('post_times', [])
            print(f"DEBUG: post_times: {post_times}")

            # Convert string timestamps to datetime objects if needed
            pre_times = [pd.to_datetime(ts, utc=True).tz_convert(None) if isinstance(ts, str) else ts for ts in pre_times]
            post_times = [pd.to_datetime(ts, utc=True).tz_convert(None) if isinstance(ts, str) else ts for ts in post_times]

            for i, t0 in enumerate(pre_times):
                if i < len(post_times):
                    mask = (df['Time'] >= t0) & (df['Time'] < post_times[i])
                    df.loc[mask, 'condition'] = f'train_{i+1}_pre'

                    if i + 1 < len(pre_times):
                        end_time = pre_times[i+1]
                    else:
                        end_time = post_times[i] + pd.Timedelta(seconds=30)

                    mask = (df['Time'] >= post_times[i]) & (df['Time'] < end_time)
                    df.loc[mask, 'condition'] = f'train_{i+1}_post'
        else:
            # Fallback
            trains = train_timestamps if isinstance(train_timestamps, list) else []
            trains = [pd.to_datetime(ts, utc=True).tz_convert(None) if isinstance(ts, str) else ts for ts in trains]

            for i, t0 in enumerate(trains):
                mask = (df['Time'] >= t0) & (df['Time'] < t0 + dur)
                df.loc[mask, 'condition'] = f'stim_{i}'

        # Get electrode lists from parameters
        pre_list = params.get('pre_electrodes', [])
        post_list = params.get('post_electrodes', [])

        # Map electrode numbers from 1-32 base to 1-128 base for MEA
        mapped_pre_list = map_channels_to_mea(pre_list)
        mapped_post_list = map_channels_to_mea(post_list)

        print(f"DEBUG: pre_list (original): {pre_list}")
        print(f"DEBUG: pre_list (mapped): {mapped_pre_list}")
        print(f"DEBUG: post_list (original): {post_list}")
        print(f"DEBUG: post_list (mapped): {mapped_post_list}")

        # Use mapped channel numbers for filtering
        df['is_pre'] = df['channel'].isin(mapped_pre_list)
        df['is_post'] = df['channel'].isin(mapped_post_list)

        # Apply stim window masking if enabled
        mask_stim = params.get('mask_stim', False)
        if mask_stim:
            pulse_times = []

            # Extract pulse times from train ts
            if isinstance(train_timestamps, dict):
                if 'pre_times' in train_timestamps:
                    pulse_times.extend(train_timestamps['pre_times'])
                if 'post_times' in train_timestamps:
                    pulse_times.extend(train_timestamps['post_times'])
            elif isinstance(train_timestamps, list):
                pulse_times.extend(train_timestamps)

            # Use explicit stim_pulse_timestamps if available <-- Future Proofing
            if 'stim_pulse_timestamps' in params:
                pulse_times.extend(params['stim_pulse_timestamps'])

            if pulse_times:
                window_ms = params.get('mask_window_ms', 3.0)

                # Apply template subtraction if enabled
                use_template_subtraction = params.get('use_template_subtraction', False)
                if use_template_subtraction:
                    stim_channels = mapped_pre_list + mapped_post_list
                    templates = compute_stim_artifact_template(df, pulse_times, window_ms + 2.0, stim_channels)
                    df = apply_template_subtraction(df, pulse_times, templates, window_ms + 2.0)

                df = mask_stim_windows(df, pulse_times, window_ms)

        return df

    def _analyse_channel_metrics(self, df, all_channels, results):
        """
        Compute channel-specific metrics for each channel in the dataset.

        1) Iterate through each channel in the dataset
        2) Filter data for the current channel
        3) Skip channels with no spikes
        4) For each channel, compute:
           a. Fast Fourier Transform (FFT)
           b. Inter-Spike Intervals (ISI)
           c. Avalanche distribution
           d. Power-law fit
           e. Firing rates
           f. Power spectral density
           g. Auto-correlation
        5) Add results to the results list

                    Parameters:
                        :parameter df: pd.DataFrame - Preprocessed neural data
                        :parameter all_channels: list - List of all channels to analyse
                        :parameter results: list - List to store analysis results
        """
        for ch in all_channels:
            ch_df = df[df['channel'] == ch].copy()
            if ch_df.empty:
                print(f"No spikes found for channel {ch}")
                continue

            print(f"Found {len(ch_df)} spikes for channel {ch}")

            # Compute FFT
            fft_res = compute_fft(
                df=ch_df,
                channel=ch,
                resample_interval=self.cfg.get('fft_resample_ms', 1),
                freq_cap=self.cfg.get('fft_freq_cap', 10.0)
            )

            results.append({
                'type': 'fft',
                'data': {
                    'channel': ch,
                    'freqs': fft_res['freqs'],
                    'amplitudes': fft_res['amplitudes']
                }
            })

            # Compute ISI
            isi_res = compute_isi(
                df=ch_df,
                channel=ch
            )

            results.append({
                'type': 'isi',
                'data': isi_res
            })

            # Compute Avalanche distribution
            if 'isi' in isi_res and len(isi_res['isi']) > 0:
                avalanche_res = compute_avalanche_distribution(
                    isi=isi_res['isi'],
                    threshold_s=self.cfg.get('avalanche_threshold_s', 0.1)
                )

                results.append({
                    'type': 'avalanche',
                    'data': {
                        'channel': ch,
                        'avalanches': avalanche_res
                    }
                })

            # Compute Powerlaw fit
            powerlaw_res = compute_powerlaw(
                df=ch_df,
                channel=ch
            )

            # Only add if there was no error
            if 'error' not in powerlaw_res:
                results.append({
                    'type': 'powerlaw',
                    'data': powerlaw_res
                })

            # Compute Firing rates
            firing_rate_res = compute_firing_rates(
                df=ch_df,
                threshold=self.cfg.get('firing_rate_threshold', 0.0),
                bin_size_s=self.cfg.get('firing_rate_bin_size_s', 1.0)
            )

            results.append({
                'type': 'firing_rate',
                'data': firing_rate_res
            })

            # Compute Power spectral density
            psd_res = compute_power_spectral_density(
                df=ch_df,
                channel=ch,
                fs=1e3,
                nperseg=self.cfg.get('psd_nperseg', None),
                noverlap=self.cfg.get('psd_noverlap', None)
            )

            results.append({
                'type': 'psd',
                'data': {
                    'channel': ch,
                    'freqs': psd_res['freqs'],
                    'psd': psd_res['psd'],
                }
            })

            # Compute Auto-correlation
            ac_res = compute_auto_correlation(
                df=ch_df,
                channel=ch
            )

            if 'error' not in ac_res:
                results.append({
                    'type': 'auto_correlation',
                    'data': {
                        'channel': ch,
                        'lags': ac_res['lags'],
                        'correlation': ac_res['correlation']
                    }
                })

    def _analyse_channel_pair_metrics(self, df, all_channels, results):
        """
        Compute metrics for pairs of channels to analyse relationships between neurons.

        1) Generate all possible pairs of channels using combinations
        2) For each channel pair:
           a. Filter data to include only the current pair of channels
           b. Compute Granger causality to identify predictive relationships
           c. Perform statistical tests to compare neural activity
           d. Calculate cross-correlation to detect temporal relationships
        3) Add results to the results list

                    Parameters:
                        :parameter df: pd.DataFrame - Pre-processed neuronal data
                        :parameter all_channels: list - List of all channels to analyse
                        :parameter results: list - List to store analysis results
        """
        # Generate all possible pairs of channels
        channel_pairs = list(combinations(all_channels, 2))

        for pre_channel, post_channel in channel_pairs:
            # Filter data for this channel pair
            pair_df = df[df['channel'].isin([pre_channel, post_channel])]

            # Compute GC
            gc_res = compute_granger_causality(
                df=pair_df,
                pre_channel=pre_channel,
                post_channel=post_channel,
                resample_interval=self.cfg.get('gc_resample_interval', '1s'),
                max_lag=self.cfg.get('gc_max_lag', 5)
            )

            # Only add if there was no error
            if 'error' not in gc_res:
                results.append({
                    'type': 'granger',
                    'data': gc_res
                })

            # Compute Stat tests
            stats_res = compute_statistical_tests(
                df=pair_df,
                pre_channel=pre_channel,
                post_channel=post_channel
            )

            # Only add if there was no error
            if 'error' not in stats_res:
                results.append({
                    'type': 'statistical_tests',
                    'data': stats_res
                })

            # Compute Cross-correlation
            cc_res = compute_cross_correlation(
                df=pair_df,
                channel1=pre_channel,
                channel2=post_channel,
            )

            if 'error' not in cc_res:
                results.append({
                    'type': 'cross_correlation',
                    'data': {
                        'channel1': pre_channel,
                        'channel2': post_channel,
                        'lags': cc_res['lags'],
                        'correlation': cc_res['correlation']
                    }
                })

    def _analyse_embeddings(self, df, results, sample_mode=False):
        """
        Generate dimensionality reduction embeddings for visualisation of neural activity patterns.

        1) Check if there is enough data for meaningful embeddings (at least 100 data points)
        2) Compute UMAP embedding:
           a. Configure UMAP parameters from configuration
           b. Generate the embedding with specified dimensions
           c. Add the embedding to results if successful
           d. Create pre vs post grouping for STDP analysis
        3) Compute PCA embedding:
           a. Configure PCA parameters from configuration
           b. Generate the embedding with specified dimensions
           c. Add the embedding to results if successful
           d. Create pre vs post grouping for STDP analysis

                    Parameters:
                        :parameter df: pd.DataFrame - Preprocessed neuronal data
                        :parameter results: list - List to store analysis results
                        :parameter sample_mode: bool - If True, use reduced data for faster processing
        """
        # Only compute embeddings if we have enough data
        if len(df) <= 100:
            return

        # Compute UMAP
        umap_res = compute_umap_embedding(
            df=df,
            n_components=self.cfg.get('umap_n_components', 3),
            bin_size_ms=self.cfg.get('umap_bin_size_ms', 10),
            random_state=self.cfg.get('umap_random_state', 42),
            sample_mode=sample_mode
        )

        if 'error' not in umap_res:
            results.append({
                'type': 'umap',
                'data': umap_res
            })

            # Create pre vs post UMAP embedding
            umap_pre_post = umap_res.copy()
            if 'conditions' in umap_pre_post:
                umap_pre_post['conditions'] = group_conditions_by_pre_post(umap_pre_post['conditions'])
                umap_pre_post['grouping'] = 'pre_post'

                results.append({
                    'type': 'umap_pre_post',
                    'data': umap_pre_post
                })

        # Compute PCA
        pca_res = compute_pca_embedding(
            df=df,
            n_components=self.cfg.get('pca_n_components', 3),
            bin_size_ms=self.cfg.get('pca_bin_size_ms', 10),
            random_state=self.cfg.get('pca_random_state', 42),
            sample_mode=sample_mode
        )

        if 'error' not in pca_res:
            results.append({
                'type': 'pca',
                'data': pca_res
            })

            # Create pre vs post PCA embedding
            pca_pre_post = pca_res.copy()
            if 'conditions' in pca_pre_post:
                pca_pre_post['conditions'] = group_conditions_by_pre_post(pca_pre_post['conditions'])
                pca_pre_post['grouping'] = 'pre_post'

                results.append({
                    'type': 'pca_pre_post',
                    'data': pca_pre_post
                })

    def _process_prc_results(self, pre_prc_res, post_prc_res, ch, results):
        """
        Process and format phase response curve (PRC) results for visualisation.

        1) Get the number of phase bins from configuration
        2) Handle different result availability scenarios:
           a. If both pre and post results are available:
              i. Extract train numbers from result keys
              ii. Create a combined delta phase dictionary
              iii. Init with empty arrays for missing trains
              iv. Populate with pre and post train PRCs
           b. If only pre results are available:
              i. Create delta phase with only pre data
              ii. Initialize empty arrays for missing trains
           c. If only post results are available:
              i. Create delta phase with only post data
              ii. Initialise empty arrays for missing trains
        3) Create a formatted PRC result with phase bins and delta phase data
        4) Add the result to the results list

                    Parameters:
                        :parameter pre_prc_res: dict - PRC results for pre-stimulation
                        :parameter post_prc_res: dict - PRC results for post-stimulation
                        :parameter ch: int - Channel number
                        :parameter results: list - List to store analysis results
        """
        num_phase_bins = self.cfg.get('prc_num_bins', 50)

        # If both pre and post results are available
        if pre_prc_res and 'error' not in pre_prc_res and post_prc_res and 'error' not in post_prc_res:
            print(f"Both pre and post PRC results computed successfully")

            delta_phase = {}
            pre_train_numbers = []
            for key in pre_prc_res.keys():
                if key.startswith('prc_train_'):
                    train_num = key.split('_')[2]
                    if train_num.isdigit():
                        pre_train_numbers.append(int(train_num) - 1)  # Convert to 0-based index

            post_train_numbers = []
            for key in post_prc_res.keys():
                if key.startswith('prc_train_'):
                    train_num = key.split('_')[2]
                    if train_num.isdigit():
                        post_train_numbers.append(int(train_num) - 1)  # Convert to 0-based index

            # Ensure we have 5 trains <- Protocol specifc
            train_numbers = sorted(set(pre_train_numbers + post_train_numbers))
            if len(train_numbers) != 5:
                print(f"Warning: Expected 5 trains, but found {len(train_numbers)}: {train_numbers}")

            empty_prc = np.zeros(num_phase_bins)

            for i in range(5):
                delta_phase[f'train_{i}_pre'] = empty_prc.copy()
                delta_phase[f'train_{i}_post'] = empty_prc.copy()

            for key, value in pre_prc_res.items():
                if key.startswith('prc_train_'):
                    parts = key.split('_')
                    if len(parts) >= 3 and parts[2].isdigit():
                        train_num = int(parts[2]) - 1  # Convert to 0-based index
                        delta_phase[f'train_{train_num}_pre'] = value

            for key, value in post_prc_res.items():
                if key.startswith('prc_train_'):
                    parts = key.split('_')
                    if len(parts) >= 3 and parts[2].isdigit():
                        train_num = int(parts[2]) - 1  # Convert to 0-based index
                        delta_phase[f'train_{train_num}_post'] = value

            prc_result = {
                'type': 'prc',
                'data': {
                    'phase_bins': np.linspace(0, 1, num_phase_bins),
                    'delta_phase': delta_phase,
                    'electrode': ch
                }
            }

            results.append(prc_result)

        # If only pre results are available
        elif pre_prc_res and 'error' not in pre_prc_res:
            print(f"Only pre PRC results available")

            # Create a new format for PRC results with only pre data
            delta_phase = {}

            # Create empty arrays for missing trains
            empty_prc = np.zeros(num_phase_bins)

            # Initialize delta_phase with empty arrays for all 5 trains (both pre and post)
            for i in range(5):
                delta_phase[f'train_{i}_pre'] = empty_prc.copy()
                delta_phase[f'train_{i}_post'] = empty_prc.copy()

            # Add pre train PRCs to delta
            for key, value in pre_prc_res.items():
                if key.startswith('prc_train_'):
                    parts = key.split('_')
                    if len(parts) >= 3 and parts[2].isdigit():
                        train_num = int(parts[2]) - 1  # Convert to 0-based index
                        delta_phase[f'train_{train_num}_pre'] = value

            prc_result = {
                'type': 'prc',
                'data': {
                    'phase_bins': np.linspace(0, 1, num_phase_bins),
                    'delta_phase': delta_phase,
                    'electrode': ch
                }
            }

            results.append(prc_result)

        # If only post results are available
        elif post_prc_res and 'error' not in post_prc_res:
            print(f"Only post PRC results available")

            # Create empty arrays for missing trains
            delta_phase = {}
            empty_prc = np.zeros(num_phase_bins)

            # Init delta_phase with empty arrays for all 5 train
            for i in range(5):
                delta_phase[f'train_{i}_pre'] = empty_prc.copy()
                delta_phase[f'train_{i}_post'] = empty_prc.copy()

            # Add post train PRCs to delta fase
            for key, value in post_prc_res.items():
                if key.startswith('prc_train_'):
                    parts = key.split('_')
                    if len(parts) >= 3 and parts[2].isdigit():
                        train_num = int(parts[2]) - 1  # Convert to 0-based index
                        delta_phase[f'train_{train_num}_post'] = value

            # Create the new PRC result format
            prc_result = {
                'type': 'prc',
                'data': {
                    'phase_bins': np.linspace(0, 1, num_phase_bins),
                    'delta_phase': delta_phase,
                    'electrode': ch
                }
            }

            results.append(prc_result)
        else:
            print(f"No valid PRC results computed")

    def _analyse_prcs(self, df, all_channels, results):
        """
        Compute phase response curves (PRCs) for analysing neuronal responses to stimulation.

        1) Extract experiment parameters for PRC computation:
           a. Get train timestamps from configuration
           b. Get stimulation frequency in Hz
        2) Validate required parameters are available
        3) Convert string timestamps to datetime objects
        4) Identify pre and post train conditions from data
        5) Create frequency dictionaries for pre and post trains
        6) For each channel:
           a. Filter data for the current channel
           b. Skip channels with no spikes
           c. Compute PRCs for pre-stimulation trains
           d. Compute PRCs for post-stimulation trains
           e. Process and format the PRC results
        7) Results are added to the results list by the _process_prc_results method

                    Parameters:
                        :parameter df: pd.DataFrame - Preprocessed neural data
                        :parameter all_channels: list - List of all channels to analyse
                        :parameter results: list - List to store analysis results
        """
        params = self.cfg.get('experiment_params', {})
        train_timestamps = params.get('train_timestamps', {})
        hz = params.get('stdp_frequency_hz', None) or params.get('stim_frequency', None)

        print(f"Computing PRCs with frequency: {hz} Hz")

        if not isinstance(train_timestamps, dict) or 'pre_times' not in train_timestamps or 'post_times' not in train_timestamps:
            print("No valid train timestamps found for PRC computation")
            return

        pre_times = train_timestamps.get('pre_times', [])
        post_times = train_timestamps.get('post_times', [])

        print(f"Pre times: {pre_times}")
        print(f"Post times: {post_times}")

        # Convert string timestamps to datetime objects if needed
        pre_times = [pd.to_datetime(ts, utc=True).tz_convert(None) if isinstance(ts, str) else ts for ts in pre_times]
        post_times = [pd.to_datetime(ts, utc=True).tz_convert(None) if isinstance(ts, str) else ts for ts in post_times]

        if not pre_times or not post_times or not hz:
            print("Missing required data for PRC computation")
            return

        # Get all train conditions
        all_conditions = df['condition'].unique()
        pre_train_conditions = [cond for cond in all_conditions if cond.startswith('train_') and 'pre' in cond]
        post_train_conditions = [cond for cond in all_conditions if cond.startswith('train_') and 'post' in cond]

        print(f"Pre train conditions: {pre_train_conditions}")
        print(f"Post train conditions: {post_train_conditions}")

        # If no explicit conditions, infer from timing
        if not post_train_conditions:
            post_train_conditions = [cond for cond in all_conditions if cond.startswith('train_') and 'pre' not in cond]
            print(f"Inferred post train conditions: {post_train_conditions}")

        # Create dict of phase frequencies for pre and post trains
        pre_phase_frequencies = {}
        post_phase_frequencies = {}

        # Set frequencies for pre and post trains
        for cond in pre_train_conditions:
            pre_phase_frequencies[cond] = hz

        for cond in post_train_conditions:
            post_phase_frequencies[cond] = hz

        for ch in all_channels:
            print(f"Processing channel: {ch}")
            ch_df = df[df['channel'] == ch].copy()
            if ch_df.empty:
                print(f"No spikes found for channel {ch}")
                continue

            print(f"Found {len(ch_df)} spikes for channel {ch}")

            # Calculate PRCs for pre trains
            pre_prc_res = None
            if pre_phase_frequencies:
                print(f"Computing PRCs for pre trains with frequencies: {pre_phase_frequencies}")
                pre_prc_res = compute_prc(
                    df=ch_df,
                    channel=ch,
                    phase_frequencies=pre_phase_frequencies,
                    num_bins=self.cfg.get('prc_num_bins', 50)
                )

            # Calculate PRCs for post trains
            post_prc_res = None
            if post_phase_frequencies:
                print(f"Computing PRCs for post trains with frequencies: {post_phase_frequencies}")
                post_prc_res = compute_prc(
                    df=ch_df,
                    channel=ch,
                    phase_frequencies=post_phase_frequencies,
                    num_bins=self.cfg.get('prc_num_bins', 50)
                )

            # Process and format the PRC results
            self._process_prc_results(pre_prc_res, post_prc_res, ch, results)

    def analyse(self, df: pd.DataFrame, sample_mode=False) -> list:
        """
        Analyse pre-processed neuronal data with STDP-specific metrics and visualisations.

        1) Initialize empty results list
        2) Check if preprocessed data is empty and return early if so
        3) Extract unique channels from the dataset
        4) Analyse channel-specific metrics:
           a. Compute FFT for frequency analysis
           b. Calculate inter-spike intervals (ISI)
           c. Detect neuronal avalanches
           d. Fit power-law distributions
           e. Calculate firing rates
           f. Compute power spectral density
           g. Analyse auto-correlations
        5) Analyse channel pair metrics:
           a. Compute Granger causality between channels
           b. Perform statistical tests between channels
           c. Calculate cross-correlations between channels
        6) Generate embeddings for visualisation:
           a. Create UMAP embeddings with pre/post grouping
           b. Create PCA embeddings with pre/post grouping
        7) Calculate phase response curves (PRCs) for pre and post stimulation
        8) Generate raster plot data
        9) Return the list of all analysis results

                    Parameters:
                        :parameter df: pd.DataFrame - Preprocessed neural data
                        :parameter sample_mode: bool - If True, use reduced data for faster processing

                    Returns:
                        :return results: list - List of dictionaries containing analysis results
        """
        results = []
        if df.empty:
            return results

        # Get all unique channels in the data
        all_channels = df['channel'].unique()
        print(f"All channels in data: {all_channels}")

        # 1.
        self._analyse_channel_metrics(df, all_channels, results)

        # 2.
        self._analyse_channel_pair_metrics(df, all_channels, results)

        # 3.
        self._analyse_embeddings(df, results, sample_mode)

        # 4.
        self._analyse_prcs(df, all_channels, results)

        # 5.
        raster = self._generate_raster_data(df)
        if raster:
            results.append(raster)

        return results

    def visualise(self, results):
        """
        Visualise analysis results with specialized plots for STDP analysis.

        1) Check if there are any results to visualise
        2) Create a Plotter instance for visualisation
        3) Process PRC results with specialized handling:
           a. Create directories for PRC visualisations
           b. Separate pre and post stimulation results
           c. Generate specialized PRC plots
           d. Handle case where no valid PRCs are available
        4) Define result types for different categories:
           a. Per-channel results (FFT, ISI, firing rate, etc.)
           b. Pairwise results (cross-correlation, Granger causality, statistical tests)
           c. Embedding results (UMAP, PCA with pre/post grouping)
        5) Process per-channel results with batch plotting
        6) Process pairwise results with batch plotting
        7) Create embeddings directory for specialized visualisations
        8) Process embedding results with appropriate plotting methods
        9) Handle any remaining result types individually

                    Parameters:
                        :parameter results: list - List of dictionaries containing analysis results

                    Returns:
                        :return: None
        """
        if not results:
            print("No results to visualise")
            return

        plotter = Plotter(self.results_dir)

        # Check if we have any valid PRC results
        prc_results = [r for r in results if r['type'] == 'prc']
        if not prc_results:
            print("No valid PRC results to visualise")

            # Create a dummy plot for no PRCs
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No valid PRC results computed", ha='center', va='center', fontsize=14)
            plt.axis('off')
            plt.savefig(self.results_dir / 'no_valid_prcs.png', dpi=300, bbox_inches='tight')
            plt.close()

        if prc_results:
            # Create a general dir for PRC results
            prc_dir = self.results_dir / 'prc'
            prc_dir.mkdir(parents=True, exist_ok=True)

            # Create pre and post directories
            pre_dir = prc_dir / 'pre'
            post_dir = prc_dir / 'post'
            pre_dir.mkdir(parents=True, exist_ok=True)
            post_dir.mkdir(parents=True, exist_ok=True)

            # Process PRC results to organize by pre/post
            for result in prc_results:
                data = result['data']
                if 'delta_phase' in data:
                    curves = data.get('delta_phase', {})
                    electrode = data.get('electrode', 'all')
                    phase_bins = data.get('phase_bins')

                    # Group curves by pre/post
                    pre_curves = {}
                    post_curves = {}
                    for key, curve in curves.items():
                        parts = key.split('_')
                        if len(parts) >= 3:
                            train_num = parts[1]
                            pre_post = parts[2]

                            if pre_post == 'pre':
                                pre_curves[key] = curve
                            elif pre_post == 'post':
                                post_curves[key] = curve

                    if pre_curves:
                        pre_data = {
                            'phase_bins': phase_bins,
                            'delta_phase': pre_curves,
                            'electrode': electrode
                        }
                        pre_plotter = Plotter(pre_dir)
                        try:
                            pre_plotter.plot('prc', pre_data)
                        except Exception as e:
                            print(f"Error plotting PRC for pre curves, electrode {electrode}: {e}")

                    if post_curves:
                        post_data = {
                            'phase_bins': phase_bins,
                            'delta_phase': post_curves,
                            'electrode': electrode
                        }

                        post_plotter = Plotter(post_dir)
                        try:
                            post_plotter.plot('prc', post_data)
                        except Exception as e:
                            print(f"Error plotting PRC for post curves, electrode {electrode}: {e}")

        # Group plotting for other result types
        per_channel_types = ['psd', 'auto_correlation', 'isi', 'firing_rate', 'powerlaw', 'fft', 'kernel_density', 'firing_rate_zscore']
        pairwise_types = ['cross_correlation', 'granger', 'statistical_tests']
        embedding_types = ['umap', 'pca', 'umap_pre_post', 'pca_pre_post']

        for result_type in per_channel_types:
            type_results = [r['data'] for r in results if r['type'] == result_type]
            if type_results:
                try:
                    plotter.plot(result_type, type_results, batch=True)
                except Exception as e:
                    print(f"Error plotting {result_type}: {e}")

        for result_type in pairwise_types:
            type_results = [r['data'] for r in results if r['type'] == result_type]
            if type_results:
                try:
                    plotter.plot(result_type, type_results, batch=True)
                except Exception as e:
                    print(f"Error plotting {result_type}: {e}")

        # Create a dir for embeddings
        embeddings_dir = self.results_dir / 'embeddings'
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        # embedding results
        for result_type in embedding_types:
            type_results = [r for r in results if r['type'] == result_type]
            if type_results:
                embeddings_plotter = Plotter(embeddings_dir)
                for result in type_results:
                    try:
                        embeddings_plotter.plot(result['type'], result['data'])
                        print(f"Successfully plotted {result['type']} embedding")
                    except Exception as e:
                        print(f"Error plotting {result['type']} embedding: {e}")

        # OLD. good for special cases
        for result in results:
            if result['type'] not in per_channel_types + pairwise_types + embedding_types + ['prc']:
                try:
                    plotter.plot(result['type'], result['data'])
                except Exception as e:
                    print(f"Error plotting {result['type']}: {e}")
