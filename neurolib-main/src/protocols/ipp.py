import json
import logging
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np

from src.analysis.correlation import compute_cross_correlation
from src.protocols.base import ProtocolBase
from src.visualisation.plotters import Plotter
from src.analysis.fft import compute_fft
from src.analysis.isi import compute_isi
from src.analysis.powerlaw import compute_powerlaw
from src.analysis.granger_causality import compute_granger_causality
from src.analysis.statistical_tools import compute_statistical_tests
from src.analysis.avalanche import compute_avalanche_distribution
from src.analysis.firing_rate import compute_firing_rates
from src.analysis.ipp_stats import (
    compute_firing_rate_zscore,
    compute_kernel_density,
    compute_inferential_statistics
)
from src.analysis.psd import compute_power_spectral_density
from src.analysis.utils import map_channels_to_mea, mask_stim_windows

logger = logging.getLogger(__name__)

class IPPProtocol(ProtocolBase):
    """
    Input-Pulse-Protocol (IPP) for analysing neural responses to specific input stimulation.

    1) Load raw neuronal recording data from CSV files
    2) Preprocess data with artifact removal and channel mapping:
       - Convert time to datetime format
       - Remove rows with missing values
       - Apply MAD-based noise reduction
       - Apply stimulus window masking for specific channels
       - Map electrode numbers to MEA channels
    3) Analyse neuronal activity with multiple metrics:
       - Compute channel-specific metrics (FFT, ISI, firing rates, etc.)
       - Calculate Z-scores and kernel density estimations
       - Perform statistical tests between channel pairs
       - Generate inferential statistics across channels
    4) Visualise results with specialized plots highlighting channels of interest
    """

    def load(self) -> pd.DataFrame:
        """
        Load raw neuronal recording data from CSV files in the raw directory.

        1) Find all CSV files in the raw directory, excluding stats.csv files
        2) Validate that at least one CSV file exists
        3) Read each CSV file with datetime parsing for the Time column
        4) Concatenate all DataFrames into a single DataFrame
        5) Return the combined DataFrame

                    Returns:
                        :return data: pd.DataFrame - Combined DataFrame with neuronal recording data
        """
        # 1) find CSVs no stats.csv -- DEPRECATED
        csvs = [p for p in Path(self.raw_dir).glob("*.csv") if not p.name.endswith("stats.csv")]
        if not csvs:
            raise FileNotFoundError(f"No CSVs in {self.raw_dir}")

        # rd, concat then return
        dfs = [pd.read_csv(p, parse_dates=['Time']) for p in csvs]
        return pd.concat(dfs, ignore_index=True)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-process raw neuronal recording data by cleaning and formatting.

        1) Check if input DataFrame is empty and return early if so
        2) Convert Time column to datetime format with UTC normalisation
        3) Remove rows with missing values in essential columns
        4) Apply MAD-based noise reduction:
           a. Calculate median absolute deviation (MAD)
           b. Convert MAD to standard deviation equivalent
           c. Set threshold based on configuration
           d. Remove amplitude values exceeding the threshold
        5) Apply stimulus window masking if enabled:
           a. Extract stimulation channels from experiment parameters
           b. Map electrode indices to MEA channels
           c. Optionally drop stimulation channels
        6) Store original channel numbers before any filtering
        7) Return the preprocessed DataFrame

                    Parameters:
                        :parameter df: pd.DataFrame - Raw neuronal recording data

                    Returns:
                        :return df: pd.DataFrame - Cleaned and preprocessed DataFrame
        """
        if df.empty:
            return df

        # 1) Unify time
        df['Time'] = pd.to_datetime(df['Time'], format='mixed', utc=True).dt.tz_convert(None)
        df = df.dropna(subset=['Time','Amplitude','channel'])

        # 2) Noise estimation
        mad = np.median(np.abs(df['Amplitude'] - np.median(df['Amplitude'])))
        sigma = mad / 0.6745
        thresh = self.cfg.get('ipp_artefact_thresh', 8.0) * sigma

        # 3) Mask out big artefacts
        df = df[np.abs(df['Amplitude']) < thresh]

        # 4) Apply stim window masking if enabled
        params = self.cfg.get('experiment_params', {})
        mask_stim = params.get('mask_stim', False)

        if mask_stim:
            stim_params = params.get('stim_params', [])
            stim_channels = []

            if stim_params:
                stim_channels = [param.get('index') for param in stim_params if 'index' in param]

            # Map electrode MEA channels
            mapped_stim_channels = map_channels_to_mea(stim_channels) if stim_channels else []

            # Drop all stim channels
            drop_stim_channels = params.get('drop_stim_channels', False)
            if drop_stim_channels and mapped_stim_channels:
                df = df[~df['channel'].isin(mapped_stim_channels)]

        # Store original numbers before any filtering
        df['original_channel'] = df['channel']

        return df.reset_index(drop=True)

    def analyse(self, df: pd.DataFrame, sample_mode=False) -> list:
        """
        Analyse pre-processed neuronal data using multiple analysis tools.

        1) Init empty results list
        2) Check if preprocessed data is empty and return early if so
        3) Extract unique channels from the dataset
        4) Identify channels of interest (COI) from experiment parameters
        5) For each channel, perform the following analyses:
           a. Fast Fourier Transform (FFT) to analyse frequency components
           b. Inter-Spike Interval (ISI) analysis to examine spike timing
           c. Neuronal avalanche detection to identify cascading activity
           d. Power law fitting to analyse statistical properties
           e. Firing rate calculation to measure neural activity levels
           f. Z-score and kernel density estimation for firing rate analysis
           g. Power Spectral Density (PSD) analysis for frequency distribution
        6) For pairs of channels, perform the following analyses:
           a. Granger causality testing to identify predictive relationships
           b. Statistical tests to compare neural activity
           c. Cross-correlation analysis to detect relationships
        7) Generate raster plot data for visualization
        8) Compute inferential statistics across all channels
        9) Return the list of all analysis results

                    Parameters:
                        :parameter df: pd.DataFrame - Preprocessed neuronal data
                        :parameter sample_mode: bool - If True, use reduced data for faster processing

                    Returns:
                        :return results: list - List of dictionaries containing analysis results
        """
        results = []
        if df.empty:
            return results

        # Same as preproc. Works, alternative would be better
        channels = df['original_channel'].unique()

        # Get electrodes from experiment parameters
        params = self.cfg.get('experiment_params', {})
        electrodes = params.get('electrodes', [])

        # If electrodes not specified, try to get from stim_params
        if not electrodes:
            stim_params = params.get('stim_params', [])
            if stim_params:
                electrodes = [param.get('index') for param in stim_params if 'index' in param]

        # Map electrode numbers from 1-32 base to 1-128 base for MEA
        mapped_electrodes = map_channels_to_mea(electrodes) if electrodes else []

        print(f"DEBUG: electrodes (original): {electrodes}")
        print(f"DEBUG: electrodes (mapped): {mapped_electrodes}")

        # Use mapped electrode numbers for filtering
        coi = mapped_electrodes if mapped_electrodes else list(channels)

        for ch in channels:
            df_ch = df[df['original_channel'] == ch]

            # FFT
            fft_res = compute_fft(df_ch, ch,
                                  self.cfg.get('fft_resample_ms', 1),
                                  self.cfg.get('fft_freq_cap', 10.0))
            results.append({'type': 'fft', 'data': {
                'channel': ch,
                'freqs': fft_res['freqs'],
                'amplitudes': fft_res['amplitudes'],
                'is_coi': ch in coi
            }})

            # ISI
            isi_res = compute_isi(df_ch, ch)
            isi_res['is_coi'] = ch in coi
            results.append({'type': 'isi', 'data': isi_res})

            # Avalanche
            aval = compute_avalanche_distribution(
                isi_res.get('isi', []),
                self.cfg.get('avalanche_threshold_s', 0.1)
            )
            has_aval = (isinstance(aval, np.ndarray) and aval.size > 0) or \
                       (not isinstance(aval, np.ndarray) and len(aval) > 0)
            if has_aval:
                results.append({'type': 'avalanche',
                                'data': {
                                    'channel': ch,
                                    'avalanches': aval,
                                    'is_coi': ch in coi
                }})

            # Powerlaw
            pl_res = compute_powerlaw(df_ch, ch)
            if 'error' not in pl_res:
                pl_res['is_coi'] = ch in coi
                results.append({'type': 'powerlaw', 'data': pl_res})

            # Firing rate
            fr_res = compute_firing_rates(df_ch,
                                          self.cfg.get('firing_rate_threshold', 0.0),
                                          self.cfg.get('firing_rate_bin_size_s', 1.0))
            fr_res['is_coi'] = ch in coi
            results.append({'type': 'firing_rate', 'data': fr_res})

            # Z-score & KDE
            bin_size_s = self.cfg.get('bin_size_s', 1.0)
            # Ensure bin_size_s is not too large compared to the data duration
            data_duration = (df_ch['Time'].max() - df_ch['Time'].min()).total_seconds()
            if data_duration > 0:
                # Use at most 1/10 of the data duration as bin size to ensure enough bins
                bin_size_s = min(bin_size_s, data_duration / 10)

                # Compute Z-score
                z_res = compute_firing_rate_zscore(df_ch, ch, bin_size_s)
                if 'error' not in z_res:
                    z_res['is_coi'] = ch in coi
                    results.append({'type': 'firing_rate_zscore', 'data': z_res})

                    # Compute KDE
                    kd_res = compute_kernel_density(df_ch, ch, bin_size_s)
                    if 'error' not in kd_res:
                        kd_res['is_coi'] = ch in coi
                        results.append({'type': 'kernel_density', 'data': kd_res})

            # PSD
            psd_res = compute_power_spectral_density(
                df=df,
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

        # Channel-pair metrics
        for pre, post in combinations(channels, 2):
            df_pre_post = df[df['original_channel'].isin([pre, post])]

            gc = compute_granger_causality(df_pre_post, pre, post,
                                           self.cfg.get('gc_resample_interval', '1s'),
                                           self.cfg.get('gc_max_lag', 5))
            if 'error' not in gc:
                gc['is_coi_pair'] = pre in coi and post in coi
                results.append({'type': 'granger', 'data': gc})

            st = compute_statistical_tests(df_pre_post, pre, post)
            if 'error' not in st:
                st['is_coi_pair'] = pre in coi and post in coi
                results.append({'type': 'statistical_tests', 'data': st})

            # c) Cross-correlation
            cc_res = compute_cross_correlation(
                df=df_pre_post,
                channel1=pre,
                channel2=post,
            )

            if 'error' not in cc_res:
                results.append({
                    'type': 'cross_correlation',
                    'data': {
                        'channel1': pre,
                        'channel2': post,
                        'lags': cc_res['lags'],
                        'correlation': cc_res['correlation']
                    }
                })

        # Raster plot
        raster = self._generate_raster_data(df)
        if raster:
            results.append(raster)

        # Inferential stats
        inf = compute_inferential_statistics(df, self.cfg.get('bin_size_s', 1.0))
        if 'error' not in inf:
            results.append({'type': 'inferential_statistics', 'data': inf})

        return results

    def visualise(self, results: list) -> None:
        """
        Visualise analysis results using appropriate plotting methods.

        1) Check if there are any results to visualise
        2) Create a Plotter instance for visualisation
        3) Define result types for different categories:
           a. Per-channel results (FFT, ISI, firing rate, etc.)
           b. Pairwise results (cross-correlation, Granger causality, statistical tests)
        4) Process per-channel results with batch plotting
        5) Process pairwise results with batch plotting
        6) Process other result types individually
        7) Handle any errors that occur during plotting

                    Parameters:
                        :parameter results: list - List of dictionaries containing analysis results

                    Returns:
                        :return: None
        """
        if not results:
            logger.info("IPPProtocol: no results to visualize.")
            return

        plotter = Plotter(self.results_dir)

        # Updated batch processor
        per_channel_types = ['psd', 'auto_correlation', 'isi', 'firing_rate', 'powerlaw', 'fft', 'kernel_density', 'firing_rate_zscore']
        pairwise_types = ['cross_correlation', 'granger', 'statistical_tests']

        for result_type in per_channel_types:
            type_results = [r['data'] for r in results if r['type'] == result_type]
            if type_results:
                plotter.plot(result_type, type_results, batch=True)

        for result_type in pairwise_types:
            type_results = [r['data'] for r in results if r['type'] == result_type]
            if type_results:
                plotter.plot(result_type, type_results, batch=True)

        # OLD
        for res in results:
            if res['type'] not in per_channel_types + pairwise_types:
                try:
                    plotter.plot(res['type'], res['data'])
                except Exception as e:
                    logger.error(f"Plot failure [{res['type']}]: {e}")
