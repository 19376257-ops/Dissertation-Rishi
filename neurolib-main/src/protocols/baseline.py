import pandas as pd
import logging
from itertools import combinations

from src.protocols.base import ProtocolBase
from src.visualisation.plotters import Plotter  # Central plotting dispatcher
from src.analysis.fft import compute_fft  # Fourier‐transform tool
from src.analysis.isi import compute_isi  # Inter-Spike Interval tool
from src.analysis.avalanche import compute_avalanche_distribution  # Avalanche tool
from src.analysis.granger_causality import compute_granger_causality  # Granger causality tool
from src.analysis.firing_rate import compute_firing_rates  # Firing rate tool
from src.analysis.powerlaw import compute_powerlaw  # Power-law tool
from src.analysis.statistical_tools import compute_statistical_tests  # Statistical tools
from src.analysis.psd import compute_power_spectral_density # Power_spectral_density tool
from src.analysis.correlation import (
    compute_cross_correlation,
    compute_auto_correlation
)
logger = logging.getLogger(__name__)  # Module‐level logger


class BaselineProtocol(ProtocolBase):
    """
    Baseline protocol for processing raw neuronal recordings without specific experiment parameters.

    1) Preprocess raw data by cleaning and formatting
    2) Analyse neural activity using multiple analysis tools:
       - Fast Fourier Transform (FFT)
       - Inter-Spike Interval (ISI) analysis
       - Neuronal avalanche detection
       - Firing rate calculation
       - Power law fitting
       - Statistical tests between channels
       - Power Spectral Density (PSD) analysis
       - Cross-correlation and auto-correlation analysis
       - Granger causality testing
    3) Visualise results using appropriate plotting methods
    """
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-process raw neuronal recording data by cleaning and formatting.

        1) Log diagnostic information about the input data (ONLY IN BASELINE)
        2) Create a copy of the input DataFrame to avoid modifying the original
        3) Convert Time column to datetime format with UTC normalisation
        4) Remove rows with missing values in essential columns (Time, channel, Amplitude)
        5) Log the shape of the cleaned DataFrame
        6) Return the preprocessed DataFrame

                    Parameters:
                        :parameter data: pd.DataFrame - Raw neural recording data

                    Returns:
                        :return df: pd.DataFrame - Cleaned and preprocessed DataFrame
        """
        print(f"DEBUG: preprocess called with data shape: {data.shape}")
        print(f"DEBUG: data columns: {data.columns.tolist()}")
        print(f"DEBUG: data head: {data.head()}")

        # 1) Work on a copy
        df = data.copy()

        # 2) Convert 'Time' to datetimec - Patched
        df['Time'] = pd.to_datetime(df['Time'], format='mixed', utc=True).dt.tz_convert(None)

        # 3) Drop rows missing essential columns
        df = df.dropna(subset=['Time', 'channel', 'Amplitude'])
        print(f"DEBUG: after dropping NA, df shape: {df.shape}")

        # 4) Return the cleaned DataFrame
        return df

    def analyse(self, preproc: pd.DataFrame, sample_mode=False) -> list:
        """
        Analyse pre-processed neuronal data using multiple analysis tools.

        1) Initlise empty results list
        2) Check if preprocessed data is empty and return early if so
        3) Extract unique channels from the dataset
        4) For each channel, perform the following analyses:
           a. Fast Fourier Transform (FFT) to analyse frequency components
           b. Inter-Spike Interval (ISI) analysis to examine spike timing
           c. Neuronal avalanche detection to identify cascading activity
           d. Firing rate calculation to measure neural activity levels
           e. Power law fitting to analyse statistical properties
           f. Power Spectral Density (PSD) analysis for frequency distribution
           g. Auto-correlation analysis to detect temporal patterns
        5) For pairs of channels, perform the following analyses:
           a. Statistical tests to compare neural activity
           b. Cross-correlation analysis to detect relationships
           c. Granger causality testing to identify predictive relationships
        6) Return the list of all analysis results

                    Parameters:
                        :parameter preproc: pd.DataFrame - Pre-processed neuronal data
                        :parameter sample_mode: bool - If True, use reduced data for faster processing

                    Returns:
                        :return results: list - List of dictionaries containing analysis results
        """
        results = []

        # 1) If there's no data, nothing to analyse
        if preproc.empty:
            logger.info("No data after preprocessing; skipping analysis.")
            print("DEBUG: preproc is empty, skipping analysis")
            return results

        # 2) Get list of unique channels in the dataset
        channels = preproc['channel'].unique()
        print(f"DEBUG: Found {len(channels)} unique channels: {channels}")

        # 3) Loop over channels, computing analyses for each
        for channel in channels:
            # a) Compute FFT
            fft_res = compute_fft(
                df=preproc,  # full DataFrame
                channel=channel,  # this channel's ID
                resample_interval=self.cfg.get('fft_resample_ms', 1),  # in milliseconds
                freq_cap=self.cfg.get('fft_freq_cap', 10.0)  # max freq to keep
            )

            # Tag and store the result for plotting
            results.append({
                'type': 'fft',  # Plotter looks for _plot_fft
                'data': {
                    'channel': channel,
                    'freqs': fft_res['freqs'],
                    'amplitudes': fft_res['amplitudes']
                }
            })

            # b) Compute ISI
            isi_res = compute_isi(
                df=preproc,
                channel=channel
            )

            results.append({
                'type': 'isi',
                'data': isi_res
            })

            # c) Compute Avalanche distribution
            if 'isi' in isi_res and len(isi_res['isi']) > 0:
                avalanche_res = compute_avalanche_distribution(
                    isi=isi_res['isi'],
                    threshold_s=self.cfg.get('avalanche_threshold_s', 0.1)
                )

                results.append({
                    'type': 'avalanche',
                    'data': {
                        'channel': channel,
                        'avalanches': avalanche_res
                    }
                })

            # d) Compute Powerlaw fit
            powerlaw_res = compute_powerlaw(
                df=preproc,
                channel=channel
            )

            # Only add if there was no error
            if 'error' not in powerlaw_res:
                results.append({
                    'type': 'powerlaw',
                    'data': powerlaw_res
                })

            # e) Compute Firing rates
            firing_rate_res = compute_firing_rates(
                df=preproc[preproc['channel'] == channel],
                threshold=self.cfg.get('firing_rate_threshold', 0.0),
                bin_size_s=self.cfg.get('firing_rate_bin_size_s', 1.0)
            )

            results.append({
                'type': 'firing_rate',
                'data': firing_rate_res
            })

            # f) Power spectral density
            psd_res = compute_power_spectral_density(
                df=preproc,
                channel=channel,
                fs=1e3,
                nperseg=self.cfg.get('psd_nperseg', None),
                noverlap=self.cfg.get('psd_noverlap', None)
            )
            results.append({
                'type': 'psd',
                'data': {
                    'channel': channel,
                    'freqs': psd_res['freqs'],
                    'psd': psd_res['psd'],
                }
            })

            # g) Auto-correlation
            ac_res = compute_auto_correlation(
                df=preproc,
                channel=channel
            )

            if 'error' not in ac_res:
                results.append({
                    'type': 'auto_correlation',
                    'data': {
                        'channel': channel,
                        'lags': ac_res['lags'],
                        'correlation': ac_res['correlation']
                    }
                })

        # 4) Compute analyses that require pairs of channels
        # Generate all possible pairs of channels
        channel_pairs = list(combinations(channels, 2))

        for pre_channel, post_channel in channel_pairs:
            # a) Compute Granger causality
            gc_res = compute_granger_causality(
                df=preproc,
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

            # b) Compute Statistical tests
            stats_res = compute_statistical_tests(
                df=preproc,
                pre_channel=pre_channel,
                post_channel=post_channel
            )

            # Only add if there was no error
            if 'error' not in stats_res:
                results.append({
                    'type': 'statistical_tests',
                    'data': stats_res
                })

            # c) Cross-correlation
            cc_res = compute_cross_correlation(
                df=preproc,
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
            else:
                print(f"DEBUG: Cross-correlation failed for channel {pre_channel} to {post_channel}")
                print(f"DEBUG: Cross-correlation error: {cc_res['error']}")
                print(f"DEBUG: Cross-correlation data: {cc_res['data']}")
                print(f"DEBUG: Cross-correlation pre_channel: {pre_channel}")

        # 5) Generate raster plot data
        raster_data = self._generate_raster_data(preproc)
        if raster_data:
            results.append(raster_data)

        # 6) Return all results
        return results

    def visualise(self, results: list) -> None:
        """
        Visualise analysis results using appropriate plotting methods.

        1) Check if there are any results to visualise
        2) Create a Plotter instance for visualisation
        3) For each result in the results list:
           a. Extract the result type
           b. Pass the result to the plotter for visualisaiton
           c. Handle any errors that occur during plotting
        4) Log the number of results visualised

                    Parameters:
                        :parameter results: list - List of dictionaries containing analysis results

                    Returns:
                        :return: None
        """
        if not results:
            logger.info("BaselineProtocol: no results to visualize.")
            return

        # Instantiate Plotter with the output folder
        plotter = Plotter(self.results_dir)

        # Group results by type
        per_channel_types = ['psd', 'auto_correlation', 'isi', 'firing_rate', 'powerlaw', 'fft', 'kernel_density', 'firing_rate_zscore']
        pairwise_types = ['cross_correlation', 'granger', 'statistical_tests']

        # Process per-channel results
        for result_type in per_channel_types:
            type_results = [r['data'] for r in results if r['type'] == result_type]
            if type_results:
                plotter.plot(result_type, type_results, batch=True)

        # Process pairwise results
        for result_type in pairwise_types:
            type_results = [r['data'] for r in results if r['type'] == result_type]
            if type_results:
                plotter.plot(result_type, type_results, batch=True)

        # Handle special cases (e.g., raster plots) individually
        for result in results:
            if result['type'] not in per_channel_types + pairwise_types:
                try:
                    plotter.plot(result['type'], result['data'])
                except Exception as e:
                    logger.error(f"Failed to plot {result['type']}: {e}")
