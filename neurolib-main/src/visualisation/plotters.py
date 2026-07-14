import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


class Plotter:
    def __init__(self, out_dir: Path, dpi: int = 300, fig_scale: float = 1.0, max_points: int = 5000):
        """
        Init the Plotter with output directory and display settings.

        1) Convert output directory to Path object
        2) Create output directory if it doesn't exist
        3) Store DPI, figure scale, and max points settings

                Parameters:
                    :parameter out_dir: Path
                    :parameter dpi: int
                    :parameter fig_scale: float
                    :parameter max_points: int
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.fig_scale = fig_scale
        self.max_points = max_points

    def plot(self, result_type: str, data, batch=False, phase=None):
        """
        Plot a single result or a batch of results with optional phase grouping.

        1) Store original output directory
        2) Create phase-specific directory if needed
        3) Process batch or single plot based on parameters
        4) Restore original output directory when done

                    Parameters:
                        :parameter result_type: str
                        :parameter data: dict or list
                        :parameter batch: bool
                        :parameter phase: str
        """
        original_out_dir = self.out_dir
        if phase and phase != 'all':
            phase_dir = self.out_dir / phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            self.out_dir = phase_dir

        try:
            if batch:
                batch_fn = getattr(self, f'_plot_{result_type}_batch', None)
                if not batch_fn:
                    for i, item in enumerate(data):
                        item_phase = item.get('phase', phase) if isinstance(item, dict) else phase
                        self.plot(result_type, item, phase=item_phase)
                    return

                if phase is None and all(isinstance(d, dict) and 'phase' in d for d in data):
                    phase_groups = {}
                    for item in data:
                        item_phase = item.get('phase', 'all')
                        if item_phase not in phase_groups:
                            phase_groups[item_phase] = []
                        phase_groups[item_phase].append(item)
                    for p, group_data in phase_groups.items():
                        self.plot(result_type, group_data, batch=True, phase=p)
                    return

                chunks = self.chunk_results(data)
                for i, chunk in enumerate(chunks):
                    batch_fn(chunk, page=i+1)
            else:
                fn = getattr(self, f'_plot_{result_type}', None)
                if not fn:
                    raise ValueError(f"No plot method for type '{result_type}'")
                fn(data)
        finally:
            # Restore original output directory
            self.out_dir = original_out_dir

    def group_plots_by_channel(self, results, result_type):
        """
        Group results by channel for batch processing.

        1) Create empty dictionary for grouped results
        2) Filter results by type
        3) Group data by result type

                    Parameters:
                        :parameter results: list
                        :parameter result_type: str

                    Returns:
                        :return grouped: dict
        """
        grouped = {}
        for result in results:
            if result['type'] == result_type:
                grouped.setdefault(result_type, []).append(result['data'])
        return grouped

    def group_plots_by_channel_pair(self, results, result_type):
        """
        Group results by channel pairs for batch processing.

        1) Create empty dictionary for grouped results
        2) Filter results by type
        3) Extract channel pairs from data
        4) Group data by channel pairs

                    Parameters:
                        :parameter results: list
                        :parameter result_type: str

                    Returns:
                        :return grouped: dict
        """
        grouped = {}
        for result in results:
            if result['type'] == result_type:
                data = result['data']
                ch1 = data.get('channel1') or data.get('pre_channel')
                ch2 = data.get('channel2') or data.get('post_channel')
                if ch1 and ch2:
                    key = (min(ch1, ch2), max(ch1, ch2))
                    grouped.setdefault(result_type, {}).setdefault(key, []).append(data)
        return grouped

    def chunk_results(self, results_list, chunk_size=8):
        """
        Split results into chunks for batch processing.

        1) Create empty list for chunks
        2) Split results into chunks of specified size
        3) Return list of chunks

                    Parameters:
                        :parameter results_list: list
                        :parameter chunk_size: int

                    Returns:
                        :return chunks: list
        """
        chunks = []
        for i in range(0, len(results_list), chunk_size):
            chunk = results_list[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

    def _plot_fft(self, data: dict):
        """
        Store FFT data for batch processing.

        1) Return without action as individual FFT plots are handled by batch method

                    Parameters:
                        :parameter data: dict
        """
        return

    def _plot_fft_batch(self, data_list, page=1):
        """
        Plot multiple FFT results in a grid layout.

        1) Calculate grid dimensions based on number of plots
        2) Create figure and axes
        3) Plot each FFT result in its own subplot
        4) Hide unused subplots
        5) Save figure to output directory

                    Parameters:
                        :parameter data_list: list
                        :parameter page: int
        """
        try:
            n_plots = len(data_list)
            n_cols = min(4, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
            axes = axes.flatten() if n_plots > 1 else [axes]

            for i, data in enumerate(data_list):
                if i < len(axes):
                    ch = data.get('channel', 'unknown')
                    try:
                        freqs, amps = data['freqs'], data['amplitudes']

                        axes[i].plot(freqs, amps, linewidth=1)
                        axes[i].set_title(f'Ch {ch}')
                        axes[i].set_xlabel('Frequency (Hz)')
                        axes[i].set_ylabel('Amplitude')
                        axes[i].grid(True, alpha=0.3)
                    except KeyError as e:
                        axes[i].text(0.5, 0.5, f'Missing data: {e}', ha='center', va='center')
                        axes[i].axis('off')

            for i in range(n_plots, len(axes)):
                axes[i].set_visible(False)

            fig.suptitle('Fast Fourier Transform', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.out_dir / f'fft_batch_page{page}.png', dpi=self.dpi)
            plt.close()
        except Exception as e:
            import traceback
            traceback.print_exc()
            plt.close('all')

    def _plot_avalanche(self, data: dict):
        """
        Plot neuronal avalanche distribution for a channel.

        1) Extract channel and phase information
        2) Store avalanche data in internal dictionary
        3) Calculate grid dimensions based on number of channels
        4) Create figure with appropriate title
        5) Plot histogram for each channel with optimised bins
        6) Apply log scaling where appropriate
        7) Save figure to output directory

                    Parameters:
                        :parameter data: dict
        """
        channel = data.get('channel', 'all')
        phase = data.get('phase', 'all')

        key = f"{channel}_{phase}"

        if not hasattr(self, '_avalanche_data'):
            self._avalanche_data = {}

        if phase not in self._avalanche_data:
            self._avalanche_data[phase] = {}

        self._avalanche_data[phase][channel] = {
            'avalanches': data['avalanches']
        }

        try:
            channels_data = self._avalanche_data[phase]

            n_channels = len(channels_data)
            n_cols = min(4, n_channels)
            n_rows = (n_channels + n_cols - 1) // n_cols

            fig = plt.figure(figsize=(5 * n_cols * self.fig_scale, 4 * n_rows * self.fig_scale))
            title = 'Neuronal Avalanche Distribution - All Channels'
            if phase != 'all':
                title += f' - Phase: {phase}'
            fig.suptitle(title, fontsize=16)

            for idx, (ch, ch_data) in enumerate(sorted(channels_data.items())):
                ax = fig.add_subplot(n_rows, n_cols, idx + 1)

                avalanches = ch_data['avalanches']
                if len(avalanches) > 1000:
                    n_bins = 50
                else:
                    n_bins = 'auto'

                counts, bins, _ = ax.hist(avalanches, bins=n_bins, alpha=0.7)
                ax.set_xlabel('Avalanche Size')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Channel {ch}')
                ax.grid(True)

                if np.any(counts > 0):
                    ax.set_yscale('log')

                if np.any(avalanches > 0):
                    ax.set_xscale('log')

            plt.tight_layout()
            filename = 'avalanche_all_channels'

            if phase != 'all':
                filename += f'_{phase}'
            plt.savefig(self.out_dir / f'{filename}.png', dpi=self.dpi, bbox_inches='tight')
            plt.close()

        except Exception as e:
            plt.close('all')

    def _plot_firing_rate(self, fr_data):
        """
        Plot firing rate data as a bar chart.

        1) Create figure with appropriate size
        2) Extract firing rate data and channel information
        3) Handle empty data case
        4) Create bar chart of firing rates
        5) Add title and labels
        6) Save figure to output directory

                    Parameters:
                        :parameter fr_data: dict or array
        """
        try:
            plt.figure(figsize=(10, 6))

            if isinstance(fr_data, dict) and 'firing_rate' in fr_data:
                fr = np.asarray(fr_data['firing_rate'])
                channel = fr_data.get('channel', 'all')
            else:
                fr = np.asarray(fr_data)
                channel = 'all'

            if fr.size == 0:
                plt.text(0.5, 0.5, 'No firing rate data available',
                         ha='center', va='center', transform=plt.gca().transAxes)
                plt.axis('off')
            else:
                plt.bar(np.arange(fr.size), fr, alpha=0.7)
                plt.xlabel('Bin Number')
                plt.ylabel('Firing Rate (Hz)')

            plt.title(f'Firing Rate - Channel {channel}')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f'firing_rate_ch{channel}.png'
            plt.savefig(self.out_dir / filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()

        except Exception as e:
            plt.close('all')

    def _plot_isi(self, isi_data):
        """
        Plot inter-spike interval distribution for one or more channels.

        1) Extract channel and phase information
        2) Store ISI data in internal dictionary
        3) Calculate grid dimensions based on number of channels
        4) Create figure with appropriate title
        5) For each channel:
           a. Extract ISI values and statistics
           b. Downsample large datasets if needed
           c. Create histogram of ISI values
           d. Add mean line if available
           e. Add labels and grid
        6) Save figure to output directory

                    Parameters:
                        :parameter isi_data: dict
        """
        if isinstance(isi_data, dict) and 'isi' in isi_data:
            channel = isi_data.get('channel', 'all')
            phase = isi_data.get('phase', 'all')

            key = f"{channel}_{phase}"

            if not hasattr(self, '_isi_data'):
                self._isi_data = {}

            if phase not in self._isi_data:
                self._isi_data[phase] = {}

            self._isi_data[phase][channel] = {
                'isi': isi_data['isi'],
                'mean_isi': isi_data.get('mean_isi', None),
                'std_isi': isi_data.get('std_isi', None)
            }

            try:
                channels_data = self._isi_data[phase]

                n_channels = len(channels_data)
                n_cols = min(4, n_channels)
                n_rows = (n_channels + n_cols - 1) // n_cols

                fig = plt.figure(figsize=(5 * n_cols * self.fig_scale, 4 * n_rows * self.fig_scale))
                title = 'ISI Distribution - All Channels'
                if phase != 'all':
                    title += f' - Phase: {phase}'
                fig.suptitle(title, fontsize=16)

                for idx, (ch, ch_data) in enumerate(sorted(channels_data.items())):
                    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                    isi_values = ch_data['isi']
                    mean_isi = ch_data.get('mean_isi', None)
                    std_isi = ch_data.get('std_isi', None)

                    if len(isi_values) > self.max_points:
                        step = len(isi_values) // self.max_points
                        isi_values = isi_values[::step]

                    if len(isi_values) > 1000:
                        n_bins = 40
                    else:
                        n_bins = min(40, max(10, len(isi_values) // 10))

                    ax.hist(isi_values, bins=n_bins, alpha=0.7)

                    if mean_isi is not None:
                        ax.axvline(mean_isi, color='r', linestyle='--',
                                   label=f'Mean: {mean_isi:.3f}s')

                    ax.set_xlabel('Inter-Spike Interval (s)')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Channel {ch}')
                    ax.grid(True)

                    if mean_isi is not None or std_isi is not None:
                        ax.legend()

                plt.tight_layout()
                filename = 'isi_all_channels'
                if phase != 'all':
                    filename += f'_{phase}'
                plt.savefig(self.out_dir / f'{filename}.png', dpi=self.dpi, bbox_inches='tight')
                plt.close()

            except Exception as e:
                plt.close('all')
        else:
            plt.figure(figsize=(10, 6))
            plt.hist(isi_data, bins=min(50, len(isi_data)))
            plt.title('ISI Distribution')
            plt.grid(True)
            plt.savefig(self.out_dir / 'isi.png', dpi=300, bbox_inches='tight')
            plt.close()



    def _plot_powerlaw(self, data: dict):
        """
        Store powerlaw data for later CSV generation and batch plotting.

        1) Validate input data format
        2) Extract powerlaw parameters (alpha, xmin, channel, phase)
        3) Init storage for powerlaw data if needed
        4) Store data for later batch processing and CSV generation

                    Parameters:
                        :parameter data: dict
        """
        print(f"Plotting powerlaw data for channel {data['channel']}")
        if not isinstance(data, dict) or 'fit' not in data:
            return

        alpha = data.get('alpha', None)
        xmin = data.get('xmin', None)
        ch = data.get('channel', 'all')
        phase = data.get('phase', 'all')

        # Store data for later CSV generation
        if not hasattr(self, '_powerlaw_data'):
            self._powerlaw_data = []

        self._powerlaw_data.append({
            'phase': phase,
            'channel': ch,
            'alpha': alpha,
            'xmin': xmin
        })

    def _plot_powerlaw_batch(self, data_list, page=1):
        """
        Process powerlaw results in batch and create a grid layout of plots.

        1) Calculate grid dimensions based on number of plots
        2) Create figure and axes for grid layout
        3) For each powerlaw result:
           a. Validate data format and extract parameters
           b. Store data for later CSV generation
           c. Plot PDF of data and fitted power-law
           d. Add labels, title, and legend
        4) Hide unused subplots
        5) Save figure to output dir

                    Parameters:
                        :parameter data_list: list
                        :parameter page: int
        """
        n_plots = len(data_list)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, data in enumerate(data_list):
            if i < len(axes):
                if not hasattr(self, '_powerlaw_data'):
                    self._powerlaw_data = []

                # Extract
                if not isinstance(data, dict) or 'fit' not in data:
                    error_msg = 'Invalid powerlaw data'
                    if isinstance(data, dict) and 'error' in data:
                        error_msg = data['error']
                    axes[i].text(0.5, 0.5, error_msg, ha='center', va='center')
                    axes[i].axis('off')
                    continue

                fit = data['fit']
                alpha = data.get('alpha', None)
                xmin = data.get('xmin', None)
                ch = data.get('channel', 'all')
                phase = data.get('phase', 'all')

                self._powerlaw_data.append({
                    'phase': phase,
                    'channel': ch,
                    'alpha': alpha,
                    'xmin': xmin
                })

                try:
                    fit.plot_pdf(ax=axes[i], linewidth=1.5, label='Data')
                    try:
                        fit.power_law.plot_pdf(ax=axes[i], linestyle='--', label=f'Fit α={alpha:.2f}')
                    except Exception:
                        pass

                    axes[i].set_title(f'Power‐Law Fit (Ch {ch})')
                    axes[i].set_xlabel('Value')
                    axes[i].set_ylabel('Probability Density')
                    axes[i].legend(fontsize='small')
                    axes[i].grid(True, alpha=0.3)
                except ValueError as e:
                    # Handle case where data has no positive values
                    print(f"Warning: Could not plot powerlaw for channel {ch}: {e}")
                    axes[i].text(0.5, 0.5, f'Cannot plot: {e}', ha='center', va='center')
                    axes[i].axis('off')

        # Hides unused plots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle('Power-Law Fits', fontsize=16)
        try:
            plt.tight_layout()
        except ValueError as e:
            print(f"Warning: Could not apply tight_layout: {e}")

        try:
            plt.savefig(self.out_dir / f'powerlaw_batch_page{page}.png', dpi=self.dpi)
        except ValueError as e:
            if "Data has no positive values, and therefore cannot be log-scaled" in str(e):
                print(f"Warning: Skipping powerlaw plot save due to log-scale error: {e}")
            else:
                # Re-raise if it's a different ValueError
                raise
        finally:
            plt.close()

        # Save consolidated CSV
        if hasattr(self, '_powerlaw_data') and self._powerlaw_data:
            try:
                df = pd.DataFrame(self._powerlaw_data)
                df.to_csv(self.out_dir / 'powerlaw_all_statistics.csv', index=False)
            except Exception as e:
                print(f"Error while saving consolidated powerlaw data to CSV: {e}")

    def _plot_granger(self, gc_data):
        """
        Store Granger causality data for later CSV generation and batch processing.

        1) Extract channel and phase information
        2) Extract p-values and significant lags from Granger causality test results
        3) Init data structures for storage if needed
        4) Store results in nested dir structure by phase and channel
        5) Handle legacy format data as fallback

                    Parameters:
                        :parameter gc_data: dict
        """
        if isinstance(gc_data, dict) and 'p_values' in gc_data:
            pre_channel = gc_data.get('pre_channel', 'unknown')
            post_channel = gc_data.get('post_channel', 'unknown')
            phase = gc_data.get('phase', 'all')
            p_values = gc_data['p_values']
            significant_lags = gc_data.get('significant_lags', [])

            if not hasattr(self, '_granger_data'):
                self._granger_data = {}

            if phase not in self._granger_data:
                self._granger_data[phase] = {}

            if pre_channel not in self._granger_data[phase]:
                self._granger_data[phase][pre_channel] = {}

            self._granger_data[phase][pre_channel][post_channel] = {
                'p_values': p_values,
                'significant_lags': significant_lags
            }

        else:
            try:
                pre_channel = gc_data.get('pre_channel', 'unknown')
                post_channel = gc_data.get('post_channel', 'unknown')
                phase = gc_data.get('phase', 'all')

                lags = list(gc_data.keys())
                pvals = [gc_data[lag][0]['ssr_chi2test'][1] for lag in lags]

                significant_lags = [lag for i, lag in enumerate(lags) if pvals[i] < 0.05]

                if not hasattr(self, '_granger_data'):
                    self._granger_data = {}

                if phase not in self._granger_data:
                    self._granger_data[phase] = {}

                if pre_channel not in self._granger_data[phase]:
                    self._granger_data[phase][pre_channel] = {}

                p_values = {lag: pval for lag, pval in zip(lags, pvals)}

                self._granger_data[phase][pre_channel][post_channel] = {
                    'p_values': p_values,
                    'significant_lags': significant_lags
                }
            except Exception as e:
                print(f"Error handling old format Granger causality data: {e}")

    def _plot_granger_batch(self, data_list, page=1):
        """
        Process multiple Granger causality results in batch and generate CSV reports.

        1) Process each Granger causality result individually
        2) Compile statistics across all results:
           a. Create data structures for summary and detailed statistics
           b. Iterate through all phases and channel pairs
           c. Calculate summary statistics (min, max, mean p-values)
           d. Record detailed p-values for each lag
        3) Generate and save CSV reports:
           a. Summary statistics CSV with significant lags
           b. Detailed CSV with all p-values and significance flags

                    Parameters:
                        :parameter data_list: list
                        :parameter page: int
        """
        for data in data_list:
            self._plot_granger(data)

        if hasattr(self, '_granger_data'):
            try:
                all_stats_data = {
                    'phase': [],
                    'pre_channel': [],
                    'post_channel': [],
                    'significant_lags': [],
                    'min_p_value': [],
                    'max_p_value': [],
                    'mean_p_value': []
                }

                all_detailed_data = {
                    'phase': [],
                    'pre_channel': [],
                    'post_channel': [],
                    'lag': [],
                    'p_value': [],
                    'significant': []
                }

                for phase in self._granger_data:
                    for pre_channel in self._granger_data[phase]:
                        post_channels_data = self._granger_data[phase][pre_channel]

                        for post_ch, post_ch_data in sorted(post_channels_data.items()):
                            all_stats_data['phase'].append(phase)
                            all_stats_data['pre_channel'].append(pre_channel)
                            all_stats_data['post_channel'].append(post_ch)
                            all_stats_data['significant_lags'].append(str(post_ch_data['significant_lags']))

                            p_values_list = list(post_ch_data['p_values'].values())
                            all_stats_data['min_p_value'].append(min(p_values_list) if p_values_list else None)
                            all_stats_data['max_p_value'].append(max(p_values_list) if p_values_list else None)
                            all_stats_data['mean_p_value'].append(
                                sum(p_values_list) / len(p_values_list) if p_values_list else None)

                            for lag, p_value in sorted(post_ch_data['p_values'].items()):
                                all_detailed_data['phase'].append(phase)
                                all_detailed_data['pre_channel'].append(pre_channel)
                                all_detailed_data['post_channel'].append(post_ch)
                                all_detailed_data['lag'].append(lag)
                                all_detailed_data['p_value'].append(p_value)
                                all_detailed_data['significant'].append(lag in post_ch_data['significant_lags'])

                stats_df = pd.DataFrame(all_stats_data)
                stats_df.to_csv(self.out_dir / 'granger_all_statistics.csv', index=False)
                detailed_df = pd.DataFrame(all_detailed_data)
                detailed_df.to_csv(self.out_dir / 'granger_all_detailed.csv', index=False)

            except Exception as e:
                print(f"Error while saving consolidated Granger causality data to CSV: {e}")

    def _plot_raster(self, data: dict):
        """
        Create a raster plot showing spike times across multiple channels.

        1) Extract spike times and channel information from data
        2) Validate that required data is present
        3) Create figure with appropriate size
        4) Plot spike times as scatter points
        5) Add title, labels, and grid
        6) Save figure to output directory

                    Parameters:
                        :parameter data: dict - Dictionary containing 'times' and 'channels' arrays
        """
        times = data.get('times')
        channels = data.get('channels')
        if times is None or channels is None:
            raise ValueError("Raster data must contain 'times' and 'channels' arrays")

        plt.figure(figsize=(10, 6))
        plt.scatter(times, channels, s=2, alpha=0.7)
        plt.title('Raster Plot of Detected Spikes')
        plt.xlabel('Time')
        plt.ylabel('Channel')
        plt.grid(True, alpha=0.3)

        out_path = self.out_dir / 'raster.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_firing_rate_zscore(self, data: dict):
        """
        Plot histogram of firing rate Z-scores for a specific channel.

        1) Create figure with appropriate size
        2) Validate input data format and extract Z-scores
        3) Calculate appropriate number of bins based on data size
        4) Create histogram with color coding for channels of interest
        5) Add title, labels, and reference lines at Z=0, Z=+2, Z=-2
        6) Handle invalid or empty data case
        7) Save figure to output directory with channel-specific filename

                    Parameters:
                        :parameter data: dict - Dictionary containing Z-scores and channel information
        """
        plt.figure(figsize=(10, 6))

        if (isinstance(data, dict) and 'z_scores' in data and 'bin_centers' in data and 
            isinstance(data['z_scores'], (list, np.ndarray)) and len(data['z_scores']) > 0):
            channel = data.get('channel', 'all')
            z_scores = data['z_scores']
            is_channel_of_interest = data.get('is_channel_of_interest', False)

            n_bins = min(30, max(10, len(z_scores) // 5))
            plt.hist(z_scores, bins=n_bins, edgecolor='black', alpha=0.7,
                     color='red' if is_channel_of_interest else 'blue')

            plt.title(f'Z-Scores of Firing Rates - Channel {channel}' +
                      (' (Channel of Interest)' if is_channel_of_interest else ''))
            plt.xlabel('Z-score')
            plt.ylabel('Number of occurrences')

            plt.axvline(0, color='r', linestyle='--', label='Mean (Z=0)')
            plt.axvline(2, color='g', linestyle='--', label='Z=+2')
            plt.axvline(-2, color='g', linestyle='--')

            plt.legend()

        else:
            plt.text(0.5, 0.5, 'Invalid or empty Z-score data',
                     ha='center', va='center', transform=plt.gca().transAxes)

        plt.grid(True)

        if isinstance(data, dict) and 'channel' in data:
            plt.savefig(self.out_dir / f'zscore_ch{data["channel"]}.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.out_dir / 'zscore.png', dpi=300, bbox_inches='tight')

        plt.close()

    def _plot_firing_rate_zscore_batch(self, data_list, page=1):
        """
        Plot multiple firing rate Z-score histograms in a grid layout.

        1) Check if data list is empty and handle empty case
        2) Calculate grid dimensions based on number of plots
        3) Create figure and axes for grid layout
        4) For each Z-score result:
           a. Validate data format and extract Z-scores
           b. Calculate appropriate number of bins
           c. Create histogram with color coding for channels of interest
           d. Add title, labels, and reference lines
        5) Hide unused subplots
        6) Add overall title and save figure

                    Parameters:
                        :parameter data_list: list - List of dictionaries containing Z-score data
                        :parameter page: int - Page number for batch output
        """
        # Check if data_list is empty
        if not data_list:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No Z-score data available for batch plotting',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / f'zscore_batch_page{page}.png', dpi=self.dpi)
            plt.close()
            return

        n_plots = len(data_list)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, data in enumerate(data_list):
            if i < len(axes):
                if (isinstance(data, dict) and 'z_scores' in data and 
                    isinstance(data['z_scores'], (list, np.ndarray)) and len(data['z_scores']) > 0):
                    channel = data.get('channel', 'unknown')
                    z_scores = data['z_scores']
                    is_channel_of_interest = data.get('is_channel_of_interest', False)

                    n_bins = min(30, max(10, len(z_scores) // 5))
                    axes[i].hist(z_scores, bins=n_bins, edgecolor='black', alpha=0.7,
                                color='red' if is_channel_of_interest else 'blue')

                    axes[i].set_title(f'Ch {channel}' + 
                                    (' *' if is_channel_of_interest else ''))
                    axes[i].set_xlabel('Z-score')
                    axes[i].set_ylabel('Count')

                    axes[i].axvline(0, color='r', linestyle='--', label='Mean (Z=0)')
                    axes[i].axvline(2, color='g', linestyle='--', label='Z=+2')
                    axes[i].axvline(-2, color='g', linestyle='--')

                    axes[i].grid(True, alpha=0.3)

                    # Only add legend to first plot to save space
                    if i == 0:
                        axes[i].legend(fontsize='small')
                else:
                    axes[i].text(0.5, 0.5, 'Invalid or empty Z-score data', ha='center', va='center')
                    axes[i].axis('off')

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle('Z-Scores of Firing Rates', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.out_dir / f'zscore_batch_page{page}.png', dpi=self.dpi)
        plt.close()

    def _plot_firing_rate_zscore_combined(self, data: dict):
        """
        Plot Z-scores of firing rates for all channels in one figure.

        1) Validate input data format and handle empty case
        2) Extract channel data and identify channels of interest
        3) Filter out channels with invalid data
        4) Calculate grid dimensions based on number of valid channels
        5) Create figure and axes for grid layout
        6) For each valid channel:
           a. Extract Z-scores and channel information
           b. Calculate appropriate number of bins
           c. Create histogram with color coding for channels of interest
           d. Add title, labels, and reference lines
        7) Hide unused subplots
        8) Add overall title and save figure

                    Parameters:
                        :parameter data: dict - Dictionary containing Z-score data for multiple channels
        """
        if not isinstance(data, dict) or 'channels' not in data or not data['channels']:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No Z-score data available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / 'zscore_combined.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        channels_data = data['channels']
        channels_of_interest = data.get('channels_of_interest', [])

        # Use all channels, not just channels of interest
        all_channels = list(channels_data.keys())
        n_channels = len(all_channels)

        if n_channels == 0:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No channels available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / 'zscore_combined.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        # Filter out channels with invalid data
        valid_channels = []
        for channel in all_channels:
            channel_key = str(channel) if str(channel) in channels_data else channel
            channel_data = channels_data[channel_key]

            # Check if channel data has valid z_scores
            if (isinstance(channel_data, dict) and 
                'z_scores' in channel_data and 
                isinstance(channel_data['z_scores'], (list, np.ndarray)) and 
                len(channel_data['z_scores']) > 0):
                valid_channels.append(channel)

        # Update n_channels to count only valid channels
        n_channels = len(valid_channels)

        if n_channels == 0:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No valid Z-score data available for any channel',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / 'zscore_combined.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        # Calculate grid dimensions
        n_cols = min(3, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle('Z-Scores of Firing Rates for All Channels', fontsize=16)

        # Ensure axes is always a 2D array
        if n_channels == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        # Plot each valid channel
        for i, channel in enumerate(valid_channels):
            # Get channel data
            channel_key = str(channel) if str(channel) in channels_data else channel
            channel_data = channels_data[channel_key]
            is_channel_of_interest = channel_data.get('is_channel_of_interest',
                                                      False) or channel in channels_of_interest

            # Calculate subplot position
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            # Plot histogram
            z_scores = channel_data['z_scores']
            n_bins = min(30, max(10, len(z_scores) // 5))

            # Use red for channels of interest, blue for others
            color = 'red' if is_channel_of_interest else 'blue'

            ax.hist(z_scores, bins=n_bins, edgecolor='black', alpha=0.7, color=color)

            # Add a marker to the title for channels of interest
            title = f'Channel {channel}' + (' *' if is_channel_of_interest else '')
            ax.set_title(title)
            ax.set_xlabel('Z-score')
            ax.set_ylabel('Number of occurrences')

            ax.axvline(0, color='r', linestyle='--', label='Mean (Z=0)')
            ax.axvline(2, color='g', linestyle='--', label='Z=+2')
            ax.axvline(-2, color='g', linestyle='--')

            ax.grid(True)
            ax.legend()

        # Hide unused subplots
        for i in range(n_channels, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.savefig(self.out_dir / 'zscore_combined.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_firing_rate_zscore_scatter(self, data: dict):
        """
        Plot Z-scores of firing rates as a scatter plot over time.

        1) Validate input data format and handle empty case
        2) Extract channel data and identify channels of interest
        3) Create figure with appropriate size
        4) For each channel:
           a. Validate data format and extract Z-scores and bin centers
           b. Apply visual differentiation for channels of interest (color, marker, size, alpha)
           c. Plot Z-scores as scatter points over time
        5) Add title, labels, and reference lines
        6) Add legend and save figure

                    Parameters:
                        :parameter data: dict - Dictionary containing Z-score data for multiple channels
        """
        if not isinstance(data, dict) or 'channels' not in data or not data['channels']:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No Z-score data available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / 'zscore_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        channels_data = data['channels']
        channels_of_interest = data.get('channels_of_interest', [])

        # Use all channels
        all_channels = list(channels_data.keys())

        if not all_channels:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No channels available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / 'zscore_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        plt.figure(figsize=(15, 8))

        # Check if any valid channels with data exist
        valid_channels_exist = False

        # Plot each channel
        for channel in all_channels:
            # Get channel data
            channel_key = str(channel) if str(channel) in channels_data else channel
            channel_data = channels_data[channel_key]

            # Check if channel data has valid z_scores and bin_centers
            if (not isinstance(channel_data, dict) or 
                'z_scores' not in channel_data or 
                'bin_centers' not in channel_data or
                not isinstance(channel_data['z_scores'], (list, np.ndarray)) or 
                len(channel_data['z_scores']) == 0 or
                not isinstance(channel_data['bin_centers'], (list, np.ndarray)) or 
                len(channel_data['bin_centers']) == 0):
                continue

            is_channel_of_interest = channel_data.get('is_channel_of_interest',
                                                      False) or channel in channels_of_interest

            # Plot scatter
            z_scores = channel_data['z_scores']
            bin_centers = channel_data['bin_centers']

            # Use red for channels of interest, blue for others
            color = 'red' if is_channel_of_interest else 'blue'
            marker = 'o' if is_channel_of_interest else '.'
            size = 30 if is_channel_of_interest else 15
            alpha = 1.0 if is_channel_of_interest else 0.5

            # Add a marker to the label for channels of interest
            label = f'Channel {channel}' + (' *' if is_channel_of_interest else '')

            plt.scatter(bin_centers, z_scores, alpha=alpha, label=label, color=color, marker=marker, s=size)
            valid_channels_exist = True

        if not valid_channels_exist:
            plt.text(0.5, 0.5, 'No valid Z-score data available for any channel',
                     ha='center', va='center', transform=plt.gca().transAxes)
        else:
            plt.title('Z-Scores of Firing Rates Over Time for All Channels')
            plt.xlabel('Time (s)')
            plt.ylabel('Z-score')

            plt.axhline(0, color='r', linestyle='--', label='Mean (Z=0)')
            plt.axhline(2, color='g', linestyle='--', label='Z=+2')
            plt.axhline(-2, color='g', linestyle='--', label='Z=-2')

            plt.grid(True)
            plt.legend()

        plt.savefig(self.out_dir / 'zscore_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_firing_rate_zscore_bar(self, data: dict):
        """
        Plot Z-scores of firing rates as a bar graph.

        1) Validate input data format and handle empty case
        2) Extract channel data and identify channels of interest
        3) Create figure with appropriate size
        4) Calculate mean Z-score for each channel
        5) Sort channels by mean Z-score
        6) Create bar chart with color coding for channels of interest
        7) Add title, labels, and reference lines
        8) Add legend and save figure

                    Parameters:
                        :parameter data: dict - Dictionary containing Z-score data for multiple channels
        """
        if not isinstance(data, dict) or 'channels' not in data or not data['channels']:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No Z-score data available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / 'zscore_bar.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        channels_data = data['channels']
        channels_of_interest = data.get('channels_of_interest', [])

        # Use all channels
        all_channels = list(channels_data.keys())

        if not all_channels:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No channels available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / 'zscore_bar.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        plt.figure(figsize=(15, 8))

        # Calculate mean Z-score for each channel
        mean_z_scores = []
        channel_labels = []
        colors = []

        # Check if any valid channels with data exist
        valid_channels_exist = False

        for channel in all_channels:
            # Get channel data
            channel_key = str(channel) if str(channel) in channels_data else channel
            channel_data = channels_data[channel_key]

            # Check if channel data has valid z_scores
            if (not isinstance(channel_data, dict) or 
                'z_scores' not in channel_data or 
                not isinstance(channel_data['z_scores'], (list, np.ndarray)) or 
                len(channel_data['z_scores']) == 0):
                continue

            is_channel_of_interest = channel_data.get('is_channel_of_interest',
                                                      False) or channel in channels_of_interest

            # Calculate mean Z-score
            z_scores = channel_data['z_scores']
            mean_z_score = np.mean(z_scores)

            mean_z_scores.append(mean_z_score)

            # Add a marker to the label for channels of interest
            channel_labels.append(f'Channel {channel}' + (' *' if is_channel_of_interest else ''))

            # Use red for channels of interest, blue for others
            colors.append('red' if is_channel_of_interest else 'blue')

            valid_channels_exist = True

        if not valid_channels_exist or not mean_z_scores:
            plt.text(0.5, 0.5, 'No valid Z-score data available for any channel',
                     ha='center', va='center', transform=plt.gca().transAxes)
        else:
            # Plot bar graph
            plt.bar(channel_labels, mean_z_scores, alpha=0.7, color=colors)

            plt.title('Mean Z-Scores of Firing Rates for All Channels')
            plt.xlabel('Channel')
            plt.ylabel('Mean Z-score')

            plt.axhline(0, color='r', linestyle='--', label='Mean (Z=0)')
            plt.axhline(2, color='g', linestyle='--', label='Z=+2')
            plt.axhline(-2, color='g', linestyle='--', label='Z=-2')

            plt.grid(True)
            plt.legend()

        plt.savefig(self.out_dir / 'zscore_bar.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_descriptive_statistics(self, data: dict):
        """
        Simple plot of descriptive firing-rate statistics:
          - Left: overall stats (mean, median, std, max, min)
          - Right: mean firing rate per channel
        """
        if not isinstance(data, dict):
            plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5, 'Invalid data format', ha='center', va='center')
            plt.savefig(self.out_dir / 'descriptive_statistics.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        overall = {
            'Mean': data.get('mean_firing_rate', 0),
            'Median': data.get('median_firing_rate', 0),
            'Std Dev': data.get('std_firing_rate', 0),
            'Max': data.get('max_firing_rate', 0),
            'Min': data.get('min_firing_rate', 0)
        }

        # Extract channel means
        ch_stats = data.get('channel_stats', [])
        channels = [ch['channel'] for ch in ch_stats]
        means = [ch['mean_amplitude'] if 'mean_amplitude' in ch else ch.get('mean', 0)
                 for ch in ch_stats]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Descriptive Statistics of Firing Rates', fontsize=14)

        ax1.bar(overall.keys(), overall.values(), alpha=0.7)
        ax1.set_ylabel('Firing Rate (Hz)')
        ax1.set_title('Overall')
        ax1.grid(True, alpha=0.3)

        ax2.bar(channels, means, alpha=0.7, color='tab:green')
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Mean Firing Rate (Hz)')
        ax2.set_title('Per-Channel Means')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(self.out_dir / 'descriptive_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_inferential_statistics(self, data: dict):
        """
        Simplified plot of pairwise correlation:
          - Single bar chart of correlation coefficients for each channel pair
        """
        tests = data.get('pairwise_tests', [])
        # Build lists of correlations
        corrs = []
        for t in tests:
            corr = None
            if 'correlation' in t and 'error' not in t['correlation']:
                corr = t['correlation'].get('r', None)
            corrs.append(corr if corr is not None else 0)

        # If no data, show placeholder
        if not corrs:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, 'No correlation data', ha='center', va='center')
            plt.savefig(self.out_dir / 'inferential_statistics.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        # Plot correlations by index (generic x-axis)
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(corrs)), corrs, alpha=0.7)
        plt.title('Pairwise Channel Correlation')
        plt.xlabel('Channel Pair Index')
        plt.ylabel('Correlation Coefficient (r)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.out_dir / 'inferential_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_kernel_density(self, data: dict):
        # 1) Extract firing rates
        rates = data.get('firing_rates') if isinstance(data, dict) else None
        if rates is None:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, 'No KDE data', ha='center', va='center')
            plt.axis('off')
            plt.savefig(self.out_dir / 'kde_invalid.png', dpi=150, bbox_inches='tight')
            plt.close()
            return

        ch = data.get('channel', 'all')

        plt.figure(figsize=(8, 4))
        # Use scipy's gaussian_kde for KDE plotting
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(rates)
        xs = np.linspace(min(rates), max(rates), 500)
        ys = kde(xs)
        plt.plot(xs, ys, linewidth=1.5)
        plt.fill_between(xs, ys, alpha=0.3)

        # 3) Labels and grid
        plt.title(f'KDE of Firing Rates – Channel {ch}')
        plt.xlabel('Firing Rate (Hz)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)

        # 4) Save and close
        plt.savefig(self.out_dir / f'kde_ch{ch}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_kernel_density_batch(self, data_list, page=1):
        """Plot multiple KDE results in a grid layout."""
        n_plots = len(data_list)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, data in enumerate(data_list):
            if i < len(axes):
                ch = data.get('channel', 'unknown')
                rates = data.get('firing_rates')

                if rates is None or len(rates) == 0:
                    axes[i].text(0.5, 0.5, 'No KDE data', ha='center', va='center')
                    axes[i].axis('off')
                    continue

                # Use scipy's gaussian_kde for KDE plotting
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(rates)
                xs = np.linspace(min(rates), max(rates), 500)
                ys = kde(xs)
                axes[i].plot(xs, ys, linewidth=1.5)
                axes[i].fill_between(xs, ys, alpha=0.3)

                axes[i].set_title(f'Ch {ch}')
                axes[i].set_xlabel('Firing Rate (Hz)')
                axes[i].set_ylabel('Density')
                axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle('Kernel Density Estimation of Firing Rates', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.out_dir / f'kde_batch_page{page}.png', dpi=self.dpi)
        plt.close()

    def _plot_psd(self, data: dict):
        """
        Plot power spectral density - PSD
        """
        if not isinstance(data, dict) or 'freqs' not in data or 'psd' not in data:
            plt.figure(figsize=(8, 5))
            plt.text(0.5, 0.5, 'No PSD data available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / 'psd_invalid.png', dpi=self.dpi)
            plt.close()
            return

        ch = data.get('channel', 'unknown')
        freqs, psd = data['freqs'], data['psd']

        # Check if freqs and psd are valid arrays with data
        if not isinstance(freqs, (list, np.ndarray)) or not isinstance(psd, (list, np.ndarray)) or len(freqs) == 0 or len(psd) == 0:
            plt.figure(figsize=(8, 5))
            plt.text(0.5, 0.5, 'Invalid or empty PSD data',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / f'psd_ch{ch}.png', dpi=self.dpi)
            plt.close()
            return

        plt.figure(figsize=(8, 5))

        # Check if PSD has positive values for log scaling
        if np.any(psd > 0):
            try:
                plt.loglog(freqs, psd, label=f'Ch {ch}')
            except ValueError as e:
                print(f"Warning: Could not use loglog scale for PSD plot: {e}")
                plt.plot(freqs, psd, label=f'Ch {ch}')
        else:
            # Use linear scale in event of no positive values
            plt.plot(freqs, psd, label=f'Ch {ch}')
            plt.text(0.5, 0.9, 'Note: Linear scale (no positive values)', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=8, color='red')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title(f'PSD — Channel {ch}')
        plt.legend()
        plt.tight_layout()

        try:
            plt.savefig(self.out_dir / f'psd_ch{ch}.png', dpi=self.dpi)
        except ValueError as e:
            if "Data has no positive values, and therefore cannot be log-scaled" in str(e):
                print(f"Warning: Skipping PSD plot save due to log-scale error: {e}")
            else:
                # Re-raise if it's a different ValueError
                raise
        finally:
            plt.close()

    def _plot_psd_batch(self, data_list, page=1):
        """Plot multiple PSD results in a grid layout."""
        # Check if data_list is empty
        if not data_list:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No PSD data available for batch plotting',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / f'psd_batch_page{page}.png', dpi=self.dpi)
            plt.close()
            return

        # Calculate grid dimensions based on number of plots
        # The chunking to 8 plots per page is handled by the plot method
        n_plots = len(data_list)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, data in enumerate(data_list):
            if i < len(axes):
                ch = data.get('channel', 'unknown')

                # Check if data is valid
                if (not isinstance(data, dict) or 
                    'freqs' not in data or 
                    'psd' not in data or
                    not isinstance(data['freqs'], (list, np.ndarray)) or 
                    not isinstance(data['psd'], (list, np.ndarray)) or
                    len(data['freqs']) == 0 or 
                    len(data['psd']) == 0):
                    axes[i].text(0.5, 0.5, 'Invalid or empty PSD data', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
                    continue

                freqs, psd = data['freqs'], data['psd']

                # Check if PSD has positive values for log scaling
                if np.any(psd > 0):
                    try:
                        axes[i].loglog(freqs, psd)
                    except ValueError as e:
                        print(f"Warning: Could not use loglog scale for PSD plot: {e}")
                        axes[i].plot(freqs, psd)
                else:
                    # Use linear scale in event of no positive values
                    axes[i].plot(freqs, psd)
                    axes[i].text(0.5, 0.9, 'Note: Linear scale (no positive values)', ha='center', va='center', transform=axes[i].transAxes,
                                fontsize=8, color='red')

                axes[i].set_title(f'Ch {ch}')
                axes[i].set_xlabel('Frequency (Hz)')
                axes[i].set_ylabel('PSD')
                axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle('Power Spectral Density', fontsize=16)
        plt.tight_layout()

        try:
            plt.savefig(self.out_dir / f'psd_batch_page{page}.png', dpi=self.dpi)
        except ValueError as e:
            if "Data has no positive values, and therefore cannot be log-scaled" in str(e):
                print(f"Warning: Skipping PSD batch plot save due to log-scale error: {e}")
            else:
                # Re-raise if it's a different ValueError
                raise
        finally:
            plt.close()



    def _plot_auto_correlation(self, data: dict):
        """
        Plot autocorrelation
        """
        if not isinstance(data, dict) or 'lags' not in data or 'correlation' not in data:
            plt.figure(figsize=(8, 5))
            plt.text(0.5, 0.5, 'No autocorrelation data available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / 'autocorr_invalid.png', dpi=self.dpi)
            plt.close()
            return

        ch = data.get('channel', 'unknown')
        lags, corr = data['lags'], data['correlation']

        # Check if lags and correlation are valid arrays with data
        if not isinstance(lags, (list, np.ndarray)) or not isinstance(corr, (list, np.ndarray)) or len(lags) == 0 or len(corr) == 0:
            plt.figure(figsize=(8, 5))
            plt.text(0.5, 0.5, 'Invalid or empty autocorrelation data',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / f'autocorr_ch{ch}.png', dpi=self.dpi)
            plt.close()
            return

        plt.figure(figsize=(8, 5))
        plt.plot(lags, corr)
        plt.xlabel('Lag (ms)')
        plt.ylabel('Autocorrelation')
        plt.title(f'Autocorrelation — Channel {ch}')
        plt.tight_layout()
        plt.savefig(self.out_dir / f'autocorr_ch{ch}.png', dpi=self.dpi)
        plt.close()

    def _plot_auto_correlation_batch(self, data_list, page=1):
        """Plot multiple autocorrelation results in a grid layout."""
        # Check if data_list is empty
        if not data_list:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No autocorrelation data available for batch plotting',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / f'autocorr_batch_page{page}.png', dpi=self.dpi)
            plt.close()
            return

        n_plots = len(data_list)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, data in enumerate(data_list):
            if i < len(axes):
                ch = data.get('channel', 'unknown')

                # Check if data is valid
                if (not isinstance(data, dict) or 
                    'lags' not in data or 
                    'correlation' not in data or
                    not isinstance(data['lags'], (list, np.ndarray)) or 
                    not isinstance(data['correlation'], (list, np.ndarray)) or
                    len(data['lags']) == 0 or 
                    len(data['correlation']) == 0):
                    axes[i].text(0.5, 0.5, 'Invalid or empty autocorrelation data', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
                    continue

                lags, corr = data['lags'], data['correlation']

                axes[i].plot(lags, corr)
                axes[i].set_title(f'Ch {ch}')
                axes[i].set_xlabel('Lag (ms)')
                axes[i].set_ylabel('Autocorrelation')
                axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle('Autocorrelation Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.out_dir / f'autocorr_batch_page{page}.png', dpi=self.dpi)
        plt.close()

    def _plot_cross_correlation(self, data: dict):
        """
        Plot cross-correlation
        """
        if not isinstance(data, dict) or 'lags' not in data or 'correlation' not in data:
            plt.figure(figsize=(8, 5))
            plt.text(0.5, 0.5, 'No cross-correlation data available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / 'crosscorr_invalid.png', dpi=self.dpi)
            plt.close()
            return

        ch1 = data.get('channel1', 'unknown')
        ch2 = data.get('channel2', 'unknown')
        lags, corr = data['lags'], data['correlation']

        # Check if lags and correlation are valid arrays with data
        if not isinstance(lags, (list, np.ndarray)) or not isinstance(corr, (list, np.ndarray)) or len(lags) == 0 or len(corr) == 0:
            plt.figure(figsize=(8, 5))
            plt.text(0.5, 0.5, 'Invalid or empty cross-correlation data',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / f'crosscorr_ch{ch1}_ch{ch2}.png', dpi=self.dpi)
            plt.close()
            return

        plt.figure(figsize=(8, 5))
        plt.plot(lags, corr)
        plt.xlabel('Lag (ms)')
        plt.ylabel('Cross-correlation')
        plt.title(f'Cross-corr — Ch{ch1} vs Ch{ch2}')
        plt.tight_layout()
        plt.savefig(self.out_dir / f'crosscorr_ch{ch1}_ch{ch2}.png', dpi=self.dpi)
        plt.close()

    def _plot_cross_correlation_batch(self, data_list, page=1):
        """Plot multiple cross-correlation results in a grid layout."""
        # Check if data_list is empty
        if not data_list:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No cross-correlation data available for batch plotting',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(self.out_dir / f'crosscorr_batch_page{page}.png', dpi=self.dpi)
            plt.close()
            return

        n_plots = len(data_list)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, data in enumerate(data_list):
            if i < len(axes):
                ch1 = data.get('channel1', 'unknown')
                ch2 = data.get('channel2', 'unknown')

                # Check if data is valid
                if (not isinstance(data, dict) or 
                    'lags' not in data or 
                    'correlation' not in data or
                    not isinstance(data['lags'], (list, np.ndarray)) or 
                    not isinstance(data['correlation'], (list, np.ndarray)) or
                    len(data['lags']) == 0 or 
                    len(data['correlation']) == 0):
                    axes[i].text(0.5, 0.5, 'Invalid or empty cross-correlation data', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
                    continue

                lags, corr = data['lags'], data['correlation']

                axes[i].plot(lags, corr)
                axes[i].set_title(f'Ch{ch1} vs Ch{ch2}')
                axes[i].set_xlabel('Lag (ms)')
                axes[i].set_ylabel('Cross-correlation')
                axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle('Cross-Correlation Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.out_dir / f'crosscorr_batch_page{page}.png', dpi=self.dpi)
        plt.close()


    def _plot_statistical_tests(self, stats_data):
        """
        Store statistical test data for later CSV generation.
        """
        if not isinstance(stats_data, dict):
            print("Invalid statistical tests data format")
            return

        pre = stats_data.get('pre_channel', 'unknown')
        post = stats_data.get('post_channel', 'unknown')
        phase = stats_data.get('phase', 'all')
        condition = stats_data.get('condition', None)

        # Initialize data structures if they don't exist
        self._stats_data = getattr(self, '_stats_data', {})
        self._stats_data.setdefault(phase, {}).setdefault(pre, {})[post] = stats_data.copy()

    def _plot_statistical_tests_batch(self, data_list, page=1):
        """
        Process multiple statistical test results in batch.
        Generates a single consolidated CSV file with all statistical test data.
        """
        # Call _plot_statistical_tests for each item in data_list to collect data
        for data in data_list:
            self._plot_statistical_tests(data)

        # After processing all data, generate a single consolidated CSV file
        if hasattr(self, '_stats_data'):
            try:
                all_records = []

                # Define extractor function to extract data from statistical tests
                def extract_row(data, pre, post, phase):
                    row = {'phase': phase, 'pre_channel': pre, 'post_channel': post}
                    for test, keys in [
                        ('ks_test', ['statistic', 'p_value']),
                        ('pearson', ['correlation', 'p_value']),
                        ('t_test', ['statistic', 'p_value']),
                        ('mann_whitney', ['statistic', 'p_value'])
                    ]:
                        res = data.get(test, {})
                        if 'error' not in res:
                            for k in keys:
                                row[f'{test}_{k}'] = res.get(k)
                            if 'p_value' in res:
                                row[f'{test}_significant'] = res['p_value'] < 0.05

                    # Shapiro separately
                    shapiro = data.get('shapiro', {})
                    for side in ('pre_channel', 'post_channel'):
                        if side in shapiro and 'p_value' in shapiro[side]:
                            p = shapiro[side]['p_value']
                            row[f'shapiro_{side}_p'] = p
                            row[f'shapiro_{side}_sig'] = p < 0.05

                    return row

                # Collect all data from all phases and channels
                for phase in self._stats_data:
                    for pre in self._stats_data[phase]:
                        for post, post_data in self._stats_data[phase][pre].items():
                            all_records.append(extract_row(post_data, pre, post, phase))

                # Create a DataFrame from the collected data
                if all_records:
                    df = pd.DataFrame(all_records)

                    # Save consolidated CSV
                    df.to_csv(self.out_dir / 'statistical_tests_all.csv', index=False)

            except Exception as e:
                print(f"Error while saving consolidated statistical tests data to CSV: {e}")

    def _plot_prc(self, data: dict):
        """
        Plot PRCs for a given channel
        """
        # Check for new format with delta_phase
        if 'delta_phase' in data:
            bins = data.get('phase_bins')
            curves = data.get('delta_phase', {})
            electrode = data.get('electrode', 'all')

            if bins is None or len(bins) == 0 or curves is None or len(curves) == 0:
                plt.figure(figsize=(6, 4))
                plt.text(0.5, 0.5, 'Invalid PRC data', ha='center', va='center')
                plt.axis('off')
                plt.savefig(self.out_dir / 'prc_invalid.png', dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return

            # Get default colour cycle for prc plots - DEPRECATED
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']

            # Sort keys to ensure trains are plotted in order
            sorted_keys = sorted(curves.keys())

            # Check if we're plotting a single curve or multiple curves
            if len(sorted_keys) == 1:
                key = sorted_keys[0]
                curve = curves[key]

                if curve is not None and isinstance(curve, np.ndarray) and len(bins) == len(curve):
                    # Extract train number
                    parts = key.split('_')
                    if len(parts) >= 3:
                        train_num = parts[1]
                        pre_post = parts[2]

                        plt.figure(figsize=(10, 6))
                        plt.plot(bins, curve, linestyle='-', color=colors[0], label=f'Train {train_num} {pre_post.capitalize()}')
                        plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
                        plt.xlabel('Phase (0 -> 1)')
                        plt.ylabel('Normalized ΔISI')
                        plt.title(f'Phase-Response Curve – Train {train_num} {pre_post.capitalize()} – Electrode {electrode}')
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.tight_layout()

                        # Save
                        plt.savefig(self.out_dir / f'prc_train_{train_num}_{pre_post}_ch{electrode}.png', dpi=self.dpi, bbox_inches='tight')
                        plt.close()

                        return
            else:
                plt.figure(figsize=(12, 8))

                # Plot each curve
                for i, key in enumerate(sorted_keys):
                    curve = curves[key]
                    if curve is not None and isinstance(curve, np.ndarray) and len(bins) == len(curve):
                        # Extract train number and pre/post status
                        parts = key.split('_')
                        if len(parts) >= 3:
                            train_num = parts[1]
                            pre_post = parts[2]

                            # Customisation here. Play around
                            color_idx = int(train_num) % len(colors)
                            color = colors[color_idx]
                            linestyle = '--' if pre_post == 'pre' else '-'
                            plt.plot(bins, curve, linestyle=linestyle, color=color,
                                     label=f'Train {train_num} {pre_post.capitalize()}')

                plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
                plt.xlabel('Phase (0 -> 1)')
                plt.ylabel('Normalized ΔISI')
                plt.title(f'Phase-Response Curves – Electrode {electrode}')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()

                # Save the plot
                plt.savefig(self.out_dir / f'prc_ch{electrode}.png', dpi=self.dpi, bbox_inches='tight')
                plt.close()

                return

        # Backward compatibility
        bins = data.get('phase_bins')
        prc = data.get('prc') or data.get('prc_values')
        if bins is None or prc is None or not isinstance(prc, np.ndarray) or len(bins) != len(prc):
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, 'Invalid PRC data', ha='center', va='center')
            plt.axis('off')
            plt.savefig(self.out_dir / 'prc_invalid.png',
                        dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return

        ch = data.get('channel', 'all')
        pre_train_keys = []
        post_train_keys = []

        # ID pre and post train keys
        for key in data.keys():
            if key.startswith('train_') and not key.startswith('train_post_'):
                pre_train_keys.append(key)
            elif key.startswith('pre_train_') or key.startswith('prc_train_pre_'):
                pre_train_keys.append(key)
            elif key.startswith('post_train_') or key.startswith('prc_train_post_'):
                post_train_keys.append(key)

        # Save average PRC (for backward compatibility)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(bins, prc, marker='o', linestyle='-')
        ax.set_xlabel('Phase (0 -> 1)')
        ax.set_ylabel('Normalized ΔISI')
        ax.set_title(f'Phase-Response Curve (ch {ch})')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.out_dir / f'prc_ch{ch}.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()

        # Plot pre-stimulus trains (overlapped)
        if pre_train_keys:
            plt.figure(figsize=(12, 8))
            for idx, key in enumerate(pre_train_keys):
                train_prc = data.get(key)
                if train_prc is not None and isinstance(train_prc, np.ndarray) and len(bins) == len(train_prc):
                    plt.plot(bins, train_prc, linestyle='--', label=f'Pre-Stim Train {idx + 1}')

            plt.xlabel('Phase of Stimulus')
            plt.ylabel('Spike Adjustment (as % of ISI)')
            plt.title(f'Phase-Response Curves (PRC) for Pre-Stimulus Trains (ch {ch})')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            # Save
            plt.savefig(self.out_dir / f'prc_pre_trains_ch{ch}.png', dpi=self.dpi, bbox_inches='tight')
            plt.close()

        # Plot post-stimulus trains (overlapped)
        if post_train_keys:
            plt.figure(figsize=(12, 8))
            for idx, key in enumerate(post_train_keys):
                train_prc = data.get(key)
                if train_prc is not None and isinstance(train_prc, np.ndarray) and len(bins) == len(train_prc):
                    plt.plot(bins, train_prc, linestyle='-', label=f'Post-Stim Train {idx + 1}')

            plt.xlabel('Phase of Stimulus')
            plt.ylabel('Spike Adjustment (as % of ISI)')
            plt.title(f'Phase-Response Curves (PRC) for Post-Stimulus Trains (ch {ch})')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            # Save
            plt.savefig(self.out_dir / f'prc_post_trains_ch{ch}.png', dpi=self.dpi, bbox_inches='tight')
            plt.close()

    def _plot_prc_batch(self, data_list, page=1):
        """Plot multiple PRC results in a grid layout."""
        try:
            n_plots = len(data_list)
            n_cols = min(2, n_plots)  # Use 2 columns for PRCs as they need more space
            n_rows = (n_plots + n_cols - 1) // n_cols

            # Limit to 8 plots per page
            plots_per_page = 8
            start_idx = (page - 1) * plots_per_page
            end_idx = min(start_idx + plots_per_page, n_plots)
            current_data_list = data_list[start_idx:end_idx]
            current_n_plots = len(current_data_list)

            if current_n_plots == 0:
                return

            n_cols = min(2, current_n_plots)
            n_rows = (current_n_plots + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 6*n_rows))
            axes = axes.flatten() if current_n_plots > 1 else [axes]

            for i, data in enumerate(current_data_list):
                if i < len(axes):
                    ax = axes[i]

                    # Check for new format with delta_phase
                    if 'delta_phase' in data:
                        bins = data.get('phase_bins')
                        curves = data.get('delta_phase', {})
                        electrode = data.get('electrode', 'all')

                        if bins is None or len(bins) == 0 or curves is None or len(curves) == 0:
                            ax.text(0.5, 0.5, 'Invalid PRC data', ha='center', va='center')
                            ax.axis('off')
                            continue

                        # Get default colour cycle
                        prop_cycle = plt.rcParams['axes.prop_cycle']
                        colors = prop_cycle.by_key()['color']

                        # Sort keys to ensure trains are plotted in order
                        sorted_keys = sorted(curves.keys())

                        # Plot each curve
                        for j, key in enumerate(sorted_keys):
                            curve = curves[key]
                            if curve is not None and isinstance(curve, np.ndarray) and len(bins) == len(curve):
                                # Extract train number and pre/post status
                                parts = key.split('_')
                                if len(parts) >= 3:
                                    train_num = parts[1]
                                    pre_post = parts[2]

                                    # Use different colors for different trains
                                    color_idx = int(train_num) % len(colors)
                                    color = colors[color_idx]
                                    linestyle = '--' if pre_post == 'pre' else '-'
                                    ax.plot(bins, curve, linestyle=linestyle, color=color,
                                            label=f'Train {train_num} {pre_post.capitalize()}')

                        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
                        ax.set_xlabel('Phase (0 -> 1)')
                        ax.set_ylabel('Normalized ΔISI')
                        ax.set_title(f'PRC - Electrode {electrode}')
                        ax.grid(True, alpha=0.3)
                        ax.legend(fontsize='small')
                    else:
                        # Backward compatibility
                        bins = data.get('phase_bins')
                        prc = data.get('prc') or data.get('prc_values')
                        ch = data.get('channel', 'all')

                        if bins is None or prc is None or not isinstance(prc, np.ndarray) or len(bins) != len(prc):
                            ax.text(0.5, 0.5, 'Invalid PRC data', ha='center', va='center')
                            ax.axis('off')
                            continue

                        ax.plot(bins, prc, marker='o', linestyle='-')
                        ax.set_xlabel('Phase (0 -> 1)')
                        ax.set_ylabel('Normalized ΔISI')
                        ax.set_title(f'PRC - Channel {ch}')
                        ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for i in range(current_n_plots, len(axes)):
                axes[i].set_visible(False)

            fig.suptitle('Phase-Response Curves', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.out_dir / f'prc_batch_page{page}.png', dpi=self.dpi, bbox_inches='tight')
            plt.close()

            # If there are more plots, process the next page
            if end_idx < n_plots:
                self._plot_prc_batch(data_list, page=page+1)

        except Exception as e:
            print(f"Error in _plot_prc_batch: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')

    def _plot_embedding(self, data: dict, title_prefix: str):
        """
        PCA / UMAP plotter, uses experimental aliaser to save on lines.
        Creates 3D plots using the first three components of the embedding.
        """
        emb = data.get('embedding')
        if emb is None or len(emb) < 2:
            # fallback: invalid
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, 'Invalid embedding data', ha='center', va='center')
            plt.axis('off')
            plt.savefig(self.out_dir / f'{title_prefix.lower()}_embedding_invalid.png',
                        dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return

        # Check for at least 3 components before fitting
        n_components = emb.shape[1]
        if n_components < 3:
            print(f"Warning: {title_prefix} embedding has only {n_components} components. Need at least 3 for 3D plotting.")
            # Fall back to 2D plotting if we don't have enough components
            fig, ax = plt.subplots(figsize=(8, 6))
            conds = data.get('conditions', None)
            if conds is not None:
                uniq = sorted(set(conds))
                for u in uniq:
                    mask = [c == u for c in conds]
                    ax.scatter(emb[mask, 0], emb[mask, 1],
                               label=str(u), alpha=0.7, s=15)
                ax.legend(title='Condition', fontsize='small', loc='best')
            else:
                ax.scatter(emb[:, 0], emb[:, 1], alpha=0.7, s=15)

            ax.set_xlabel(f'{title_prefix} Component 1')
            ax.set_ylabel(f'{title_prefix} Component 2')
            ax.set_title(f'2D {title_prefix} Visualization')
            ax.grid(True, alpha=0.3)
        else:
            # 3D plotting
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            conditions = data.get('conditions', None)
            if conditions is not None:
                uniq = sorted(set(conditions))
                for u in uniq:
                    mask = [c == u for c in conditions]
                    ax.scatter(
                        emb[mask, 0],  # X-axis (Component 1)
                        emb[mask, 1],  # Y-axis (Component 2)
                        emb[mask, 2],  # Z-axis (Component 3)
                        label=str(u), alpha=0.7, s=15
                    )
                ax.legend(title='Condition', fontsize='small', loc='best')
            else:
                ax.scatter(
                    emb[:, 0],  # X-axis (Component 1)
                    emb[:, 1],  # Y-axis (Component 2)
                    emb[:, 2],  # Z-axis (Component 3)
                    alpha=0.7, s=15
                )

            ax.set_xlabel(f'{title_prefix} Component 1')
            ax.set_ylabel(f'{title_prefix} Component 2')
            ax.set_zlabel(f'{title_prefix} Component 3')
            ax.set_title(f'3D {title_prefix} Neural Manifold Visualization')

        # if PCA - show variance
        evr = data.get('explained_variance_ratio')
        if evr is not None and n_components >= 3:
            ax.set_title(f'3D {title_prefix} (PC1-3: '
                         f'{evr[0]:.2f}, {evr[1]:.2f}, {evr[2]:.2f})', fontsize=14)
        elif evr is not None:
            ax.set_title(f'2D {title_prefix} (PC1-2: '
                         f'{evr[0]:.2f}, {evr[1]:.2f})', fontsize=14)

        filename = f'{title_prefix.lower()}_embedding.png'
        plt.tight_layout()
        plt.savefig(self.out_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def _plot_phase_group_embedding(self, data_list, title_prefix):
        """
        Creates a plot with subplots for each phase in a phase group.
        """
        if not data_list:
            return

        # Extract group name from the first item
        group_name = data_list[0].get('group', 'unknown')

        # Get all unique phases
        phases = []
        for data in data_list:
            phase = data.get('phase')
            if phase and phase not in phases:
                phases.append(phase)

        if not phases:
            return

        n_phases = len(phases)
        n_cols = min(3, n_phases)
        n_rows = (n_phases + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
        fig.suptitle(f'{title_prefix} Embeddings for Phase Group: {group_name}', fontsize=16)

        # Create subplots for each phase
        for i, phase in enumerate(phases):
            phase_data = None
            for data in data_list:
                if data.get('phase') == phase:
                    phase_data = data
                    break

            if not phase_data:
                continue

            # Get the embedding
            emb = phase_data.get('embedding')
            if emb is None or len(emb) < 2:
                continue

            ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d' if emb.shape[1] >= 3 else None)

            # Plot
            conditions = phase_data.get('conditions', None)
            if conditions is not None:
                uniq = sorted(set(conditions))
                for u in uniq:
                    mask = [c == u for c in conditions]
                    if emb.shape[1] >= 3:
                        ax.scatter(
                            emb[mask, 0],  # X-axis
                            emb[mask, 1],  # Y-axis
                            emb[mask, 2],  # Z-axis
                        )
                    else:
                        ax.scatter(
                            emb[mask, 0],  # X-axis
                            emb[mask, 1],  # Y-axis
                            label=str(u), alpha=0.7, s=15
                        )
                ax.legend(title='Condition', fontsize='small', loc='best')
            else:
                if emb.shape[1] >= 3:
                    ax.scatter(
                        emb[:, 0],  # X-axis
                        emb[:, 1],  # Y-axis
                        emb[:, 2],  # Z-axis
                        alpha=0.7, s=15
                    )
                else:
                    ax.scatter(
                        emb[:, 0],  # X-axis
                        emb[:, 1],  # Y-axis
                        alpha=0.7, s=15
                    )

            # Set labels and title
            ax.set_xlabel(f'{title_prefix} Component 1')
            ax.set_ylabel(f'{title_prefix} Component 2')
            if emb.shape[1] >= 3:
                ax.set_zlabel(f'{title_prefix} Component 3')
            ax.set_title(f'Phase: {phase}')

            # If PCA, show variance
            evr = phase_data.get('explained_variance_ratio')
            if evr is not None:
                if emb.shape[1] >= 3:
                    ax.set_title(f'Phase: {phase} (PC1-3: {evr[0]:.2f}, {evr[1]:.2f}, {evr[2]:.2f})')
                else:
                    ax.set_title(f'Phase: {phase} (PC1-2: {evr[0]:.2f}, {evr[1]:.2f})')

        plt.tight_layout()
        filename = f'{title_prefix.lower()}_phase_group_{group_name}.png'
        plt.savefig(self.out_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def _plot_phase_group_embedding_layered(self, data_list, title_prefix):
        """
        Creates a single, layered graph for each phase group
        """
        if not data_list:
            return

        group_name = data_list[0].get('group', 'unknown')

        # Get all unique phases
        phases = []
        for data in data_list:
            phase = data.get('phase')
            if phase and phase not in phases:
                phases.append(phase)

        if not phases:
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle(f'{title_prefix} Embeddings for Phase Group: {group_name} (Layered)', fontsize=16)

        # Plot each phase with a different color
        for phase in phases:
            phase_data = None
            for data in data_list:
                if data.get('phase') == phase:
                    phase_data = data
                    break

            if not phase_data:
                continue

            emb = phase_data.get('embedding')
            if emb is None or len(emb) < 2 or emb.shape[1] < 3:
                continue

            # Plot the embedding for this phase
            ax.scatter(
                emb[:, 0],  # X-axis
                emb[:, 1],  # Y-axis
                emb[:, 2],  # Z-axis
                label=phase, alpha=0.7, s=15
            )

        # Set labels and title
        ax.set_xlabel(f'{title_prefix} Component 1')
        ax.set_ylabel(f'{title_prefix} Component 2')
        ax.set_zlabel(f'{title_prefix} Component 3')
        ax.legend(title='Phase', fontsize='small', loc='best')

        plt.tight_layout()
        filename = f'{title_prefix.lower()}_phase_group_{group_name}_layered.png'
        plt.savefig(self.out_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    # Alias for _plot_embedding
    _plot_pca = lambda self, d: self._plot_embedding(d, 'PCA')
    _plot_umap = lambda self, d: self._plot_embedding(d, 'UMAP')
    _plot_umap_with_conditions = lambda self, d: self._plot_embedding(d, 'UMAP')
    _plot_umap_pre_post = lambda self, d: self._plot_embedding(d, 'UMAP (Pre vs Post)')
    _plot_pca_pre_post = lambda self, d: self._plot_embedding(d, 'PCA (Pre vs Post)')

    # Alias for _plot_phase_group_embedding
    _plot_pca_phase_group = lambda self, d: self._plot_phase_group_embedding(d, 'PCA')
    _plot_umap_phase_group = lambda self, d: self._plot_phase_group_embedding(d, 'UMAP')

    # Alias for _plot_phase_group_embedding_layered
    _plot_pca_phase_group_layered = lambda self, d: self._plot_phase_group_embedding_layered(d, 'PCA')
    _plot_umap_phase_group_layered = lambda self, d: self._plot_phase_group_embedding_layered(d, 'UMAP')
