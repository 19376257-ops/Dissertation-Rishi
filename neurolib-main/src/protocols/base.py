from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ProtocolBase(ABC):
    """
    Abstract base class defining the standard pipeline for all neuronal recording protocols.

    1) Init with configuration and paths
    2) Load raw CSVs containing amplitude data
    3) Preprocess the combined DataFrame
    4) Analyse data using protocol-specific methods
    5) Automatically generate a raster plot result
    6) Visualise results via appropriate plotting methods
    7) Return analysis results

                Parameters:
                    :parameter cfg: dict - Configuration dictionary with protocol settings
                    :parameter paths: dict - Dictionary containing paths to raw and results directories
    """

    def __init__(self, cfg: dict, paths: dict):
        # Save config and resolve paths
        self.cfg = cfg
        self.raw_dir = Path(paths['raw'])
        self.results_dir = Path(paths['results'])
        self._ensure_results_dir()

    def _ensure_results_dir(self) -> None:
        """
        Create results directory if it doesn't exist to store analysis outputs.

        1) Attempt to create the results directory with parents if needed
        2) Log success message if directory is created or already exists
        3) Log error message if directory creation fails

                    Returns:
                        :return: None
        """
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Results directory ready: {self.results_dir}")
        except Exception as e:
            logger.error(f"Could not create results dir {self.results_dir}: {e}")

    def run(self, sample_mode=False) -> list:
        """
        Execute the complete protocol pipeline from data loading to visualisation.

        1) Load raw data from CSV files
        2) Preprocess the loaded data
        3) Analyse the preprocessed data with protocol-specific methods
        4) Generate raster plot data automatically
        5) Visualise all results
        6) Return the list of analysis results

                    Parameters:
                        :parameter sample_mode: bool - If True, use reduced data for faster processing

                    Returns:
                        :return results: list - List of dictionaries containing analysis results
        """
        # 1) LOAD
        data = self.load()

        # 2) PREPROCESS
        preproc = self.preprocess(data)

        # 3) ANALYSE
        results = self.analyse(preproc, sample_mode=sample_mode)

        # 4) RASTER: auto-generate raster plot data for all protocols
        raster = self._generate_raster_data(preproc)
        if raster:
            results.append(raster)

        # 5) VISUALISE
        self.visualise(results)

        return results

    def load(self) -> pd.DataFrame:
        """
        Load and concatenate amplitude-only CSV files from the raw data directory.

        1) Get file pattern from configuration (must be:'*.csv')
        2) Find all matching CSV files in the raw directory
        3) Validate that at least one matching file exists
        4) For each CSV file:
           a. Check if it contains the required columns (Time, channel, Amplitude)
           b. Skip files missing required columns
           c. Read the file with appropriate data types and date parsing
           d. Add the data to the collection
        5) Validate that at least one valid CSV file was processed
        6) Concatenate all valid data frames into a single DataFrame
        7) Return the combined DataFrame

                    Returns:
                        :return data: pd.DataFrame - Combined DataFrame with Time, channel, and Amplitude columns
        """
        pattern = self.cfg.get('file_pattern', '*.csv')
        paths = list(self.raw_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No CSV files found in {self.raw_dir} matching pattern '{pattern}'")

        frames = []
        required_cols = {'Time', 'channel', 'Amplitude'}

        # error handling for multiple CSV types
        for p in paths:
            try:
                cols = set(pd.read_csv(p, nrows=1).columns)
                if not required_cols.issubset(cols):
                    logger.warning(f"Skipping {p.name}: missing required columns")
                    continue

                df = pd.read_csv(
                    p,
                    parse_dates=['Time'],
                    usecols=['Time', 'channel', 'Amplitude'],
                    dtype={'channel': int}
                )
                frames.append(df)

            except Exception as e:
                logger.warning(f"Skipping {p.name}: {e} ")

        if not frames:
            raise ValueError(f"No valid CSV files with columns {required_cols} found in {self.raw_dir}")

        return pd.concat(frames, ignore_index=True)

    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def analyse(self, preproc: pd.DataFrame) -> list:
        pass

    def _generate_raster_data(self, df: pd.DataFrame, phase: str = None) -> dict:
        """
        Build raster plot data from spike times and channel information.

        1) Check if input DataFrame is empty
        2) Filter data by phase/condition if specified
        3) Apply amplitude threshold from config if available
        4) Check if filtered data contains any spikes
        5) Create result dictionary with spike times and channel information
        6) Add phase information to the result
        7) Return the raster plot data dictionary

                    Parameters:
                        :parameter df: pd.DataFrame - DataFrame containing spike data
                        :parameter phase: str - Optional condition/phase to filter by

                    Returns:
                        :return result: dict - Dictionary containing raster plot data or None if no valid data
        """
        if df.empty:
            return None

        if phase is not None:
            df = df[df["condition"] == phase]

        # Error handling
        thr = self.cfg.get('raster_threshold', None)
        if thr is not None:
            spikes = df[df['Amplitude'] < thr]
        else:
            spikes = df

        if spikes.empty:
            return None

        result = {
            'type': 'raster',
            'data': {
                'times': spikes['Time'].values,
                'channels': spikes['channel'].values
            }
        }

        if phase is not None:
            result['data']['phase'] = phase
        else:
            # Use 'all' as the phase name for the overall raster plot
            result['data']['phase'] = 'all'

        return result

    @abstractmethod
    def visualise(self, results: list) -> None:
        pass
