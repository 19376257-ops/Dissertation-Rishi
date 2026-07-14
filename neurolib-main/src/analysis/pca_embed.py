import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.analysis.utils import assign_conditions_to_bins

# Sample Mode reduced variable set
SAMPLE_BINS = 500

def compute_pca_embedding(df: pd.DataFrame, n_components: int = 3, 
                         bin_size_ms: int = 10, random_state: int = 42, sample_mode: bool = False) -> dict:
    """
    Compute PCA embedding for neural activity data with binning and scaling.

    1) Check if DataFrame is empty
    2) Convert time data to datetime format
    3) Calculate time range and create bins
    4) Extract unique channels and create neural activity DataFrame
    5) Bin neural activity by channel
    6) Assign conditions to bins
    7) Apply sample mode if enabled
    8) Scale neural activity data
    9) Apply PCA for dimensionality reduction
    10) Return embedding results or error information

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter n_components: int
                    :parameter bin_size_ms: int
                    :parameter random_state: int
                    :parameter sample_mode: bool

                Returns:
                    :return embedding: dict
    """
    # Check if DataFrame is empty
    if df.empty:
        return {
            'error': 'Empty data',
            'n_components': n_components,
            'explained_variance_ratio': None,
            'bin_size_ms': bin_size_ms,
            'binned_data': pd.DataFrame(),
            'conditions': np.array([])
        }

    data = df.copy()

    data['Time'] = pd.to_datetime(data['Time'], utc=True).dt.tz_localize(None)

    start_time = data['Time'].min()
    end_time = data['Time'].max()
    total_ms = (end_time - start_time).total_seconds() * 1000
    num_bins = int(np.ceil(total_ms / bin_size_ms))

    bin_edges = np.linspace(0, total_ms, num_bins + 1)

    channels = data['channel'].unique()
    neural_activity = pd.DataFrame(index=range(num_bins), columns=channels)

    for channel in channels:
        times_ms = (data.loc[data['channel']==channel,'Time'] - start_time).dt.total_seconds() * 1000
        counts, _ = np.histogram(times_ms, bins=bin_edges)
        neural_activity[channel] = counts

    bin_edges_dt = [start_time + pd.Timedelta(ms, unit='ms') for ms in bin_edges]
    binned_conditions = assign_conditions_to_bins(data, start_time, bin_edges_dt)

    # SAMPLE MODE
    if sample_mode:
        neural_activity = neural_activity.iloc[:SAMPLE_BINS]
        binned_conditions = binned_conditions[:SAMPLE_BINS]

    neural_activity_array = neural_activity.fillna(0).values

    # Normalise - improved
    neural_activity_scaled = StandardScaler().fit_transform(neural_activity_array)

    try:

        n_components = min(n_components, neural_activity_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=random_state)
        embedding = pca.fit_transform(neural_activity_scaled)

        return {
            'embedding': embedding,
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'bin_size_ms': bin_size_ms,
            'binned_data': neural_activity,
            'conditions': binned_conditions
        }
    except Exception as e:
        return {
            'error': str(e),
            'n_components': n_components,
            'explained_variance_ratio': None,
            'bin_size_ms': bin_size_ms,
            'binned_data': neural_activity,
            'conditions': binned_conditions
        }
