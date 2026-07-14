import numpy as np
import pandas as pd
from typing import List, Union, Dict, Optional


def map_channel_to_mea(channel: int) -> int:
    """
    Maps a channel number from 1-32 base to 1-128 base for MEA.

    1) Validate that channel is in the expected range (1-32 or 1-128)
    2) If channel is already in 1-128 range, return it unchanged
    3) If channel is in 1-32 range, map it to the 4th quadrant (96-128)
    4) Return the mapped channel number

                Parameters:
                    :parameter channel: int

                Returns:
                    :return mapped_channel: int
    """
    # Ensure channel is in the 1-32 range
    if channel < 1 or channel > 32:
        # If already in 1-128 range, return as is
        if 1 <= channel <= 128:
            return channel
        else:
            raise ValueError(f"Channel {channel} is out of range (1-32 or 1-128)")

    # Map to 1-128 base (assuming 4th quadrant)
    return 96 + channel


def map_channels_to_mea(channels: List[int]) -> List[int]:
    """
    Maps a list of channel numbers from 1-32 base to 1-128 base for MEA.

    1) Check if the input list is empty
    2) If empty, return an empty list
    3) Apply map_channel_to_mea function to each channel in the list
    4) Return the list of mapped channel numbers

                Parameters:
                    :parameter channels: List[int]

                Returns:
                    :return mapped_channels: List[int]
    """
    if not channels:
        return []
    return [map_channel_to_mea(ch) for ch in channels]


def assign_conditions_to_bins(df: pd.DataFrame, start_time: pd.Timestamp, bin_edges) -> np.ndarray:
    """
    Assigns the most common condition to each time bin based on spike data.

    1) Calculate the number of bins from bin edges
    2) Initialise an empty array to store conditions for each bin
    3) Check if bin edges are in milliseconds and convert to datetime if needed
    4) For each bin, find the time range between adjacent bin edges
    5) Extract spikes within each bin's time range
    6) Determine the most common condition in each bin
    7) If no condition is found, assign 'unknown'
    8) Return array of conditions for each bin

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter start_time: pd.Timestamp
                    :parameter bin_edges: list or array

                Returns:
                    :return binned_conditions: np.ndarray
    """
    num_bins = len(bin_edges) - 1
    binned_conditions = np.empty(num_bins, dtype=object)

    if np.issubdtype(type(bin_edges[0]), np.floating) or np.issubdtype(type(bin_edges[0]), np.integer):
        # Assume bin_edges are milliseconds, convert to datetime
        bin_edges = [start_time + pd.to_timedelta(ms, unit='ms') for ms in bin_edges]

    for i in range(num_bins):
        t0 = bin_edges[i]
        t1 = bin_edges[i + 1]
        mode = df[(df['Time'] >= t0) & (df['Time'] < t1)]['condition'].mode()
        binned_conditions[i] = mode[0] if not mode.empty else 'unknown'

    return binned_conditions


def group_conditions_by_pre_post(conditions: np.ndarray) -> np.ndarray:
    """
    Group conditions by pre vs post categories for STDP analysis.

    1) Init an empty array with the same shape as input conditions
    2) Iterate through each condition in the input array
    3) Check if condition is 'unknown' or 'no_stim' and keep as 'no_stim'
    4) Check if condition contains '_pre' and categorise as 'pre'
    5) Check if condition contains '_post' and categrorise as 'post'
    6) Default to 'no_stim' for any other conditions
    7) Return the array of grouped conditions

                Parameters:
                    :parameter conditions: np.ndarray

                Returns:
                    :return grouped_conditions: np.ndarray
    """
    grouped_conditions = np.empty_like(conditions)

    for i, condition in enumerate(conditions):
        if condition == 'unknown' or condition == 'no_stim':
            grouped_conditions[i] = 'no_stim'
        elif '_pre' in str(condition):
            grouped_conditions[i] = 'pre'
        elif '_post' in str(condition):
            grouped_conditions[i] = 'post'
        else:
            grouped_conditions[i] = 'no_stim'

    return grouped_conditions


def mask_stim_windows(df: pd.DataFrame, pulse_times: List[Union[str, pd.Timestamp]],
                      window_ms: float = 3.0) -> pd.DataFrame:
    """
    Removes spike entries within +/-window_ms around known pulse times to filter out stimulation artifacts.

    1) Check if input DataFrame or pulse_times list is empty
    2) Convert string timestamps to datetime objects if needed
    3) Init a boolean mask to keep track of spikes to retain
    4) Calculate the time window as a Timedelta object
    5) For each pulse time, identify spikes within the window
    6) Update the mask to exclude spikes within stimulation windows
    7) Apply the mask to filter the DataFrame
    8) Return the filtered DataFrame with reset index

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter pulse_times: list
                    :parameter window_ms: float, default=3.0 ms

                Returns:
                    :return filtered_df: pd.DataFrame
    """
    if df.empty or not pulse_times:
        return df

    pulse_times = [pd.to_datetime(ts, utc=True).tz_convert(None)
                   if isinstance(ts, str) else ts for ts in pulse_times]

    keep_mask = pd.Series(True, index=df.index)
    window_delta = pd.Timedelta(milliseconds=window_ms)

    for pulse_time in pulse_times:
        # Mask out spikes within +/-window_ms of the pulse
        mask = (df['Time'] >= (pulse_time - window_delta)) & (df['Time'] <= (pulse_time + window_delta))
        keep_mask = keep_mask & ~mask

    # Apply mask
    return df[keep_mask].reset_index(drop=True)


def compute_stim_artifact_template(df: pd.DataFrame, pulse_times: List[Union[str, pd.Timestamp]],
                                   window_ms: float = 5.0, channels: Optional[List[int]] = None) -> Dict[
    int, pd.DataFrame]:
    """
    Computes the average stimulation artifact template for each channel to enable artifact removal.

    1) Check if input DataFrame or pulse_times list is empty
    2) Convert string timestamps to datetime objects if needed
    3) Calculate the time window as a Timedelta object
    4) Determine which channels to process (use provided list or all channels in data)
    5) For each channel:
       a. Filter data to include only the specified channel
       b. Skip empty channels
       c. For each pulse time, collect spikes within the window
       d. Normalize time relative to pulse time
       e. Accumulate artifacts for the channel
    6) Combine artifacts for each channel into a template
    7) Return dictionary of templates mapped by channel number

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter pulse_times: list
                    :parameter window_ms: float, default=5.0 ms
                    :parameter channels: list, optional

                Returns:
                    :return templates: dict, mapping channel numbers to their artifact templates
    """
    if df.empty or not pulse_times:
        return {}

    # Convert string timestamps to datetime if needed
    pulse_times = [pd.to_datetime(ts, utc=True).tz_convert(None)
                   if isinstance(ts, str) else ts for ts in pulse_times]

    window_delta = pd.Timedelta(milliseconds=window_ms)
    channels_to_use = channels if channels is not None else df['channel'].unique()
    templates = {}

    for channel in channels_to_use:
        channel_df = df[df['channel'] == channel]
        if channel_df.empty:
            continue

        # Collect artifacts around each pulse
        artifacts = []
        for pulse_time in pulse_times:
            window_start = pulse_time - window_delta
            window_end = pulse_time + window_delta

            # Get spikes in this window
            window_spikes = channel_df[(channel_df['Time'] >= window_start) &
                                       (channel_df['Time'] <= window_end)]

            if not window_spikes.empty:
                # Normalise time relative to pulse
                window_spikes = window_spikes.copy()
                window_spikes['relative_time_ms'] = (window_spikes['Time'] - pulse_time) / pd.Timedelta(milliseconds=1)
                artifacts.append(window_spikes)

        if artifacts:
            # Combine artifacts for this channel
            combined = pd.concat(artifacts, ignore_index=True)
            templates[channel] = combined

    return templates


def apply_template_subtraction(df: pd.DataFrame, pulse_times: List[Union[str, pd.Timestamp]],
                               templates: Dict[int, pd.DataFrame], window_ms: float = 5.0) -> pd.DataFrame:
    """
    Applies template subtraction to reduce stimulation artifacts in neural recordings.

    1) Check if input DataFrame, pulse_times list, or templates dictionary is empty
    2) Create a copy of the input DataFrame to avoid modifying the original
    3) Calculate the time window as a Timedelta object
    4) Convert string timestamps to datetime objects if needed
    5) For each channel with a template:
       a. Skip channels with empty templates
       b. Create a mask for the current channel
       c. For each pulse time:
          i. Define the time window around the pulse
          ii. Find spikes within the window for this channel
          iii. Skip if no spikes found
          iv. Calculate relative time for each spike in the window
          v. For each spike, find the closest template point
          vi. Subtract the template amplitude from the spike amplitude
    6) Remove temporary columns used for calculations
    7) Return the DataFrame with template-subtracted amplitudes

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter pulse_times: list
                    :parameter templates: dict
                    :parameter window_ms: float, default=5.0 ms

                Returns:
                    :return df: pd.DataFrame with template-subtracted amplitudes
    """
    if df.empty or not pulse_times or not templates:
        return df

    result_df = df.copy()
    window_delta = pd.Timedelta(milliseconds=window_ms)

    pulse_times = [pd.to_datetime(ts, utc=True).tz_convert(None)
                   if isinstance(ts, str) else ts for ts in pulse_times]

    for channel, template in templates.items():

        if template.empty:
            continue

        channel_mask = result_df['channel'] == channel

        for pulse_time in pulse_times:
            window_start = pulse_time - window_delta
            window_end = pulse_time + window_delta

            # Find spikes in this window for this channel
            window_mask = (result_df['Time'] >= window_start) & \
                          (result_df['Time'] <= window_end) & channel_mask

            if not any(window_mask):
                continue

            # Calculate relative time for each spike in the window
            result_df.loc[window_mask, 'relative_time_ms'] = (result_df.loc[window_mask, 'Time'] - pulse_time) / pd.Timedelta(milliseconds=1)

            # For each spike -> find the closest template point and subtract
            for idx in result_df[window_mask].index:
                rel_time = result_df.loc[idx, 'relative_time_ms']
                closest_template_idx = (template['relative_time_ms'] - rel_time).abs().idxmin()
                template_amplitude = template.loc[closest_template_idx, 'Amplitude']
                result_df.loc[idx, 'Amplitude'] -= template_amplitude

    # Remove temporary column
    if 'relative_time_ms' in result_df.columns:
        result_df = result_df.drop(columns=['relative_time_ms'])

    return result_df
