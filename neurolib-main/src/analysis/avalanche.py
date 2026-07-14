import numpy as np

def compute_avalanche_distribution(isi: np.ndarray, threshold_s: float) -> np.ndarray:
    """
    Compute neuronal avalanche size distribution from inter-spike intervals.

    1) Init empty list to store avalanche sizes
    2) Init avalanche size counter
    3) Iterate through each inter-spike interval
    4) If interval is less than threshold, increment avalanche size
    5) If interval exceeds threshold, record completed avalanche and reset counter
    6) Handle case where last avalanche is still in progress at end of data
    7) Convert list of avalanche sizes to numpy array
    8) Return array of avalanche sizes

                Parameters:
                    :parameter isi: np.ndarray
                    :parameter threshold_s: float

                Returns:
                    :return avalanches: np.ndarray
    """
    avalanches = []
    size = 0
    for interval in isi:
        if interval < threshold_s:
            size += 1
        else:
            if size>0:
                avalanches.append(size)
            size=0
    if size>0:
        avalanches.append(size)
    return np.array(avalanches)
