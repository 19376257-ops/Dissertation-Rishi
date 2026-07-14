import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, pearsonr, shapiro, ttest_ind, mannwhitneyu

def compute_statistical_tests(df: pd.DataFrame, pre_channel: int, post_channel: int, condition: str = None, sample_mode: bool = False) -> dict:
    """
    Compute various statistical tests between pre and post channel data to analyse neural activity.

    1) Init result dictionary with channel information
    2) Apply condition filtering if specified
    3) Apply sample mode if enabled (limit to 1000 data points)
    4) Extract pre and post channel data
    5) Filter by condition if specified
    6) Sort data by time
    7) Calculate inter-spike intervals (ISI)
    8) Perform Kolmogorov-Smirnov test to compare distributions
    9) Align ISI data for correlation analysis
    10) Calculate Pearson correlation between aligned ISIs
    11) Perform Shapiro-Wilk test to check for normality
    12) Perform t-test to compare means
    13) Perform Mann-Whitney U test for non-parametric comparison
    14) Return comprehensive statistical results

                Parameters:
                    :parameter df: pd.DataFrame
                    :parameter pre_channel: int
                    :parameter post_channel: int
                    :parameter condition: str
                    :parameter sample_mode: bool

                Returns:
                    :return result: dict
    """

    result = {
        'pre_channel': pre_channel,
        'post_channel': post_channel
    }

    if condition is not None:
        result['condition'] = condition

    # Sample mode
    data = df.copy()
    if sample_mode and len(data) > 1000:
        # Take a random sample of 1000 data points
        data = data.sample(n=1000, random_state=42)

    pre_data = data[data['channel'] == pre_channel].copy()
    post_data = data[data['channel'] == post_channel].copy()

    if condition is not None and 'condition' in data.columns:
        pre_data = pre_data[pre_data['condition'] == condition]
        post_data = post_data[post_data['condition'] == condition]

    pre_data = pre_data.sort_values('Time')
    post_data = post_data.sort_values('Time')

    if pd.api.types.is_datetime64_any_dtype(pre_data['Time']):
        pre_isi = pre_data['Time'].diff().dt.total_seconds().dropna().values
        post_isi = post_data['Time'].diff().dt.total_seconds().dropna().values
    else:
        pre_isi = pre_data['Time'].diff().dropna().values
        post_isi = post_data['Time'].diff().dropna().values

    if len(pre_isi) > 1 and len(post_isi) > 1:
        # Kolmogorov-Smirnov (K-S) Test
        try:
            min_ks_size = 2
            if len(pre_isi) < min_ks_size or len(post_isi) < min_ks_size:
                result['ks_test'] = {
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'error': f"Not enough samples for Kolmogorov-Smirnov test (pre: {len(pre_isi)}, post: {len(post_isi)}). Minimum required: {min_ks_size}."
                }
            else:
                ks_stat, ks_pvalue = ks_2samp(pre_isi, post_isi)
                result['ks_test'] = {
                    'statistic': ks_stat,
                    'p_value': ks_pvalue
                }
        except Exception as e:
            result['ks_test'] = {'error': str(e)}

        min_len = min(len(pre_isi), len(post_isi))
        pre_isi_aligned = pre_isi[:min_len]
        post_isi_aligned = post_isi[:min_len]

        # Pearson Correlation
        try:
            min_pearson_size = 2
            if min_len < min_pearson_size:
                result['pearson'] = {
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'error': f"Not enough samples for Pearson correlation (aligned length: {min_len}). Minimum required: {min_pearson_size}."
                }
            elif len(np.unique(pre_isi_aligned)) == 1 or len(np.unique(post_isi_aligned)) == 1:
                result['pearson'] = {
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'error': "Cannot compute correlation: one or both input arrays are constant."
                }
            else:
                corr, corr_pvalue = pearsonr(pre_isi_aligned, post_isi_aligned)
                result['pearson'] = {
                    'correlation': corr,
                    'p_value': corr_pvalue
                }
        except Exception as e:
            result['pearson'] = {'error': str(e)}

        # Shapiro-Wilk Test (Normality Test)
        try:
            max_shapiro_size = 5000

            min_shapiro_size = 3

            if len(pre_isi) > max_shapiro_size:
                step = len(pre_isi) // max_shapiro_size
                pre_isi_sample = pre_isi[::step][:max_shapiro_size]
                shapiro_pre = shapiro(pre_isi_sample)
                accuracy_warning_pre = True
                small_sample_pre = False
            elif len(pre_isi) < min_shapiro_size:
                shapiro_pre = (np.nan, np.nan)
                accuracy_warning_pre = False
                small_sample_pre = True
            else:
                if len(pre_isi) >= 4:
                    shapiro_pre = shapiro(pre_isi)
                else:
                    shapiro_pre = (np.nan, np.nan)
                    small_sample_pre = True
                accuracy_warning_pre = False
                small_sample_pre = False

            # For post-channel data
            if len(post_isi) > max_shapiro_size:
                step = len(post_isi) // max_shapiro_size
                post_isi_sample = post_isi[::step][:max_shapiro_size]
                shapiro_post = shapiro(post_isi_sample)
                accuracy_warning_post = True
                small_sample_post = False
            elif len(post_isi) < min_shapiro_size:
                shapiro_post = (np.nan, np.nan)
                accuracy_warning_post = False
                small_sample_post = True
            else:
                if len(post_isi) >= 4:  # Shapiro-Wilk requires at least 3 samples, but use 4 to be safe
                    shapiro_post = shapiro(post_isi)
                else:
                    shapiro_post = (np.nan, np.nan)
                    small_sample_post = True
                accuracy_warning_post = False
                small_sample_post = False

            result['shapiro'] = {
                'pre_channel': {
                    'statistic': shapiro_pre[0],
                    'p_value': shapiro_pre[1],
                    'downsampled': accuracy_warning_pre,
                    'small_sample': small_sample_pre,
                    'original_size': len(pre_isi)
                },
                'post_channel': {
                    'statistic': shapiro_post[0],
                    'p_value': shapiro_post[1],
                    'downsampled': accuracy_warning_post,
                    'small_sample': small_sample_post,
                    'original_size': len(post_isi)
                }
            }
        except Exception as e:
            result['shapiro'] = {'error': str(e)}

        # t-Test
        try:
            # t-test requires at least 2 samples in each group
            min_ttest_size = 2
            if len(pre_isi) < min_ttest_size or len(post_isi) < min_ttest_size:
                result['t_test'] = {
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'error': f"Not enough samples for t-test (pre: {len(pre_isi)}, post: {len(post_isi)}). Minimum required: {min_ttest_size}."
                }
            else:
                t_stat, t_pvalue = ttest_ind(pre_isi, post_isi, equal_var=False)
                result['t_test'] = {
                    'statistic': t_stat,
                    'p_value': t_pvalue
                }
        except Exception as e:
            result['t_test'] = {'error': str(e)}

        # Mann-Whitney U Test
        try:
            # Mann-Whitney U test requires at least 1 sample in each group,
            # but practically we need more for meaningful results
            min_mw_size = 2
            if len(pre_isi) < min_mw_size or len(post_isi) < min_mw_size:
                result['mann_whitney'] = {
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'error': f"Not enough samples for Mann-Whitney U test (pre: {len(pre_isi)}, post: {len(post_isi)}). Minimum required: {min_mw_size}."
                }
            else:
                mw_stat, mw_pvalue = mannwhitneyu(pre_isi, post_isi)
                result['mann_whitney'] = {
                    'statistic': mw_stat,
                    'p_value': mw_pvalue
                }
        except Exception as e:
            result['mann_whitney'] = {'error': str(e)}
    else:
        result['error'] = f"Not enough data points (pre: {len(pre_isi)}, post: {len(post_isi)}) for statistical tests"

    return result
