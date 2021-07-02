import numpy as np
from scipy import stats


def prop_diff_CI_rel(sample1, sample2, alpha=0.05):
    '''
        Return CI for proportions difference in dependent samples
            sample1 -- list/array,
            sample2 -- list/array,
            alpha -- float, significance level
    '''
    n = len(sample1)
    f = sum([1 if (i == 1 and j == 0) else 0 for i, j in zip(sample1, sample2)])
    g = sum([1 if (i == 0 and j == 1) else 0 for i, j in zip(sample1, sample2)])

    z = stats.norm.ppf(1 - alpha / 2.)
    disper = np.sqrt(float((f + g)) / n**2 - float((f - g)**2) / n**3)
    m = float(f - g) / n

    left_boundary = m - z * disper
    right_boundary = m + z * disper
    return left_boundary, right_boundary


def prop_diff_ZStat_rel(sample1, sample2):
    '''
        Return z-statistic for proportions difference in dependent samples
            sample1 -- list/array,
            sample2 -- list/array,
    '''
    n = len(sample1)
    f = sum([1 if (i == 1 and j == 0) else 0 for i, j in zip(sample1, sample2)])
    g = sum([1 if (i == 0 and j == 1) else 0 for i, j in zip(sample1, sample2)])
    return float(f - g) / np.sqrt(f + g - float((f - g)**2) / n)


def prop_diff_CI_ind(sample1, sample2, alpha=0.05):
    '''
        Return CI for proportions difference in indepentent samples
            sample1 -- list/array,
            sample2 -- list/array,
            alpha -- float, significance level
    '''
    z = stats.norm.ppf(1 - alpha / 2.)
    n1 = np.sum(sample1)
    N1 = len(sample1)
    n2 = np.sum(sample2)
    N2 = len(sample2)

    p1 = float(n1 / N1)
    p2 = float(n2 / N2)

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1) / N1 + p2 * (1 - p2) / N2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1) / N1 + p2 * (1 - p2) / N2)

    return left_boundary, right_boundary


def prop_diff_ZStat_ind(sample1, sample2):
    '''
        Return z-statistic for proportions difference in dependent samples
            sample1 -- list/array,
            sample2 -- list/array
    '''
    n1 = np.sum(sample1)
    N1 = len(sample1)
    n2 = np.sum(sample2)
    N2 = len(sample2)

    p1 = float(n1) / N1
    p2 = float(n2) / N2
    P = float(p1 * N1 + p2 * N2) / (N1 + N2)

    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / N1 + 1. / N2))


def prop_diff_ZTest(z_stat, alternative='two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    if alternative == 'two-sided':
        return 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        return stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - stats.norm.cdf(z_stat)