
import numpy as np


def calculate_conformal_interval(prediction, residuals, coverage=0.90):
    """
    Calculate a conformal prediction interval.

    prediction: predicted animal weight
    residuals: calibration absolute errors
    coverage: target coverage, e.g. 0.90 for 90%
    """

    residuals = np.asarray(residuals)
    n = len(residuals)

    if n == 0:
        raise ValueError("Residuals cannot be empty.")

    if not 0 < coverage < 1:
        raise ValueError("Coverage must be between 0 and 1.")

    level = np.ceil((n + 1) * coverage) / n

    if level > 1:
        q = np.inf
    else:
        q = np.quantile(residuals, level, method="higher")

    lower = max(0, prediction - q)
    upper = prediction + q

    return lower, upper