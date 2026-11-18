"""
diagnostics.py
==============
Statistical diagnostics and spectral periodogram analysis for time series.
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy import signal

try:
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class PeriodogramHelper:
    """Spectral periodogram helper to detect dominant seasonal periods."""

    def __init__(self, ts: pd.Series):
        self.ts = ts.astype(float).values
        self.freq, self.spec = signal.periodogram(self.ts)

    def best_freq(self, best: int = 3) -> np.ndarray:
        res = np.argsort(-self.spec)
        res = res[(res != 0) & (res != 1)]
        res = res[:best]
        return self.freq[res]

    def best_periods(self, best: int = 3) -> np.ndarray:
        f = self.best_freq(best=best)
        f = np.where(f == 0, np.nan, f)
        return 1.0 / f


class TimeSeriesDiagnostics:
    """Diagnostic suite for temporal stationarity and autocorrelation structures."""

    @staticmethod
    def stationarity_test(series: pd.Series, significance: float = 0.05) -> pd.DataFrame:
        clean_s = series.dropna()
        if len(clean_s) < 15:
            raise ValueError("Series length must be at least 15 observations for stationarity tests.")

        if not STATSMODELS_AVAILABLE:
            return pd.DataFrame([{
                "test": "statsmodels_missing",
                "statistic": np.nan,
                "p_value": np.nan,
                "stationary": False,
                "conclusion": "Install statsmodels (pip install statsmodels) to run ADF and KPSS tests."
            }])

        adf_res = adfuller(clean_s, autolag="AIC")
        adf_stat, adf_pvalue = adf_res[0], adf_res[1]
        adf_stationary = adf_pvalue < significance

        kpss_res = kpss(clean_s, regression="c", nlags="auto")
        kpss_stat, kpss_pvalue = kpss_res[0], kpss_res[1]
        kpss_stationary = kpss_pvalue >= significance

        results = [
            {
                "test": "Augmented Dickey-Fuller (ADF)",
                "statistic": round(float(adf_stat), 4),
                "p_value": round(float(adf_pvalue), 4),
                "critical_5pct": round(float(adf_res[4]["5%"]), 4),
                "is_stationary": adf_stationary,
                "interpretation": "Stationary (Reject H0)" if adf_stationary else "Non-Stationary (Fail to reject H0)"
            },
            {
                "test": "KPSS Test",
                "statistic": round(float(kpss_stat), 4),
                "p_value": round(float(kpss_pvalue), 4),
                "critical_5pct": round(float(kpss_res[3]["5%"]), 4),
                "is_stationary": kpss_stationary,
                "interpretation": "Stationary (Fail to reject H0)" if kpss_stationary else "Non-Stationary (Reject H0)"
            }
        ]
        return pd.DataFrame(results)

    @staticmethod
    def autocorrelation_summary(series: pd.Series, nlags: int = 10) -> pd.DataFrame:
        clean_s = series.dropna()
        if not STATSMODELS_AVAILABLE:
            lags = list(range(1, nlags + 1))
            acf_vals = [round(float(clean_s.autocorr(lag=lag)), 4) for lag in lags]
            return pd.DataFrame({"lag": lags, "autocorrelation": acf_vals})

        acf_vals = acf(clean_s, nlags=nlags)
        pacf_vals = pacf(clean_s, nlags=nlags)

        return pd.DataFrame({
            "lag": list(range(len(acf_vals))),
            "acf": np.round(acf_vals, 4),
            "pacf": np.round(pacf_vals, 4)
        }).iloc[1:].reset_index(drop=True)