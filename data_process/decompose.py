import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

from scipy.ndimage import gaussian_filter1d
import torch
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Data structure
# ----------------------------------------------------------------------
@dataclass
class DecompositionResult:
    original: np.ndarray
    trend: np.ndarray
    season: np.ndarray
    residual: np.ndarray
    detrended: np.ndarray
    deseasonalized: np.ndarray

    dominant_period: int
    seasonal_pattern: np.ndarray

    metadata: Dict


# ----------------------------------------------------------------------
# Main decomposer
# ----------------------------------------------------------------------
class TimeSeriesDecomposer:
    """
    x = trend + season + residual
    """

    def __init__(
        self,
        trend_sigma: float = 6.0,
        fft_top_k: int = 3,
        remove_dc_for_fft: bool = True,
    ):
        self.trend_sigma = trend_sigma
        self.fft_top_k = fft_top_k
        self.remove_dc_for_fft = remove_dc_for_fft

    # ------------------------------------------------------------------
    def decompose(self, x: np.ndarray) -> DecompositionResult:
        x = self._to_1d_float_array(x)

        # ---- trend ----
        trend = self._extract_trend(x)
        detrended = x - trend

        # ---- season ----
        season, dominant_period, seasonal_pattern = self._extract_season(detrended)
        deseasonalized = detrended - season

        # ---- residual ----
        residual = deseasonalized

        return DecompositionResult(
            original=x,
            trend=trend,
            season=season,
            residual=residual,
            detrended=detrended,
            deseasonalized=deseasonalized,
            dominant_period=dominant_period,
            seasonal_pattern=seasonal_pattern,
            metadata={
                "trend_sigma": self.trend_sigma,
                "fft_top_k": self.fft_top_k,
            },
        )

    # ------------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------------
    def _extract_trend(self, x: np.ndarray) -> np.ndarray:
        return gaussian_filter1d(x, sigma=self.trend_sigma, mode="nearest")

    # ------------------------------------------------------------------
    # Season (FFT top-k + extract one cycle)
    # ------------------------------------------------------------------
    def _extract_season(self, x: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
        n = len(x)

        x_for_fft = x.copy()
        if self.remove_dc_for_fft:
            x_for_fft = x_for_fft - x_for_fft.mean()

        fft_vals = np.fft.rfft(x_for_fft)
        freqs = np.fft.rfftfreq(n, d=1.0)

        magnitudes = np.abs(fft_vals)
        magnitudes[0] = 0.0

        # ---- top-k frequencies ----
        top_k = min(self.fft_top_k, len(magnitudes) - 1)
        top_idx = np.argsort(magnitudes)[-top_k:]

        # ---- reconstruct season ----
        masked_fft = np.zeros_like(fft_vals, dtype=np.complex128)
        masked_fft[top_idx] = fft_vals[top_idx]
        season = np.fft.irfft(masked_fft, n=n).real

        # ---- dominant frequency ----
        main_idx = max(top_idx, key=lambda i: magnitudes[i])

        if main_idx == 0:
            dominant_period = n
        else:
            dominant_period = int(round(n / main_idx))

        dominant_period = max(2, min(dominant_period, n))

        # ---- extract one cycle ----
        seasonal_pattern = season[:dominant_period].copy()

        return season, dominant_period, seasonal_pattern

    # ------------------------------------------------------------------
    @staticmethod
    def _to_1d_float_array(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {x.shape}")
        return x


# ----------------------------------------------------------------------
# Trend description (linear / exp / log)
# ----------------------------------------------------------------------
def describe_trend(trend: np.ndarray) -> Dict:
    n = len(trend)
    x = np.arange(n, dtype=np.float64)

    # ---- linear ----
    coef_lin = np.polyfit(x, trend, 1)
    pred_lin = np.polyval(coef_lin, x)
    err_lin = np.mean((trend - pred_lin) ** 2)

    # ---- exponential ----
    trend_pos = trend - trend.min() + 1e-6
    log_y = np.log(trend_pos)
    coef_exp = np.polyfit(x, log_y, 1)
    pred_exp = np.exp(np.polyval(coef_exp, x))
    err_exp = np.mean((trend_pos - pred_exp) ** 2)

    # ---- log ----
    x_log = np.log(x + 1)
    coef_log = np.polyfit(x_log, trend, 1)
    pred_log = np.polyval(coef_log, x_log)
    err_log = np.mean((trend - pred_log) ** 2)

    errors = {
        "linear": err_lin,
        "exponential": err_exp,
        "log": err_log
    }

    trend_type = min(errors, key=errors.get)

    return {
        "trend_type": trend_type,
        "errors": errors
    }

def generate_text_description(result: DecompositionResult) -> str:
    """
    Generate structured natural language description for trend + seasonal.
    """

    trend_info = describe_trend(result.trend)
    season_info = describe_season(result.dominant_period, result.seasonal_pattern)

    # -------- Trend description --------
    trend_type = trend_info["trend_type"]

    # direction
    total_change = result.trend[-1] - result.trend[0]
    if total_change > 1e-6:
        direction = "increasing"
    elif total_change < -1e-6:
        direction = "decreasing"
    else:
        direction = "constant"

    # curvature strength
    curvature = np.mean(np.abs(np.diff(result.trend, n=2))) if len(result.trend) > 2 else 0.0
    if curvature < 0.01:
        shape = "linear"
    else:
        shape = "nonlinear"

    trend_sentence = f"The trend is {trend_type} and {direction} with a {shape} shape."

    # -------- Seasonal description --------
    period = season_info["period"]
    amplitude = season_info["amplitude"]

    if amplitude < 0.1:
        amp_level = "weak"
    elif amplitude < 0.5:
        amp_level = "moderate"
    else:
        amp_level = "strong"

    if period > 0:
        season_sentence = (
            f"The time series exhibits a periodic pattern with a period of {period} "
            f"and {amp_level} amplitude."
        )
    else:
        season_sentence = "No clear periodic pattern is detected."

    # -------- Combine --------
    description = trend_sentence + " " + season_sentence

    return description
# ----------------------------------------------------------------------
# Season description
# ----------------------------------------------------------------------
def describe_season(period: int, pattern: np.ndarray) -> Dict:
    return {
        "period": period,
        "pattern_length": len(pattern),
        "amplitude": float(np.std(pattern))
    }


# ----------------------------------------------------------------------
# Combined description
# ----------------------------------------------------------------------
def build_structured_description(result: DecompositionResult) -> Dict:

    trend_description = describe_trend(result.trend)
    season_description = describe_season(result.dominant_period, result.seasonal_pattern)

    return {
        "trend": trend_description,
        "season": season_description,
    }


def process_ts(x_ts, save_name):
    if isinstance(x_ts, torch.Tensor):
        x_ts = x_ts.detach().cpu().numpy()
    decomposer = TimeSeriesDecomposer(
        trend_sigma=20.0,
        fft_top_k=3,
    )
    result = decomposer.decompose(x_ts)
    structure_desc = build_structured_description(result)
    text_desc = generate_text_description(result)

    plt.figure(figsize=(8, 6))
    plt.plot(result.trend)
    plt.yticks([])  # 🔥 removes y-ticks
    plt.savefig(save_name)
    # plt.figure(figsize=(12, 6))
    #
    # plt.subplot(4, 1, 1)
    # plt.plot(result.original)
    # plt.title("Original")
    #
    # plt.subplot(4, 1, 2)
    # plt.plot(result.trend)
    # plt.title("Trend")
    #
    # plt.subplot(4, 1, 3)
    # plt.plot(result.season)
    # plt.title("Season")
    #
    # plt.subplot(4, 1, 4)
    # plt.plot(result.residual)
    # plt.title("Residual")
    #
    # plt.tight_layout()
    # plt.show()

    return text_desc, result.trend, result.season, result.residual



# ----------------------------------------------------------------------
# Example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # data = np.load("/Users/zhc/Documents/LitsDatasets/128_len_ts/ETTh1/train_ts.npy", allow_pickle=True)
    data = np.load("/playpen-shared/haochenz/LitsDatasets/128_len_ts/ETTh1/train_ts.npy", allow_pickle=True)
    save_path = "/playpen-shared/haochenz/LitsDatasets/128_len_ts_trend_image/ETTh1/train"
    import os
    os.makedirs(save_path, exist_ok=True)
    for i, datum in enumerate(data):
        for j in datum.shape[-1]:

            text_desc, trend, seasonal_pattern, residual = process_ts(
                datum[:, j], save_name=f"ts{i}_ch{j}.png"
            )



