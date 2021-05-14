import os
import numpy as np
import pandas as pd

pd.set_option("display.precision", 4)

DATA_PATH = os.path.join("..", "data", "imputed", "1min")
GROUPED_BY_DATE_PATH = os.path.join("..", "data", "imputed", "grouped_by_date")

TENKAN_SEN = 9
KIJUN_SEN = 26
SENKOU_SPAN_B = 52
CHIKOU_SPAN = 26

FILES = os.listdir(DATA_PATH)
FILES_GROUPED_BY_DATE = os.listdir(GROUPED_BY_DATE_PATH)

def get_min_ichimoku(daily_data):
    close = daily_data["close"].to_numpy()
    high = daily_data["high"].to_numpy()
    low = daily_data["low"].to_numpy()

    out = {
        "min_conversion_line": [0] * TENKAN_SEN,
        "min_base_line": [0] * KIJUN_SEN,
        "leading_span_a": [0] * max(KIJUN_SEN, TENKAN_SEN),
        "leading_span_b": [0] * SENKOU_SPAN_B,
        "language_span": [0] * CHIKOU_SPAN,
    }

    for i in range(TENKAN_SEN, KIJUN_SEN):
        subset_low_9min = low[i - TENKAN_SEN:i]
        subset_high_9min = high[i - TENKAN_SEN:i]

        conversion_line = (np.max(subset_high_9min)+np.min(subset_low_9min))/2
        out["min_conversion_line"].append(conversion_line)
    for i in range(KIJUN_SEN, SENKOU_SPAN_B):
        subset_low_9min = low[i - TENKAN_SEN:i]
        subset_high_9min = high[i - TENKAN_SEN:i]
        subset_low_26min = low[i - KIJUN_SEN:i]
        subset_high_26min = high[i - KIJUN_SEN:i]

        conversion_line = (np.max(subset_high_9min) + np.min(subset_low_9min)) / 2
        out["min_conversion_line"].append(conversion_line)

        base_line = (np.max(subset_high_26min)+np.min(subset_low_26min))/2
        out["min_base_line"].append(base_line)

        out["leading_span_a"].append((conversion_line+base_line)/2)

        out["language_span"].append(close[i - CHIKOU_SPAN])

    for i in range(SENKOU_SPAN_B, len(close)):
        subset_low_9min = low[i - TENKAN_SEN:i]
        subset_high_9min = high[i - TENKAN_SEN:i]
        subset_low_26min = low[i - KIJUN_SEN:i]
        subset_high_26min = high[i - KIJUN_SEN:i]
        subset_low_52min = low[i - KIJUN_SEN:i]
        subset_high_52min = high[i - KIJUN_SEN:i]

        conversion_line = (np.max(subset_high_9min) + np.min(subset_low_9min)) / 2
        out["min_conversion_line"].append(conversion_line)
        base_line = (np.max(subset_high_26min) + np.min(subset_low_26min)) / 2
        out["min_base_line"].append(base_line)
        out["leading_span_a"].append((conversion_line + base_line) / 2)
        out["language_span"].append(close[i - CHIKOU_SPAN])
        out["leading_span_b"].append(np.max(subset_high_52min)+np.min(subset_low_52min)/2)

    return out
