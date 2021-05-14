import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


pd.set_option("display.precision", 4)

DATA_PATH = os.path.join("..", "data", "imputed", "1min")
GROUPED_BY_DATE_PATH = os.path.join("..", "data", "imputed", "grouped_by_date")
OFFSET = 20  # minutes from start
FILES = os.listdir(DATA_PATH)
FILES_GROUPED_BY_DATE = os.listdir(GROUPED_BY_DATE_PATH)
EMA20_MULTIPLIER = (2 / (OFFSET + 1))


def get_technical_indicator_features(daily_data):
    close = daily_data["close"].to_numpy()
    high = daily_data["high"].to_numpy()  # daily closing prices as np array
    low = daily_data["low"].to_numpy()  # daily closing prices as np array
    typical_price = np.sum(close, high, low) / 3

    out = {
        "keep_row": [False] * OFFSET,  # these rows will be deleted later, first N minutes of a day
        "typical_price": typical_price,
        "20min_sma": [0] * OFFSET,
        "20min_tsma": [0] * OFFSET,
        "20min_sd_tp": [0] * OFFSET,
        "20min_ema": [0] * (OFFSET-1) + np.mean(close[0:OFFSET-1]),

    }

    for i in range(OFFSET, len(close)):

        subset_close = close[i - OFFSET:i]
        subset_typical_price = typical_price[i - OFFSET:i]

        out["keep_row"].append(True)

        out["20min_sma"].append(np.mean(subset_close[i - OFFSET:i]))
        out["20min_tsma"].append(np.mean(subset_typical_price[i - OFFSET:i]))

        out["20min_sd_tp"].append(np.std(subset_typical_price))

        prev_ema = out["20min_ema"][i-1]
        out["20min_ema"].append(((close[i] - prev_ema) * EMA20_MULTIPLIER) + prev_ema)

    out["20min_diff_sma_ema"] = out["20min_sma"] - out["20min_ema"]
    out["bbu"] = out["20min_tsma"] + 2*out["20min_sd_tp"]
    out["bbl"] = out["20min_tsma"] - 2*out["20min_sd_tp"]

    return out


def initiate_20min_indicators(daily_data):
    close = daily_data["close"].to_numpy()  # daily closing prices as np array

    # initialize output with first N rows
    out = {
        "keep_row": [False] * OFFSET,  # these rows will be deleted later, first N minutes of a day
        "20min_sma": [0] * OFFSET,
        "20min_tsma": [0] * OFFSET,
    }

    for i in range(OFFSET, len(close)):
        out["keep_row"].append(True)
        out["20min_sma"].append(np.mean(subset_close[i - OFFSET:i]))
        out["20min_tsma"].append(np.mean(subset_typical_price[i - OFFSET:i]))
    return out


def get_typical_price(daily_data):
    close = daily_data["close"].to_numpy()  # daily closing prices as np array
    high = daily_data["high"].to_numpy()  # daily closing prices as np array
    low = daily_data["low"].to_numpy()  # daily closing prices as np array

    sum_chl = np.sum(close, high, low)

    out = {
        # Doesn't matter the initiation.
        "typical_price": [],
    }

    for i in range(0, len(close)):
        out["typical_price"].append(sum_chl[i]/3)

    return out


def get_20min_sma_tsma(daily_data, typical_price_data):
    close = daily_data["close"].to_numpy()  # daily closing prices as np array
    typical_price = typical_price_data["typical_price"].to_numpy()

    out = {
        "20min_sma": [0] * OFFSET,
        "20min_tsma": [0] * OFFSET,
    }
    for i in range(OFFSET, len(close)):
        subset_close = close[i - OFFSET:i]
        subset_typical_price = typical_price[i - OFFSET:i]

        # For the 20th min we took mean of the 0th...19th minutes.
        out["20min_sma"].append(np.mean(subset_close[i - OFFSET:i]))
        out["20min_tsma"].append(np.mean(subset_typical_price[i - OFFSET:i]))

    return out


def get_20min_sd_typical_price(typical_price_data):
    typical_price = typical_price_data["typical_price"].to_numpy()

    out = {
        "20min_sd_tp": [0] * OFFSET,
    }

    for i in range(OFFSET, len(typical_price)):
        subset_typical_price = typical_price[i - OFFSET:i]
        out["20min_sd_tp"].append(np.std(subset_typical_price))

    return out


def get_20min_ema(daily_data):
    close = daily_data["close"].to_numpy()
    out = {
        "20min_ema": [0] * OFFSET,
    }

    # We are going to use SMA for the first previous EMA. We wanted to reach first SMA.
    first_20_close = close[0:OFFSET]
    previous_ema = np.mean(first_20_close[0:OFFSET])
    out["20min_ema"].append(previous_ema)
    multiplier = (2 / (OFFSET + 1))

    # Starts from OFFSET+1 because we used SMA to initiate first EMA.
    for i in range(OFFSET+1, len(close)):
        # Index in close should be same for all of the file. It will be discussed afterward.
        current_ema = ((close[i] - previous_ema) * multiplier) + previous_ema
        out["20min_ema"].append(current_ema)
        previous_ema = current_ema

    return out


def get_20min_diff_sma_ema(sma_data, ema_data):
    sma = sma_data["20min_sma"].to_numpy()
    ema = ema_data["20min_ema"].to_numpy()

    out = {
        "20min_diff_sma_ema": sma - ema,
    }

    return out


def get_20min_bb(tsma_data, sd_tp_data):
    TSMA_20min = tsma_data["20min_tsma"].to_numpy()
    SD_TP_20min = sd_tp_data["20min_sd_tp"].to_numpy()

    # 2 usual standart for the BB width
    out = {"bbu": TSMA_20min + 2 * SD_TP_20min, "bbl": TSMA_20min - 2 * SD_TP_20min}

    return out

