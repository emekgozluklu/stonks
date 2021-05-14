import os
import numpy as np
import pandas as pd

pd.set_option("display.precision", 4)

DATA_PATH = os.path.join("..", "data", "imputed", "1min")
GROUPED_BY_DATE_PATH = os.path.join("..", "data", "imputed", "grouped_by_date")
OFFSET1 = 12  # minutes from start
OFFSET2 = 26
OFFSET3 = 9
FILES = os.listdir(DATA_PATH)
FILES_GROUPED_BY_DATE = os.listdir(GROUPED_BY_DATE_PATH)
EMA12_MULTIPLIER = (2 / (OFFSET1 + 1))
EMA26_MULTIPLIER = (2 / (OFFSET2 + 1))
EMA9_MULTIPLIER = (2 / (OFFSET3 + 1))


def get_12min_ema(daily_data):
    close = daily_data["close"].to_numpy()
    out = {
        "12min_ema": [0] * OFFSET1,
    }

    # We are going to use SMA for the first previous EMA. We wanted to reach first SMA.
    first_12_close = close[0:OFFSET1]
    previous_ema = np.mean(first_12_close[0:OFFSET1])
    out["12min_ema"].append(previous_ema)

    # Starts from OFFSET+1 because we used SMA to initiate first EMA.
    for i in range(OFFSET1 + 1, len(close)):
        # Index in close should be same for all of the file. It will be discussed afterward.
        current_ema = ((close[i] - previous_ema) * EMA12_MULTIPLIER) + previous_ema
        out["12min_ema"].append(current_ema)
        previous_ema = current_ema

    return out


def get_26min_ema(daily_data):
    close = daily_data["close"].to_numpy()
    out = {
        "26min_ema": [0] * OFFSET2,
    }

    # We are going to use SMA for the first previous EMA. We wanted to reach first SMA.
    first_26_close = close[0:OFFSET2]
    previous_ema = np.mean(first_26_close[0:OFFSET2])
    out["26min_ema"].append(previous_ema)

    # Starts from OFFSET+1 because we used SMA to initiate first EMA.
    for i in range(OFFSET2 + 1, len(close)):
        # Index in close should be same for all of the file. It will be discussed afterward.
        current_ema = ((close[i] - previous_ema) * EMA26_MULTIPLIER) + previous_ema
        out["26min_ema"].append(current_ema)
        previous_ema = current_ema

    return out


def get_9min_ema(daily_data):
    close = daily_data["close"].to_numpy()
    out = {
        "9min_ema": [0] * OFFSET3,
    }

    # We are going to use SMA for the first previous EMA. We wanted to reach first SMA.
    first_9_close = close[0:OFFSET3]
    previous_ema = np.mean(first_9_close[0:OFFSET3])
    out["9min_ema"].append(previous_ema)

    # Starts from OFFSET+1 because we used SMA to initiate first EMA.
    for i in range(OFFSET3 + 1, len(close)):
        # Index in close should be same for all of the file. It will be discussed afterward.
        current_ema = ((close[i] - previous_ema) * EMA9_MULTIPLIER) + previous_ema
        out["9min_ema"].append(current_ema)
        previous_ema = current_ema

    return out


def get_min_macd_line(ema_data):
    ema_12min = ema_data["12min_ema"].to_numpy()
    ema_26min = ema_data["26min_ema"].to_numpy()

    out = {
        "min_macd_line": [0] * OFFSET2,
    }

    for i in range(OFFSET2, len(ema_12min)):
        out["min_macd_line"].append(ema_12min[i] - ema_26min[i])

    return out


def get_min_macd_signal_line(min_macd_line_data):
    macd_line = min_macd_line_data["min_macd_line"]

    out = {
        "min_signal_macd_line": [0] * (OFFSET2 + OFFSET3)
    }

    first_9_macd_line = macd_line[OFFSET2:OFFSET2+OFFSET3]
    previous_ema = np.mean(first_9_macd_line)
    out["min_signal_macd_line"].append(previous_ema)

    for i in range((OFFSET2+OFFSET3) + 1, len(macd_line)):
        current_ema = ((macd_line[i] - previous_ema) * EMA9_MULTIPLIER) + previous_ema
        out["min_signal_macd_line"].append(current_ema)
        previous_ema = current_ema

    return out

