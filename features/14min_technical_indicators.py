import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


pd.set_option("display.precision", 4)

DATA_PATH = os.path.join("..", "data", "imputed", "1min")
GROUPED_BY_DATE_PATH = os.path.join("..", "data", "imputed", "grouped_by_date")
OFFSET = 14  # minutes from start
FILES = os.listdir(DATA_PATH)
FILES_GROUPED_BY_DATE = os.listdir(GROUPED_BY_DATE_PATH)


def initiate_14min_indicators(daily_data):
    close = daily_data["close"].to_numpy()  # daily closing prices as np array

    # initialize output with first N rows
    out = {
        "keep_row": [False] * OFFSET,  # these rows will be deleted later, first N minutes of a day
    }

    for i in range(OFFSET, len(close)):
        out["keep_row"].append(True)

    return out


def get_14min_rsi(daily_data):
    close = daily_data["close"].to_numpy()

    out = {
        "gain": [0],
        "loss": [0],
        "relative_strength": [0] * OFFSET,
        "relative_strength_index": [0] * OFFSET,
    }

    # Starts from 1 because we set the first index above.
    for i in range(1, len(close)):
        difference = close[i] - close[i-1]
        if difference <= 0:
            out["loss"].append(abs(difference))
            out["gain"].append(0)
        else:
            out["gain"].append(difference)
            out["loss"].append(0)

    gain = out["gain"].to_numpy()

    for i in range(OFFSET, len(close)):
        subset_gain = gain[i - OFFSET:i]
        subset_loss = loss[i - OFFSET:i]

    return out

