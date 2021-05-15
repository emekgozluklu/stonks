import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

pd.set_option("display.precision", 4)

DATA_PATH = os.path.join("..", "data", "imputed_1min")
GROUPED_BY_DATE_PATH = os.path.join("..", "data", "grouped_by_date")

FILES = os.listdir(DATA_PATH)
FILES_GROUPED_BY_DATE = os.listdir(GROUPED_BY_DATE_PATH)


# noinspection PyTypeChecker
def price_based_features_for_day(daily_data, offset=30, previous_close=0):
    close = daily_data["close"].to_numpy()  # daily closing prices as np array
    high = daily_data["high"].to_numpy()  # daily high prices as np array
    low = daily_data["low"].to_numpy()  # daily low prices as np array

    # initialize output with first N rows
    out = {
        "keep_row": [False] * offset,  # these rows will be deleted later, first N minutes of a day
        "last_close": [0] + list(close[:offset - 1]),
        "second_last_close": [0, 0] + list(close[:offset - 2]),
        "last_high": [0] + list(high[:offset - 1]),
        "second_last_high": [0, 0] + list(high[:offset - 2]),
        "last_low": [0] + list(low[:offset - 1]),
        "second_last_low": [0, 0] + list(low[:offset - 2]),
        "deviation_from_5_min_avg": [0] * offset,
        "deviation_from_10_min_avg": [0] * offset,
        "deviation_from_15_min_avg": [0] * offset,
        "deviation_from_30_min_avg": [0] * offset,
        "last_5_min_high": [0] * offset,
        "last_10_min_high": [0] * offset,
        "last_30_min_high": [0] * offset,
        "last_5_min_low": [0] * offset,
        "last_10_min_low": [0] * offset,
        "last_30_min_low": [0] * offset,
        "last_5_min_interval": [0] * offset,
        "last_10_min_interval": [0] * offset,
        "last_30_min_interval": [0] * offset,
        "daily_avg_until_now": [0] * offset,
        "deviation_from_daily_avg_until_now": [0] * offset,
    }

    for i in range(offset, len(close)):
        # Dont include the current OHCL values to the calculation
        # Use values until 1 min before
        subset_close = close[i-30:i]
        subset_high = high[i-30:i]
        subset_low = low[i-30:i]

        out["keep_row"].append(True)

        out["last_close"].append(subset_close[-1])  # last observed closing price value
        out["second_last_close"].append(subset_close[-2])  # second last observed closing price value

        out["last_high"].append(subset_close[-1])  # last observed closing price value
        out["second_last_high"].append(subset_close[-2])  # second last observed closing price value

        out["last_low"].append(subset_close[-1])  # last observed closing price value
        out["second_last_low"].append(subset_close[-2])  # second last observed closing price value

        # deviation of the last observed closing price from averages
        out["deviation_from_5_min_avg"].append(subset_close[-1] - np.mean(subset_close[-5:]))
        out["deviation_from_10_min_avg"].append(subset_close[-1] - np.mean(subset_close[-10:]))
        out["deviation_from_15_min_avg"].append(subset_close[-1] - np.mean(subset_close[-15:]))
        out["deviation_from_30_min_avg"].append(subset_close[-1] - np.mean(subset_close))

        # highest value of last 5 minutes
        out["last_5_min_high"].append(np.max(subset_high[-5:]))
        out["last_10_min_high"].append(np.max(subset_high[-10:]))
        out["last_30_min_high"].append(np.max(subset_high[-30:]))

        # lowest value of last 5 minutes
        out["last_5_min_low"].append(np.min(subset_low[-5:]))
        out["last_10_min_low"].append(np.min(subset_low[-10:]))
        out["last_30_min_low"].append(np.min(subset_low[-30:]))

        # daily average closing price until now
        out["daily_avg_until_now"].append(np.mean(close[:i]))
        out["deviation_from_daily_avg_until_now"].append(subset_close[-1] - np.mean(close[:i]))

    out["last_5_min_interval"] = np.array(out["last_5_min_high"]) - np.array(out["last_5_min_low"])
    out["last_10_min_interval"] = np.array(out["last_10_min_high"]) - np.array(out["last_10_min_low"])
    out["last_30_min_interval"] = np.array(out["last_30_min_high"]) - np.array(out["last_30_min_low"])

    out["previous_day_close"] = [previous_close] * len(close)

    return pd.DataFrame(data=out)


def get_concatenated_price_based_features(grouped_data):
    price_based_features = pd.DataFrame()
    prev_close = None

    for i in tqdm(grouped_data.values()):
        if len(price_based_features) == 0:
            price_based_features = price_based_features_for_day(i)
            prev_close = price_based_features["last_close"].iloc[-1]
        else:
            todays_pbf = price_based_features_for_day(i, previous_close=prev_close)
            price_based_features = pd.concat([price_based_features, todays_pbf], axis=0)
    price_based_features.reset_index(inplace=True)
    return price_based_features


def apply_price_based_feature_generator(label):
    # read grouped data
    try:
        grouped_data = joblib.load(os.path.join(GROUPED_BY_DATE_PATH, f"{label}.joblib"))
    except FileNotFoundError:
        raise Exception("This label is not included in our dataset.")

    price_based_features = get_concatenated_price_based_features(grouped_data)

    return price_based_features


def extract_price_based_feature_for_stock(label):
    features = apply_price_based_feature_generator(label)
    features = features.rename(columns={"index": "intraday_index"})

    return features
