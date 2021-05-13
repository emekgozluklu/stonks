import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm



pd.set_option("display.precision", 4)

DATA_PATH = os.path.join("..", "data", "imputed", "1min")
GROUPED_BY_DATE_PATH = os.path.join("..", "data", "imputed", "grouped_by_date")
OFFSET = 30  # minutes from start
FILES = os.listdir(DATA_PATH)
FILES_GROUPED_BY_DATE = os.listdir(GROUPED_BY_DATE_PATH)


def get_ohcl_features(data):
    ohcl = pd.DataFrame()
    ohcl["timestamp"] = data["signal_time"]
    ohcl["close"] = data["close"]
    ohcl["high"] = data["high"]
    ohcl["low"] = data["low"]
    ohcl["open"] = data["open"]
    return ohcl


def get_daily_price_based_features(daily_data):
    close = daily_data["close"].to_numpy()  # daily closing prices as np array
    high = daily_data["high"].to_numpy()  # daily high prices as np array
    low = daily_data["low"].to_numpy()  # daily low prices as np array

    # initialize output with first N rows
    out = {
        "keep_row": [False] * OFFSET,  # these rows will be deleted later, first N minutes of a day
        "last_close": [0] + list(close[:OFFSET - 1]),
        "second_last_close": [0, 0] + list(close[:OFFSET - 2]),
        "deviation_from_5_min_avg": [0] * OFFSET,
        "deviation_from_10_min_avg": [0] * OFFSET,
        "deviation_from_15_min_avg": [0] * OFFSET,
        "deviation_from_30_min_avg": [0] * OFFSET,
        "last_5_min_high": [0] * OFFSET,
        "last_10_min_high": [0] * OFFSET,
        "last_30_min_high": [0] * OFFSET,
        "last_5_min_low": [0] * OFFSET,
        "last_10_min_low": [0] * OFFSET,
        "last_30_min_low": [0] * OFFSET,
        "last_5_min_interval": [0] * OFFSET,
        "last_10_min_interval": [0] * OFFSET,
        "last_30_min_interval": [0] * OFFSET,
        "daily_avg_until_now": [0] * OFFSET,
        "deviation_from_daily_avg_until_now": [0] * OFFSET,

    }

    for i in range(30, len(close)):
        subset_close = close[i - 30:i]
        subset_high = high[i - 30:i]
        subset_low = low[i - 30:i]

        out["keep_row"].append(True)

        out["last_close"].append(subset_close[-1])
        out["second_last_close"].append(subset_close[-2])

        out["deviation_from_5_min_avg"].append(close[i] - np.mean(subset_close[-5:]))
        out["deviation_from_10_min_avg"].append(close[i] - np.mean(subset_close[-10:]))
        out["deviation_from_15_min_avg"].append(close[i] - np.mean(subset_close[-15:]))
        out["deviation_from_30_min_avg"].append(close[i] - np.mean(subset_close))

        out["last_5_min_high"].append(np.max(subset_high[-5:]))
        out["last_10_min_high"].append(np.max(subset_high[-10:]))
        out["last_30_min_high"].append(np.max(subset_high[-30:]))

        out["last_5_min_low"].append(np.min(subset_low[-5:]))
        out["last_10_min_low"].append(np.min(subset_low[-10:]))
        out["last_30_min_low"].append(np.min(subset_low[-30:]))

        out["daily_avg_until_now"].append(np.mean(close[:i]))
        out["deviation_from_daily_avg_until_now"].append(close[i] - np.mean(close[:i]))

    out["last_5_min_interval"] = np.array(out["last_5_min_high"]) - np.array(out["last_5_min_low"])
    out["last_10_min_interval"] = np.array(out["last_10_min_high"]) - np.array(out["last_10_min_low"])
    out["last_30_min_interval"] = np.array(out["last_30_min_high"]) - np.array(out["last_30_min_low"])

    return pd.DataFrame(data=out)


def apply_price_based_feature_generator(label):
    try:
        grouped_data = joblib.load(GROUPED_BY_DATE_PATH + f"{label}_full.joblib")
    except FileNotFoundError:
        raise Exception("This label is not included in our dataset.")

    price_based_features = pd.DataFrame()

    for i in tqdm(grouped_data.values()):
        if len(price_based_features) == 0:
            price_based_features = get_daily_price_based_features(i)
        else:
            price_based_features = pd.concat([price_based_features, get_daily_price_based_features(i)], axis=0)
    price_based_features.reset_index(inplace=True)
    return price_based_features


def extract(label):
    raw_data = pd.read_csv(DATA_PATH + label + "_full.csv")

    ohcl = get_ohcl_features(raw_data)
    pbf = apply_price_based_feature_generator(label)

    features = pd.concat([ohcl, pbf], axis=1)

    columns = list(features.columns)
    columns.insert(0, columns.pop(5))
    features = features[columns]

    features = features.rename(columns={"index": "intraday_index"})

    return features
