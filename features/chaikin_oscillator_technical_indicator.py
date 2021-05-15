import os
import numpy as np
import pandas as pd

pd.set_option("display.precision", 4)

DATA_PATH = os.path.join("..", "data", "imputed", "1min")
GROUPED_BY_DATE_PATH = os.path.join("..", "data", "imputed", "grouped_by_date")

OFFSET1 = 3  # minutes from start
OFFSET2 = 10

FILES = os.listdir(DATA_PATH)
FILES_GROUPED_BY_DATE = os.listdir(GROUPED_BY_DATE_PATH)

EMA3_MULTIPLIER = (2 / (OFFSET1 + 1))
EMA10_MULTIPLIER = (2 / (OFFSET2 + 1))


def get_daily_adl(daily_data):
    close = daily_data["close"].to_numpy()
    high = daily_data["high"].to_numpy()
    low = daily_data["low"].to_numpy()
    volume = daily_data["volume"].to_numpy()

    out = {
        "daily_money_flow_multiplier ": [],
        "daily_money_flow_volume": [],
        "daily_adl": [0],
    }

    daily_money_flow_multiplier = ((close[0] - low[0]) - (high[0] - close[0])) / (high[0] - low[0])
    out["daily_money_flow_multiplier"].append(daily_money_flow_multiplier)
    money_flow_volume = daily_money_flow_multiplier * volume[0]
    out["daily_money_flow_volume"].append(money_flow_volume)

    for i in range(1, len(close)):
        daily_money_flow_multiplier = ((close[i]-low[i])-(high[i]-close[i]))/(high[i]-low[i])
        out["daily_money_flow_multiplier"].append(daily_money_flow_multiplier)
        money_flow_volume = daily_money_flow_multiplier * volume[i]
        out["daily_money_flow_volume"].append(money_flow_volume)
        daily_adl = money_flow_volume + out["daily_money_flow_volume"][i-1]
        out["daily_adl"].append(daily_adl)

    return out

def get_daily_chaikin_oscillator(adl_data):
    adl = adl_data["daily_adl"].to_numpy()

    out = {
        "3day_ema_adl": [0] * (OFFSET1 + 1),
        "10day_ema_adl": [0] * (OFFSET2 + 1),
        "chaikin_oscillator": [0]* (OFFSET2 + 1),
    }

    first_3day_adl = adl[1:OFFSET1]
    previous_3day_ema_adl = np.mean(first_3day_adl)
    out["3day_ema_adl"].append(previous_3day_ema_adl)
    first_10day_adl = adl[1:OFFSET2]
    previous_10day_ema_adl = np.mean(first_10day_adl)
    out["10day_ema_adl"].append(previous_10day_ema_adl)

    for i in range((OFFSET1+2), (OFFSET2+2)):
        current_3day_ema_adl = ((adl[i] - previous_3day_ema_adl) * EMA3_MULTIPLIER ) + previous_3day_ema_adl
        out["3day_ema_adl"].append(current_3day_ema_adl)
        previous_3day_ema_adl = current_3day_ema_adl

    for i in range((OFFSET2+2), len(adl)):
        current_3day_ema_adl = ((adl[i] - previous_3day_ema_adl) * EMA3_MULTIPLIER ) + previous_3day_ema_adl
        out["3day_ema_adl"].append(current_3day_ema_adl)
        previous_3day_ema_adl = current_3day_ema_adl

        current_10day_ema_adl = ((adl[i] - previous_10day_ema_adl) * EMA10_MULTIPLIER) + previous_10day_ema_adl
        out["10day_ema_adl"].append(current_10day_ema_adl)
        previous_10day_ema_adl = current_10day_ema_adl

        out["chaikin_oscillator"].append(previous_3day_ema_adl - previous_10day_ema_adl)

    return out