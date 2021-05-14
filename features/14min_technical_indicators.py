import os
import numpy as np
import pandas as pd

pd.set_option("display.precision", 4)

DATA_PATH = os.path.join("..", "data", "imputed", "1min")
GROUPED_BY_DATE_PATH = os.path.join("..", "data", "imputed", "grouped_by_date")
OFFSET = 14  # minutes from start
RS_MAX = 10000
MF_RATIO_MAX = 10000
MF_INDEX_MAX = 100
FILES = os.listdir(DATA_PATH)
FILES_GROUPED_BY_DATE = os.listdir(GROUPED_BY_DATE_PATH)


def get_raw_money_flow_data(data):
    raw_money_flow = pd.DataFrame()
    raw_money_flow["raw_money_flow"] = data["volume"]
    return raw_money_flow


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
        "avg_gain": [0] * OFFSET,
        "avg_loss": [0] * OFFSET,
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
    loss = out["loss"].to_numpy()

    for i in range(OFFSET, len(close)):
        subset_gain = gain[i - OFFSET:i]
        subset_loss = loss[i - OFFSET:i]
        avg_gain = np.mean(subset_gain)
        avg_loss = np.mean(subset_loss)
        out["avg_gain"].append(avg_gain)
        out["avg_loss"].append(avg_loss)
        if avg_loss == 0:
            out["relative_strength"].append(9999) # this is unknown
            out["relative_strength_index"].append(100)
        else:
            relative_strength = avg_gain / avg_loss
            out["relative_strength"].append(RS_MAX)
            out["relative_strength_index"].append(100 - (100 / (1 + relative_strength)))

    return out


def get_14min_mfi(typical_price_data, raw_money_flow_data):
    typical_price = typical_price_data["typical_price"].to_numpy()
    # Typical pricela bulmadın direkt excelde olduğunu düşündün

    raw_money_flow = raw_money_flow_data["raw_money_flow"].to_numpy()

    # We want to calculate + and - moneyflows 0 represents no flow, 1 represents + flow, -1 represents 1 flow
    up_or_down = np.array([-2])
    for i in range(1, len(typical_price)):
        if typical_price[i] < typical_price[i-1]:
            up_or_down = np.append(-1)
        elif typical_price[i] > typical_price[i-1]:
            up_or_down = np.append(1)
        else:
            up_or_down = np.append(0)

    positive_money_flow = np.zeros(OFFSET+1)
    negative_money_flow = np.zeros(OFFSET+1)
    money_flow_ratio = np.zeros(OFFSET+1)
    money_flow_index = np.zeros(OFFSET+1)
    for i in range(OFFSET+1, len(typical_price)):
        subset_raw_money_flow = raw_money_flow[i - OFFSET:i]
        subset_up_or_down = up_or_down[i - OFFSET:1]
        current_positive_money_flow = 0
        current_negative_money_flow = 0
        for j in range(0, len(subset_up_or_down)):
            if subset_up_or_down[j] == -1:
                current_negative_money_flow += subset_raw_money_flow[j]
            elif subset_up_or_down[j] == 1:
                current_positive_money_flow += subset_raw_money_flow[j]
        positive_money_flow = np.append(current_positive_money_flow)
        negative_money_flow = np.append(current_negative_money_flow)

        if current_negative_money_flow == 0:
            money_flow_ratio = np.append(MF_RATIO_MAX)
            money_flow_index = np.append(MF_INDEX_MAX)
        else:
            MFR = current_positive_money_flow / current_negative_money_flow
            MFI = 100 - (100 / (1 + MFR))
            money_flow_ratio = np.append(MFR)
            money_flow_index = np.append(MFI)
    out = {
        "min_up_or_down": up_or_down,
        "14min_positive_money_flow": positive_money_flow,
        "14min_negative_money_flow": negative_money_flow,
        "14min_money_flow_ratio": money_flow_ratio,
        "14min_money_flow_index": money_flow_index,
    }

    return out


def get_14min_stochastic_oscillator(daily_data):
    # Slow stochastic starts at OFFSET + 3
    close = daily_data["close"].to_numpy()  # daily closing prices as np array
    fast_stochastic = np.zeros(14)
    out = {
        "fast_stochastic": [],
        "slow_stochastic": [0] * (OFFSET+3),
    }

    for i in range(OFFSET, OFFSET+3):
        subset_close = close[i - OFFSET:i]
        fast_stochastic = np.append(((subset_close[-1]-min(subset_close))/(max(subset_close)-min(subset_close)))*100)

    for i in range(OFFSET+3, len(close)):
        out["slow_stochastic"].append(fast_stochastic[-3:])
        subset_close = close[i - OFFSET:i]
        fast_stochastic = np.append(((subset_close[-1]-min(subset_close))/(max(subset_close) - min(subset_close)))*100)
    out["fast_stochastic"] = fast_stochastic

    return out


def get_min_obv(daily_data, raw_money_flow_data):
    close = daily_data["close"].to_numpy()
    volume = raw_money_flow_data["raw_money_flow"].to_numpy()

    out = {
        "min_obv": [0],
    }

    # Indexing problem may happen
    obv = 0
    for i in range(1, len(close)):
        if close[i] <= close[i-1]:
            obv -= volume[i]
        else:
            obv += volume[i]
        out["min_obv"].append(obv)

    return out
