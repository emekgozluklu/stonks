import pandas as pd
import joblib
import os

# set read and write data paths
data_path = os.path.join("..", "data", "imputed", "grouped_by_date")
write_data_path = os.path.join("..", "data", "imputed", "daily_closing_prices")

# get the list of files to be read
files = os.listdir(data_path)

# create write directory
try:
    os.mkdir(write_data_path)
except FileExistsError:
    pass


# create joblib
for file in files:

    if not file.endswith(".joblib"):
        print(file, " passed.")
        continue

    label = file.split(".")[0]
    print("Processing: ", label)

    grouped_data = joblib.load(os.path.join(data_path, file))

    daily = {
        "date": list(),
        "open": list(),
        "close": list(),
        "high": list(),
        "low": list(),
        "volume": list(),
        "weighted_average": list(),
        "total_quantity": list()
    }

    for i in grouped_data:
        daily["date"].append(i)
        daily["open"].append(grouped_data[i]["open"].iloc[-1])
        daily["high"].append(grouped_data[i]["high"].iloc[-1])
        daily["low"].append(grouped_data[i]["low"].iloc[-1])
        daily["close"].append(grouped_data[i]["close"].iloc[-1])
        daily["volume"].append(grouped_data[i]["volume"].iloc[-1])
        daily["total_quantity"].append(grouped_data[i]["total_quantity"].iloc[-1])
        daily["weighted_average"].append(grouped_data[i]["weighted_average"].iloc[-1])

    joblib.dump(pd.DataFrame(data=daily), os.path.join(write_data_path, label[:-4] + ".joblib"))