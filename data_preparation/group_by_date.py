import pandas as pd
import numpy as np
import joblib
import os

# set read and write data paths
data_path = "../data/imputed/1min"
write_data_path= "../data/imputed/grouped_by_date"

# get the list of files to be read
files = os.listdir(data_path)

# create write directory
try:
    os.mkdir(write_data_path)
except FileExistsError:
    pass


def add_date_column(df):
    try:
        df["date"] = df["date_time"].apply(lambda x: x[:10])
    except KeyError:
        print("Error in file ", file)
    return df


# create joblib
for file in files:

    if not file.endswith(".csv"):
        continue

    label = file.split(".")[0]
    print("Processing: ", label)

    df = pd.read_csv(f"{data_path}/{file}")
    df = add_date_column(df)

    grouped = {}
    for i in df.groupby("date"):
        grouped[i[0]] = i[1]

    joblib.dump(grouped, f"{write_data_path}/{label}.joblib")