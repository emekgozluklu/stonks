from collections import OrderedDict
import pandas as pd
import joblib
import os

# set read and write data paths
DATA_PATH = os.path.join("..", "data", "imputed_1min")
WRITE_DATA_PATH = os.path.join("..", "data", "grouped_by_date")

# get the list of files to be read
try:
    files = os.listdir(DATA_PATH)
except FileNotFoundError:
    raise Exception("You shoul first run the imputer.")

# create write directory
try:
    os.mkdir(WRITE_DATA_PATH)
except FileExistsError:
    pass


def add_date_column(df):
    """ Add date column to the given dataframe to make easier grouping data by date. """
    try:
        df["date"] = df["date_time"].apply(lambda x: x[:10])
    except KeyError:
        print("Error in file ", file)
    return df


def get_label_from_filename(filename):
    return filename.split()[0]


def group_data_by_date(file):
    df = pd.read_csv(os.path.join(DATA_PATH, file))

    df = add_date_column(df)
    grouped = OrderedDict()

    for i in df.groupby("date"):
        grouped[i[0]] = i[1]
    return grouped


if __name__ == "__main__":
    for file in files:

        if not file.endswith(".csv"):
            continue

        label = get_label_from_filename(file)
        print("Processing: ", label)

        grouped_data = group_data_by_date(file)
        joblib.dump(grouped_data, os.path.join(WRITE_DATA_PATH, label[:-4] + ".joblib"))
