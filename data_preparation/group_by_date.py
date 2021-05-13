from collections import OrderedDict
import pandas as pd
import joblib
import os

# set read and write data paths
data_path = os.path.join("..","data","imputed","1min")
write_data_path= os.path.join("..","data","imputed","grouped_by_date")

# ters slasha cevir


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

    df = pd.read_csv(os.path.join(data_path,file))
    #ters slah at

    df = add_date_column(df)

    grouped = OrderedDict()
    for i in df.groupby("date"):
        grouped[i[0]] = i[1]

    joblib.dump(grouped, os.path.join(write_data_path,label + ".joblib"))

