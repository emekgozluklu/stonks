import pandas as pd
import os

DATA_PATH = os.path.join("..", "data", "2017-01-01_2020-12-28_BIST30_1min", "bar")
DATA_WRITE_PATH = os.path.join("..", "data", "combined")
STOCK_FILE_NAMES = os.listdir(DATA_PATH)

stock_files = dict()

for file in STOCK_FILE_NAMES:
    stock_name = file.split("_")[2]
    if stock_name in stock_files:
        stock_files[stock_name].append(file)
    else:
        stock_files[stock_name] = [file]

# get all stock labels in a list
stock_labels = sorted(list(stock_files.keys()))


def concat_data_in_files(files_of_stock):
    df = pd.DataFrame()
    for f in files_of_stock:
        if df.empty:  # initialize if file is empty
            df = pd.read_csv(os.path.join(DATA_PATH, f))
        else:  # concatenate if file is not empty
            df2 = pd.read_csv(os.path.join(DATA_PATH, f))
            df = pd.concat([df, df2], axis=0)
    return df


def save_dataframe(stock, df, filename):
    try:
        df.to_csv(filename)
    except FileNotFoundError:  # if data write directory does not exist, then create it and continue
        os.mkdir(DATA_WRITE_PATH)
        df.to_csv(filename)
    print(f"{stock} files are combined and saved to {filename}")


def combine_files_of_stock(stock):

    files_of_stock = sorted(stock_files[stock])  # sorted to maintain sequentiality
    combined_data_filename = os.path.join(DATA_WRITE_PATH, stock+".csv")

    # read files and concatenate
    df = concat_data_in_files(files_of_stock)
    save_dataframe(stock, df, combined_data_filename)


def combine_all_files(all_stock_files):
    """
    Combine the data of a stock which is stored in several files. Concatenate them using
    pandas DataFrame library.
    """

    for stock in all_stock_files.keys():
        combine_files_of_stock(stock)
    print("Process finished.")


combine_all_files(stock_files)
