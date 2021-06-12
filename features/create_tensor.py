import os
import pandas as pd
import sys
from utils.stock_utils import get_stock_indices, get_stock_names
from utils.utils import create_folder

# location where feature csv's to be read
data_path = os.path.join("..", "data", "extracted")

# static variables
stocks = get_stock_names()
indices = get_stock_indices()

prefix_drop_line_count = 50  # drop this much line from the beginning of the file
suffix_drop_line_count = 30  # drop this much line from the end of the file

# default number of stocks is 30 if it is not given while running
num_of_stocks_to_concat = int(sys.argv[1]) if len(sys.argv) == 2 else 30

tensor = pd.DataFrame()

for i in range(num_of_stocks_to_concat):
    stock = stocks[i]
    df = pd.read_csv(os.path.join(data_path, stock+".csv"))
    print("file read")
    tensor = pd.concat([tensor, df], axis=0, ignore_index=True)
    print("concat done")
    tensor.columns = df.columns

print("writing to file")
tensor.to_csv(os.path.join("..", "data", "tensor.joblib"))
