import os
import joblib
import pandas as pd
import sys

data_path = os.path.join("..", "data", "extracted")

stocks = [name[:-4] for name in os.listdir(data_path)]

num = 30
if len(sys.argv) == 2:
    num = int(sys.argv[1])


tensor = pd.DataFrame()

for stock in stocks[:num]:
    df = pd.read_csv(os.path.join(data_path, stock+".csv"))
    tensor = pd.concat([tensor, df], axis=0, ignore_index=True)

tensor.to_csv(os.path.join("..", "data", "tensor.joblib"))
