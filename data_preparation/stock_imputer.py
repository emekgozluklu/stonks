from datetime import datetime
import pandas as pd
import numpy as np
import os


class StockImputer:

    def __init__(self, data):
        self.raw_data = data
        self.imputed_data = []
        self.imputed_data_as_array = None

        self.START_DATE = datetime.strptime(data[0][3], "%Y-%m-%dT%X+03:00")
        self.START_TIMESTAMP = data[0][2]
        self.END_DATE = datetime.strptime(data[-1][3], "%Y-%m-%dT%X+03:00")
        self.END_TIMESTAMP = data[-1][2]

        self.filler_index = 0
        self.parser_index = 0

        self.filler_row = self.raw_data[0].copy()
        self.parser_row = self.raw_data[0]

    def _add_row(self, row, keep_values=False):

        if not keep_values:
            self.filler_row[4:] = row[4:]  # take the values of current row
        else:
            self.filler_row[8:10] = [0, 0]

        self.imputed_data.append(self.filler_row.copy())  # add row to the imputed dataset

        self.filler_row[0] += 1  # inc index
        self.filler_row[2] += 60000  # inc timestamp

        # inc time string
        self.filler_row[3] = datetime.fromtimestamp(self.filler_row[2] / 1000).strftime("%Y-%m-%dT%X+03:00")

    def _add_until_row(self, row):

        while self.filler_row[2] != row[2]:
            self._add_row(row, keep_values=True)  # add rows between

        self._add_row(row, keep_values=False)  # add last row too

    def impute(self):

        for row in self.raw_data:

            if self.filler_row[2] == row[2]:
                self._add_row(row)
                continue

            else:
                if (row[2] - self.filler_row[2]) / 60000 > 30:
                    # day or session changed
                    self.filler_row = row.copy()
                    self._add_row(row)
                    continue

                else:
                    self._add_until_row(row)
                    continue
        self.imputed_data_as_array = np.array(self.imputed_data)


if __name__ == "__main__":

    DATA_PATH = os.path.join("..", "data", "combined")
    stocks = [file[:-4] for file in os.listdir(DATA_PATH)]
    imputed_write_path = os.path.join("..", "data", "imputed_1min")

    try:
        os.mkdir(imputed_write_path)
    except FileExistsError:
        pass

    for stock in stocks:

        historical_df = pd.read_csv(os.path.join(DATA_PATH, f"{stock}.csv"))
        historical_array = historical_df.to_numpy()

        imputer = StockImputer(historical_array)
        imputer.impute()

        imputed_df = pd.DataFrame(data=imputer.imputed_data_as_array, columns=historical_df.columns)

        imputed_df.to_csv(os.path.join(imputed_write_path, stock+".csv"))
        print(f"{stock} done!")
