import os

path = os.path.join("..", "data", "imputed_1min")
files = os.listdir(path)


def get_stock_indices():
    counter = 0
    stock_indices = {}

    for file in files:
        if len(file) > 5:
            stock_indices[file[:4]] = counter
            counter += 1
    return stock_indices


def get_stock_names():
    return [name[:-4] for name in os.listdir(path)]

