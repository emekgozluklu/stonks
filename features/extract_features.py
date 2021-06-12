import joblib

from utils.stock_utils import get_stock_names, get_stock_indices
from config import PREFIX_OFFSET, POSTFIX_OFFSET
import technical_indicator_features as indicators
import price_based_features as price
from data_preparation.normalization import normalize_features
from utils.utils import create_folder

import pandas as pd
import numpy as np
import os


stocks = get_stock_names()
write_path = os.path.join("..", "data", "extracted")
scaler_write_path = os.path.join("..", "data", "scalers")

create_folder(write_path)
create_folder(scaler_write_path)


def apply_price_based_features(stock_label):
    price_based_features = price.extract_price_based_feature_for_stock(stock_label)
    columns = price_based_features.columns
    return price_based_features, columns


def apply_technical_analysis_features(stock_label):
    ta_feature_extractor = indicators.TechnicalAnalysisFeatures(stock_label)
    technical_analysis_features = ta_feature_extractor.run_all()
    columns = technical_analysis_features.columns
    return technical_analysis_features, columns


def send_label_to_end(data):
    cols = list(data.columns)
    try:
        cols.remove("label")
        cols.append("label")
    except (KeyError, IndexError):
        pass
    return data[cols]


def add_stock_name_ohe_feature(df, stock_name):
    stock_names = get_stock_names()
    num_of_rows = df.shape[0]

    for st in stock_names:
        if st == stock_name:
            df[st] = np.ones((num_of_rows,))
        else:
            df[st] = np.zeros((num_of_rows,))

    return df


def concatenate_features(price_based_feat, ta_feat, all_columns):
    combined_features = pd.concat([price_based_feat, ta_feat], axis=1, ignore_index=True)
    combined_features.columns = all_columns
    combined_features = send_label_to_end(combined_features)
    return combined_features


def drop_unnecessary_rows(df):
    return df.drop(df[~df["keep_row"]].index)


if __name__ == "__main__":
    for stock in stocks:
        print(f"Running for {stock}", end="...\n")

        pbf, pbf_cols = apply_price_based_features(stock)
        pbf = add_stock_name_ohe_feature(pbf, stock)

        tif, tif_cols = apply_technical_analysis_features(stock)

        all_cols = list(pbf_cols) + get_stock_names() + list(tif_cols)

        features = concatenate_features(pbf, tif, all_cols)
        features = drop_unnecessary_rows(features)

        features, scaler = normalize_features(features)

        features.to_csv(os.path.join(write_path, stock+".csv"))
        joblib.dump(scaler, os.path.join(scaler_write_path, stock+"_scaler.joblib"))

        print(" Done!")
