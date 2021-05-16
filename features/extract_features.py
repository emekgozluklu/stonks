import technical_indicator_features as indicators
import price_based_features as price
import pandas as pd
import os

stocks = [name.split(".")[0] for name in os.listdir(os.path.join("..", "data", "daily"))]
write_path = os.path.join("..", "data", "extracted")

try:
    os.mkdir(write_path)
except FileExistsError:
    pass

for stock in stocks:
    print(f"Running for {stock}", end="...")
    pbf = price.extract_price_based_feature_for_stock(stock)
    pbf_cols = pbf.columns

    ind = indicators.TechnicalAnalysisFeatures(stock)
    tif = ind.run_all()
    tif_cols = tif.columns

    all_cols = list(pbf_cols) + list(tif_cols)

    features = pd.concat([pbf, tif], axis=1, ignore_index=True)
    features.columns = all_cols

    features.to_csv(os.path.join(write_path, stock+".csv"))
    print(" Done!")
