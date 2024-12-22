import pandas as pd
from config.config import RAW_DATA_PATH, PRODUCTION_LOG_PATH, FEATURES_JSON_PATH
from src.utils.data_drift import detect_drift
from src.utils.file_io import load_from_json

if __name__ == "__main__":

    columns_of_interest = load_from_json(FEATURES_JSON_PATH)["columns_to_keep"]
    columns_of_interest.remove("fail")
    df1 = pd.read_csv(RAW_DATA_PATH)
    df2 = pd.read_csv(PRODUCTION_LOG_PATH)

    drift_exists = detect_drift(df1, df2, columns_of_interest)
    print("Drift detected across any column:", drift_exists)
