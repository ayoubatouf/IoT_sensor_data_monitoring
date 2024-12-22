import pandas as pd
from config.data import INFERENCE_JSON_INPUT_PATH
from config.paths import INFERECE_INPUT_PATH
from src.utils.file_io import load_from_json


def json_to_csv(json_file_path, csv_file_path):
    json_data = load_from_json(json_file_path)
    df = pd.DataFrame(json_data)
    df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    json_file_path = INFERENCE_JSON_INPUT_PATH
    csv_file_path = INFERECE_INPUT_PATH

    json_to_csv(json_file_path, csv_file_path)
