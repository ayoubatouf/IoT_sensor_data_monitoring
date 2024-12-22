from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


MODEL_PARAMS_JSON_PATH = PROJECT_ROOT / "src" / "models" / "best_params.json"
FEATURES_JSON_PATH = PROJECT_ROOT / "src" / "features" / "features.json"
SEARCH_SPACE_JSON_PATH = PROJECT_ROOT / "src" / "models" / "search_space.json"
REQUIRED_COLUMNS_JSON_PATH = PROJECT_ROOT / "src" / "data" / "required_columns.json"
INFERENCE_JSON_INPUT_PATH = PROJECT_ROOT / "data" / "inference" / "inference_input.json"
