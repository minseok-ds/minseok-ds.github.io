import pandas as pd
import yaml
from typing import Any


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """
    YAML 설정 파일을 로드
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        raise
    except Exception as e:
        print(f"Error loading config file: {e}")
        raise


def load_data(config: dict[str, Any]) -> pd.DataFrame:
    """
    설정 파일에 지정된 경로에서 데이터를 로드
    """
    data_path = config["data"]["path"]
    user_col = config["data"]["user_col"]
    item_col = config["data"]["item_col"]

    try:
        data = pd.read_csv(data_path)
        required_cols = [user_col, item_col]
        if "rating_col" in config["data"]:
            required_cols.append(config["data"]["rating_col"])

        if not all(col in data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data.columns]
            raise ValueError(f"Missing required columns in data: {missing_cols}")

        print(f"Data loaded successfully from {data_path}. Shape: {data.shape}")
        return data

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        raise

    except ValueError as v:
        print(f"Data loading error: {v}")
        raise

    except Exception as e:
        print(f"An unexpected error occurred during data loader: {e}")
        raise
