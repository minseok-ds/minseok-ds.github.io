import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class DataPreproceessor:
    """
    추천 시스템 학습에 사용될 데이터를 전처리하는 클래스입니다.
    사용자 ID 와 아이템 ID 를 연속적인 정수 인덱스로 변환하고, 학습 / 테스트 데이터셋으로 분리합니다.

    주요 기능:
    1. 사용자/아이템 ID 인코딩 : 문자열 또는 비연속적인 ID 를 0부터 시작하는 정수 인덱스로 매핑합니다.
    2. 데이터 분할 : 데이터를 학습 데이터셋과 테스트 데이터셋으로 분리합니다.
    3. 매핑 정보 저장/로드 : 생성된 인코더 (매핑 정보)를 저장하고 로드하여 일관된 변환을 보장합니다.

    Attributes:
        user_encoder (LabelEncoder): 사용자 ID 를 인덱스로 변환되는 인코더.
        item_encoder (LabelEncoder): 아이템 ID 를 인덱스로 변환되는 인코더.
        user_col (str): 사용자 ID 컬럼명.
        item_col (str): 아이템 ID 컬럼명.
        rating_col (str): 평점 컬럼명.
        n_users (int): 고유한 사용자 수.
        n_items (int): 고유한 아이템 수.
    """

    def __init__(
        self,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
    ):
        """
        DataPreprocessor 객체를 초기화합니다.

        Args:
            user_col (str): 데이터프레임에서 사용자 ID 를 나타내는 컬럼의 이름
            item_col (str): 데이터프레임에서 아이템 ID 를 나타내는 컬럼의 이름
            rating_col (str): 데이터프레임에서 평점을 나타내는 컬럼의 이름
        """
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.n_users = 0
        self.n_items = 0

        print(
            f"전치리기 초기화 완료. 사용자 컬럼: '{user_col}', 아이템 컬럼: '{item_col}', 평점 컬럼: '{rating_col}'"
        )

    def fit(self, data: pd.DataFrame):
        """
        데이터 프레임을 기반으로 사용자/아이템 인코더를 학습시킵니다.

        Args:
            data (pd.DataFrame): 사용자 ID, 아이템 ID, 평점 컬럼을 포함하는 데이터프레임.
        """
        # NaN 값이 있는 행 제거
        data = data[[self.user_col, self.item_col, self.rating_col]].dropna()

        # 사용자 ID 와 아이템 ID 를 정수 인덱스로 변환
        self.user_encoder.fit(data[self.user_col])
        self.item_encoder.fit(data[self.item_col])

        # 고유 사용자 및 아이템 수 저장
        self.n_users = len(self.user_encoder.classes_)
        self.n_items = len(self.item_encoder.classes_)

        print(
            f"인코더 학습 완료. 고유 사용자 수: {self.n_users}, 고유 아이템 수: {self.n_items}"
        )
        return self

    def transform(self, data: pd.DataFrame):
        """
        학습된 인코더를 사용하여 데이터프레임의 사용자/아이템 ID 를 변환합니다.
        학습 시 보지 못했던 ID 는 변환되지 않고 해당 행이 제거될 수 있습니다.

        Args:
            data (pd.DataFrame): 변환할 데이터프레임.
        """
        data_transformed = data.copy()
        # NaN 값이 있는 행 제거
        data_transformed = data_transformed[
            [self.user_col, self.item_col, self.rating_col]
        ].dropna()

        # 인코딩 전 데이터 타입 확인 및 변환 (LabelEncoder 는 문자열이나 정수형을 기대)
        data_transformed[self.user_col] = data_transformed[self.user_col].astype(str)
        data_transformed[self.item_col] = data_transformed[self.item_col].astype(str)

        # 학습된 인코더에 없는 새로운 ID 가 포함된 행을 식별하고 제거
        valid_users = data_transformed[self.user_col].isin(self.user_encoder.classes_)
        valid_items = data_transformed[self.item_col].isin(self.item_encoder.classes_)
        data_transformed = data_transformed[valid_users & valid_items]

        if data_transformed.empty:
            print(
                "경고: 변환 후 데이터가 비어있습니다. 입력 데이터에 학습된 인코더에 없는 ID 만 포함되어 있을 수 있습니다."
            )
            return pd.DataFrame(columns=["user_idx", "item_idx", self.rating_col])

        data_transformed["user_idx"] = self.user_encoder.transform(
            data_transformed[self.user_col]
        )
        data_transformed["item_idx"] = self.item_encoder.transform(
            data_transformed[self.item_col]
        )

        print(
            f"데이터 변환 완료. 원본 행 수: {len(data)}, 변환 후 행 수: {len(data_transformed)}"
        )

        return data_transformed[["user_idx", "item_idx", self.rating_col]]

    def fit_transform(self, data: pd.DataFrame):
        """
        fit() 과 transform() 을 순차적으로 실행합니다.

        Args:
            data (pd.DataFrame): 학습 및 변환할 데이터프레임.
        """
        return self.fit(data).transform(data)

    def split_data(
        self,
        data_processed: pd.DataFrame,
        test_size: float = 0.2,
        random_state=20250422,
    ):
        """
        처리된 데이터를 학습 데이터셋과 테스트 데이터셋으로 분할합니다.

        Args:
            data_processed (pd.DataFrame): user_idx, item_idx, rating 컬럼을 포함하는 전처리된 데이터프레임.
            test_size (float): 테스트 데이터셋의 비율 (0 ~ 1).
            random_state (int): 재현성을 위한 난수 시드.
        """
        if data_processed.empty:
            print("경고: 데이터 분할 시 입력 데이터가 비어있습니다.")
            return pd.DataFrame(columns=data_processed.columns), pd.DataFrame(
                columns=data_processed.columns
            )

        train_data, test_data = train_test_split(
            data_processed,
            test_size=test_size,
            random_state=random_state,
            stratify=data_processed["user_idx"],  # 사용자를 기준으로 계층적 샘플링
        )

        print(
            f"데이터 분할 완료. 학습 데이터 수: {len(train_data)}, 테스트 데이터 수: {len(test_data)}"
        )

        return train_data, test_data

    def save(self, filepath: str):
        """
        학습된 인코더와 설정 정보를 파일에 저장합니다.

        Args:
            filepath (str): 저장할 파일 경로 (확장자 없이 이름만 지정 권장).
                            '.joblib' 확장자가 자동으로 추가됩니다.
        """
        data_to_save = {
            "user_encoder": self.user_encoder,
            "item_encoder": self.item_encoder,
            "user_col": self.user_col,
            "item_col": self.item_col,
            "rating_col": self.rating_col,
            "n_users": self.n_users,
            "n_items": self.n_items,
        }

        joblib.dump(data_to_save, filepath + ".joblib")
        print(f"전처리기 정보가 '{filepath}.joblib'에 저장되었습니다.")

    @classmethod
    def load(cls, filepath: str):
        """
        파일로부터 인코더와 설정 정보를 로드하여 DataPreprocessor 객체를 생성합니다.

        Args:
            filepath (str): 로드할 파일 경로 (확장자 없이 이름만 지정 권장).
                            '.joblib' 확장자가 자동으로 추가됩니다.
        """
        loaded_data = joblib.load(filepath + ".joblib")
        preprocessor = cls(
            user_col=loaded_data["user_col"],
            item_col=loaded_data["item_col"],
            rating_col=loaded_data["rating_col"],
        )

        preprocessor.user_encoder = loaded_data["user_encoder"]
        preprocessor.item_encoder = loaded_data["item_encoder"]
        preprocessor.n_users = loaded_data["n_users"]
        preprocessor.n_items = loaded_data["n_items"]

        print(f"전처리기 정보를 '{filepath}.joblib'에서 로드했습니다.")
        return preprocessor

    def inverse_transform_user(self, user_indices: np.ndarray):
        """
        사용자 인덱스를 원래 ID 로 변환합니다.

        Args:
            user_indices (np.ndarray) : 변환할 사용자 인덱스 배열.
        """
        return self.user_encoder.inverse_transform(user_indices)

    def inverse_transform_item(self, item_indices: np.ndarray):
        """
        아이템 인덱스를 원래 ID 로 변환합니다.

        Args:
            item_indices (np.ndarray) : 변환할 아이템 인덱스 배열.
        """
        return self.item_encoder.inverse_transform(item_indices)


class BaseRecommender:
    """
    모든 추천 모델의 기본 클래스 입니다.
    공통 인터페이스 (fit, predict) 를 정의합니다.

    Attributes:
        n_users (int): 총 사용자 수.
        n_items (int): 총 아이템 수.
        model: 실제 추천 알고리즘 모델 객체.
    """

    def __init__(self, n_users: int, n_items: int):
        """
        BaseRecommender 객체를 초기화합니다.

        Args:
            n_users (int): 데이터셋의 고유 사용자 수.
            n_items (int): 데이터셋의 고유 아이템 수.
        """
        self.n_users = n_users
        self.n_items = n_items
        self.model = None

        # Device 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"기본 추천 모델 초기화. 사용자 수: {n_users}, 아이템 수: {n_items}, 사용 장치: {self.device}"
        )

    def fit(self, train_data: pd.DataFrame):
        """
        학습 데이터를 사용하여 추천 모델을 학습시킵니다.
        하위 클래스에서 반드시 구현해야 합니다.

        Args:
            train_data (pd.DataFrame): 'user_idx', 'item_idx', 'rating" 컬럼을 포함하는 학습 데이터.
        """
        raise NotImplementedError("fit 메소드는 하위 클래에서 구현되어야 합니다.")

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray):
        """
        주어진 사용자-아이템 쌍에 대한 평점을 예측합니다.
        하위 클래스에서 반드시 구현해야 합니다.

        Args:
            user_idx (np.ndarray): 예측할 사용자 인덱스 배열.
            item_idx (np.ndarray): 예측할 아이템 인덱스 배열.
        """
        raise NotImplementedError("predict 메소드는 하위 클래스에서 구현되어야 합니다.")

    def save(self, filepath: str):
        """
        학습된 모델을 파일에 저장합니다.
        모델 저장 방식은 하위 클래스에 따라 다를 수 있습니다.

        Args:
            filepath (str): 모델을 저장할 파일 경로.
        """
        raise NotImplementedError("save 메소드는 하위 클래스에서 구현되어야 합니다.")

    @classmethod
    def load(
        cls,
        filepath: str,
        n_users: int | None = None,
        n_items: int | None = None,
    ):
        """
        파일로부터 학습된 모델을 로드합니다.
        모델 로드 방식은 하위 클래스에 따라 다를 수 있습니다.
        일부 모델은 로드 시 n_users, n_items 정보가 필요할 수 있습니다.

        Args:
            filepath (str): 모델을 로드할 파일 경로.
            n_users (int, optional): 로드 시 필요한 총 사용자 수.
            n_items (int, optional): 로드 시 필요한 총 아이템 수.
        """
        raise NotImplementedError("load 메소드는 하위 클래스에서 구현되어야 합니다.")
