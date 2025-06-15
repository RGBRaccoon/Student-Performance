from typing import List

import pandas as pd


def load_train_data():
    train_data = pd.read_csv("archive/train.csv")
    return train_data


def load_validation_data():
    validation_data = pd.read_csv("archive/validation.csv")
    return validation_data


def load_test_data():
    test_data = pd.read_csv("archive/test.csv")
    return test_data


def data_preprocess(features: List[str], target: str):
    train_data = load_train_data()
    validation_data = load_validation_data()
    test_data = load_test_data()
    train_data = train_data
    validation_data = validation_data
    test_data = test_data

    train_x = data_one_hot_encoder(train_data[features])
    train_y = train_data[target]
    validation_x = data_one_hot_encoder(validation_data[features])
    validation_y = validation_data[target]
    test_x = data_one_hot_encoder(test_data[features])
    test_y = test_data[target]

    return train_x, train_y, validation_x, validation_y, test_x, test_y


def data_one_hot_encoder(data: pd.DataFrame):
    categorical_columns = ["ParentalEducation", "Locale", "SchoolType", "Gender", "Race"]
    # 실제 데이터에 존재하는 칼럼만 필터링
    existing_columns = [col for col in categorical_columns if col in data.columns]
    if not existing_columns:
        return data
    df_encoded = pd.get_dummies(data, columns=existing_columns)
    return df_encoded
