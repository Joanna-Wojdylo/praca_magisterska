import os

import pandas as pd
from consts import RAW_DIR, BITCOIN_FILE, DEFAULT_PRICE_COLUMN_NAME, STANDARDIZED_PRICE_COLUMN_NAME
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def import_data(path):
    data = pd.read_csv(path)
    return data


def interpolate_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    #print('Number of missing values in each column:')
    #print(dataframe.isnull().sum())
    #print('Rows with at least one value missing: ')
    #print(dataframe[dataframe.isnull().any(axis=1)])
    # interpolate missing values (mean of previous and following value)
    dataframe = dataframe.interpolate(method='time')
    return dataframe


def remove_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    #print('Number of missing values in each column:')
    #print(dataframe.isnull().sum())
    #print('Rows with at least one value missing: ')
    #print(dataframe[dataframe.isnull().any(axis=1)])
    dataframe = dataframe.dropna()  # remove row with at least one NaN
    return dataframe


def set_date_column_as_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    try:
        # choose datetime format automatically and set it as index
        dataframe['Date'] = pd.to_datetime(dataframe['Date'], infer_datetime_format=True)
        dataframe = dataframe.set_index('Date')
        return dataframe
    except KeyError as e:
        print(f'I got a KeyError - please check if passed dataframe has {e} column.')


def choose_and_rename_column(df: pd.DataFrame, column: str = 'Close', new_column_name: str = DEFAULT_PRICE_COLUMN_NAME):
    dataframe = df[[column]]
    dataframe.columns = [new_column_name]
    return dataframe


def split_into_train_and_test_by_ratio(df: pd.DataFrame, ratio: float):
    dataframe = df.copy()
    nrow = len(dataframe)
    #print(f'Total samples in dataframe: {nrow}')
    split_row = int(nrow*ratio)
    #print('Training samples: ', split_row)
    # print('Testing samples: ', nrow - split_row)
    train = dataframe.iloc[:split_row]
    test = dataframe.iloc[split_row:]
    return train, test


def split_into_train_and_test_by_date(df: pd.DataFrame, date: str):
    dataframe = df.copy()
    try:
        train = dataframe[: pd.to_datetime(date, infer_datetime_format=True)]
        test = dataframe[pd.to_datetime(date, infer_datetime_format=True) + pd. Timedelta(days=1):]
        return train, test
    except IndexError as e:
        print(f'I got an IndexError - please check if date {e} is valid index.')


def create_df_with_lags(dataframe: pd.DataFrame, N: int, column: str = DEFAULT_PRICE_COLUMN_NAME):
    df = dataframe.copy()
    for i in range(N):
        df['Lag' + str(i + 1)] = df[column].shift(i + 1)
    df = df.dropna()
    return df  # we are getting here dataframe with price, Lag1, Lag2, ... , LagN columns


def create_numpy_arrays_with_lags(df_with_lags: pd.DataFrame, main_column: str = DEFAULT_PRICE_COLUMN_NAME):
    y = df_with_lags[main_column].values
    X = df_with_lags.iloc[:, 1:].values
    return X, y


def fit_standard_scaler_to_df_column(df: pd.DataFrame, column: str = DEFAULT_PRICE_COLUMN_NAME):
    standard_scaler = StandardScaler().fit(df[[column]])
    return standard_scaler


def fit_minmax_scaler_to_df_column(df: pd.DataFrame, column: str = DEFAULT_PRICE_COLUMN_NAME):
    minmax_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(df[[column]])
    return minmax_scaler


def apply_scaler_df_column(scaler, dataframe: pd.DataFrame, column: str = DEFAULT_PRICE_COLUMN_NAME, standardized_column_name: str = STANDARDIZED_PRICE_COLUMN_NAME):
    df = dataframe.copy()
    df[[standardized_column_name]] = scaler.transform(df[[column]])
    return df


def inverse_scaler_on_df_column(scaler, dataframe: pd.DataFrame, column: str = DEFAULT_PRICE_COLUMN_NAME):
    df = dataframe.copy()
    df[[column]] = scaler.inverse_transform(df[[column]])
    return df




# bitcoin_train, bitcoin_test = split_into_train_and_test_by_date(bitcoin_all, '2019-10-31')
# bitcoin_train_1, bitcoin_test_1 = split_into_train_and_test_by_ratio(bitcoin_all, 0.9)

