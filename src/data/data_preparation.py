import os

import pandas as pd
from consts import RAW_DIR, BITCOIN_FILE, DEFAULT_PRICE_COLUMN_NAME


def import_data(path):
    data = pd.read_csv(path)
    return data


def remove_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    print('Number of missing values in each column:')
    print(dataframe.isnull().sum())
    print('Rows with at least one value missing: ')
    print(dataframe[dataframe.isnull().any(axis=1)])
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


def choose_and_rename_column(dataframe: pd.DataFrame, column: str = 'Close', new_column_name: str = DEFAULT_PRICE_COLUMN_NAME):
    dataframe = dataframe[[column]]
    dataframe.columns = [new_column_name]
    return dataframe


def split_into_train_and_test_by_ratio(dataframe: pd.DataFrame, ratio: float):
    nrow = len(dataframe)
    print(f'Total samples in dataframe: {nrow}')
    split_row = int(nrow*ratio)
    print('Training samples: ', split_row)
    print('Testing samples: ', nrow - split_row)
    train = dataframe.iloc[:split_row]
    test = dataframe.iloc[split_row:]
    return train, test


def split_into_train_and_test_by_date(dataframe: pd.DataFrame, date: str):
    try:
        train = dataframe[: pd.to_datetime(date, infer_datetime_format=True)]
        test = dataframe[pd.to_datetime(date, infer_datetime_format=True) + pd. Timedelta(days=1):]
        return train, test
    except IndexError as e:
        print(f'I got an IndexError - please check if date {e} is valid index.')


def create_df_with_lags(df: pd.DataFrame, N: int, column: str = DEFAULT_PRICE_COLUMN_NAME):
    for i in range(N):
        df['Lag' + str(i + 1)] = df[column].shift(i + 1)
    df = df.dropna()
    return df  # we are getting here dataframe with price, Lag1, Lag2, ... , LagN columns


def create_numpy_arrays_with_lags(df_with_lags: pd.DataFrame, main_column: str = DEFAULT_PRICE_COLUMN_NAME):
    y = df_with_lags[main_column].values
    X = df_with_lags.iloc[:, 1:].values
    return X, y


bitcoin_all = set_date_column_as_index(import_data(os.path.join(os.curdir, RAW_DIR, BITCOIN_FILE)))

# bitcoin_train, bitcoin_test = split_into_train_and_test_by_date(bitcoin_all, '2019-10-31')
# bitcoin_train_1, bitcoin_test_1 = split_into_train_and_test_by_ratio(bitcoin_all, 0.9)

