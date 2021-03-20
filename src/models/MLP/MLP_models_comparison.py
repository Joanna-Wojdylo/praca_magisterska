
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from data.data_preparation import *
from models.metrics_utils import calculate_metrics, predict_future
from models.MLP.MLP_utils import fit_best_MLP_model_using_GridSearch
from consts import BITCOIN_FILE, ETHEREUM_FILE, LITECOIN_FILE, RIPPLE_FILE, TETHER_FILE, CHAINLINK_FILE, NEM_FILE, \
    STELLAR_FILE


def explore_model(data_file, number_of_lags, use_scaler: int = 0):
    data_all = import_data(os.path.join(os.curdir, RAW_DIR, data_file))
    data_all = interpolate_missing_values(data_all)
    data_all = set_date_column_as_index(data_all)
    data_all = choose_and_rename_column(data_all, 'Close', DEFAULT_PRICE_COLUMN_NAME)
    data_train, data_test = split_into_train_and_test_by_ratio(data_all, 0.9)

    scaler = None
    if use_scaler == 0:
        pass
    elif use_scaler == 1:
        scaler = fit_standard_scaler_to_df_column(data_train)
        data_train = apply_scaler_df_column(scaler, data_train,
                                            standardized_column_name=DEFAULT_PRICE_COLUMN_NAME)
    elif use_scaler == 2:
        scaler = fit_minmax_scaler_to_df_column(data_train)
        data_train = apply_scaler_df_column(scaler, data_train, DEFAULT_PRICE_COLUMN_NAME, DEFAULT_PRICE_COLUMN_NAME)
    else:
        print('Incorrect use scaler parameter, acceptable values are 0, 1 or 2.')
        return None

    data_train_with_lags = create_df_with_lags(data_train, number_of_lags, column=DEFAULT_PRICE_COLUMN_NAME)
    X, y = create_numpy_arrays_with_lags(data_train_with_lags, main_column=DEFAULT_PRICE_COLUMN_NAME)

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=1)
    mlp_model = fit_best_MLP_model_using_GridSearch(X_train, y_train)
    #mlp_model = SVR(kernel='rbf', C=10, gamma=1e-02)  # bitcoin 1000, 1e-04
    #mlp_model.fit(X_train, y_train)

    if use_scaler == 0:
        train_metrics = calculate_metrics(y_train, mlp_model.predict(X_train))
        validate_metrics = calculate_metrics(y_validate, mlp_model.predict(X_validate))
    else:
        train_metrics = calculate_metrics(scaler.inverse_transform(y_train.reshape(-1, 1)), scaler.inverse_transform(mlp_model.predict(X_train).reshape(-1, 1)))
        validate_metrics = calculate_metrics(scaler.inverse_transform(y_validate.reshape(-1, 1)), scaler.inverse_transform(mlp_model.predict(X_validate).reshape(-1, 1)))

    return mlp_model.best_params_, train_metrics, validate_metrics

"""
for data_file in [BITCOIN_FILE, ETHEREUM_FILE, TETHER_FILE, RIPPLE_FILE, LITECOIN_FILE, STELLAR_FILE]:
    for number_of_days_ahead in [10, 30, 90, 150]:
        for number_of_lags in [10, 30, 90, 150, 365]:
            for use_scaler in [True, False]:
                explore_model(data_file, number_of_lags, number_of_days_ahead, use_scaler)
"""


def compare_models():
    columns_naming = ['data', 'lags used', 'transformation', 'chosen model', 'RMSE', 'MAE', 'MAPE', 'R2']
    train_table = []
    validate_table = []
    test_table = []
    for data_file in [STELLAR_FILE  ]: #, ETHEREUM_FILE, TETHER_FILE, RIPPLE_FILE, LITECOIN_FILE, STELLAR_FILE]:
        for number_of_lags in [7, 14, 21, 30, 90, 150]:
            print(f'******************* DOING {number_of_lags} LAGS *********************')
            for use_scaler in [0, 1, 2]:
                if use_scaler == 0:
                    scaler_text = 'Without transformation'
                elif use_scaler == 1:
                    scaler_text = 'Standardized'
                else:
                    scaler_text = 'Normalized into (-1,1)'
                best_score, train_metrics, validate_metrics = explore_model(data_file, number_of_lags, use_scaler)
                train_table.append([data_file, number_of_lags, scaler_text, best_score] + train_metrics)
                validate_table.append([data_file, number_of_lags, scaler_text, best_score] + validate_metrics)
    train_df = pd.DataFrame.from_records(train_table)
    train_df.columns = columns_naming
    train_df.to_csv('data/processed/train_df_scores.csv', index=False)
    validate_df = pd.DataFrame.from_records(validate_table)
    validate_df.columns = columns_naming
    validate_df.to_csv('data/processed/validate_df_scores.csv', index=False)

'''
print(train_df.head())
print('---------------------')
print(validate_df.head())
print('---------------------')
print(test_df.head())
'''

compare_models()
#explore_model(BITCOIN_FILE, 10, use_scaler=2)