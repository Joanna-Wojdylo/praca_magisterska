
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from data.data_preparation import *
from models.SVR.SVR_utils import predict_future, fit_best_SVR_model_using_GridSearch, calculate_metrics
from consts import BITCOIN_FILE, ETHEREUM_FILE, LITECOIN_FILE, RIPPPLE_FILE, TETHER_FILE, CHAINLINK_FILE, NEM_FILE, \
    STELLAR_FILE

#number_of_lags = 30
#number_of_days_ahead = 30


def explore_model(data_file, number_of_lags, number_of_days_ahead, use_scaler: bool = True):
    data_all = import_data(os.path.join(os.curdir, RAW_DIR, data_file))
    data_all = interpolate_missing_values(data_all)
    data_all = set_date_column_as_index(data_all)
    data_all = choose_and_rename_column(data_all, 'Close', DEFAULT_PRICE_COLUMN_NAME)
    data_train, data_test = split_into_train_and_test_by_ratio(data_all, 0.9)

    standard_scaler = None
    if not use_scaler:
        pass
    else:
        standard_scaler = fit_standard_scaler_to_df_column(data_train)
        data_train = apply_scaler_df_column(standard_scaler, data_train,
                                            standardized_column_name=DEFAULT_PRICE_COLUMN_NAME)

    data_train_with_lags = create_df_with_lags(data_train, number_of_lags, column=DEFAULT_PRICE_COLUMN_NAME)
    X, y = create_numpy_arrays_with_lags(data_train_with_lags, main_column=DEFAULT_PRICE_COLUMN_NAME)

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=1)
    svr_model = fit_best_SVR_model_using_GridSearch(X_train, y_train)
    #svr_model = SVR(kernel='rbf', C=10, gamma=1e-02)  # bitcoin 1000, 1e-04
    #svr_model.fit(X_train, y_train)

    y_test = data_test[DEFAULT_PRICE_COLUMN_NAME].values[:number_of_days_ahead]
    if not use_scaler:
        y_pred = predict_future(data_train, number_of_lags, svr_model, number_of_days_ahead)[
            DEFAULT_PRICE_COLUMN_NAME].values
    else:
        y_pred = inverse_scaler_on_df_column(standard_scaler,
                                             predict_future(data_train, number_of_lags, svr_model,
                                                            number_of_days_ahead))[DEFAULT_PRICE_COLUMN_NAME].values
    # on train values
    train_metrics = calculate_metrics(y_train, svr_model.predict(X_train))

    # on validate values
    validate_metrics = calculate_metrics(y_validate, svr_model.predict(X_validate))

    # on test values

    test_metrics = calculate_metrics(y_test, y_pred)
    return svr_model.best_params_, train_metrics, validate_metrics, test_metrics

"""
for data_file in [BITCOIN_FILE, ETHEREUM_FILE, TETHER_FILE, RIPPPLE_FILE, LITECOIN_FILE, STELLAR_FILE]:
    for number_of_days_ahead in [10, 30, 90, 150]:
        for number_of_lags in [10, 30, 90, 150, 365]:
            for use_scaler in [True, False]:
                explore_model(data_file, number_of_lags, number_of_days_ahead, use_scaler)
"""


def compare_models():
    columns_naming = ['data', 'predicted days', 'lags used', 'standardized', 'chosen model', 'RMSE', 'MAE', 'MAPE', 'explained variance score', 'R2', 'residuals mean', 'residuals std']
    train_table = []
    validate_table = []
    test_table = []
    for data_file in [STELLAR_FILE]: #, ETHEREUM_FILE, TETHER_FILE, RIPPPLE_FILE, LITECOIN_FILE, STELLAR_FILE]:
        for number_of_days_ahead in [1, 3, 5, 7, 10, 14, 30, 60]:
            print(f'******************* DOING {number_of_days_ahead} DAYS AHEAD *********************')
            for number_of_lags in [7, 14, 21, 30, 90, 150]:
                print(f'******************* DOING {number_of_lags} LAGS *********************')
                for use_scaler in [True, False]:
                    best_score, train_metrics, validate_metrics, test_metrics = explore_model(data_file, number_of_lags, number_of_days_ahead, use_scaler)
                    train_table.append([data_file, number_of_days_ahead, number_of_lags, use_scaler, best_score] + train_metrics)
                    validate_table.append([data_file, number_of_days_ahead, number_of_lags, use_scaler, best_score] + validate_metrics)
                    test_table.append([data_file, number_of_days_ahead, number_of_lags, use_scaler, best_score] + test_metrics)
        train_df = pd.DataFrame.from_records(train_table)
        train_df.columns = columns_naming
        train_df.to_csv('data/processed/train_df_scores.csv', index=False)
        validate_df = pd.DataFrame.from_records(validate_table)
        validate_df.columns = columns_naming
        validate_df.to_csv('data/processed/validate_df_scores.csv', index=False)
        test_df = pd.DataFrame.from_records(test_table)
        test_df.columns = columns_naming
        test_df.to_csv('data/processed/test_df_scores.csv', index=False)

    '''
    print(train_df.head())
    print('---------------------')
    print(validate_df.head())
    print('---------------------')
    print(test_df.head())
    '''

compare_models()