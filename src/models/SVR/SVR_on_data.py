import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics
from data.data_preparation import *
from models.SVR.SVR_utils import predict_future, fit_best_SVR_model_using_GridSearch, calculate_metrics
from consts import BITCOIN_FILE, ETHEREUM_FILE, LITECOIN_FILE, RIPPLE_FILE, TETHER_FILE, STELLAR_FILE

def perform_SVR_on_data(data_file: str, number_of_lags: int, use_scaler: int, C:int, gamma:float, number_of_days_ahead_in_recursive_forecast: int ):
    number_of_days_ahead = number_of_days_ahead_in_recursive_forecast


    data_all = import_data(os.path.join(os.curdir, RAW_DIR, data_file))
    #print('number of days: ', data_all.shape)
    data_all = interpolate_missing_values(data_all)
    data_all = set_date_column_as_index(data_all)
    data_all = choose_and_rename_column(data_all, 'Close', DEFAULT_PRICE_COLUMN_NAME)
    data_train, data_test = split_into_train_and_test_by_ratio(data_all, 0.9)
    #print('number of train days: ', data_train.shape)
    #print('number of test days: ', data_test.shape)
    if use_scaler == 0:
        # NO SCALER
        scaler = None
    elif use_scaler == 1:
        # STANDARD SCALER
        scaler = fit_standard_scaler_to_df_column(data_train)
        data_train = apply_scaler_df_column(scaler, data_train, standardized_column_name=DEFAULT_PRICE_COLUMN_NAME)
    elif use_scaler == 2:
        # MIN MAX SCALER (-1,1)
        scaler = fit_minmax_scaler_to_df_column(data_train)
        data_train = apply_scaler_df_column(scaler, data_train, DEFAULT_PRICE_COLUMN_NAME, DEFAULT_PRICE_COLUMN_NAME)
    else:
        print('Invalid use scaler parameter. Valid options: 0 (None), 1 (Standard scaler), 2 (Minmax scaler)')
        return None

    data_train_with_lags = create_df_with_lags(data_train, number_of_lags, column=DEFAULT_PRICE_COLUMN_NAME)
    X, y = create_numpy_arrays_with_lags(data_train_with_lags, main_column=DEFAULT_PRICE_COLUMN_NAME)

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=1)
    # print('number of train days: ', X_train.shape)
    # print('number of validate days: ', X_validate.shape)
    #svr_model = fit_best_SVR_model_using_GridSearch(X_train, y_train)
    svr_model = SVR(kernel='rbf', C=C, gamma=gamma)
    svr_model.fit(X_train, y_train)

    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_validate, color='blue', label='y')
    plt.plot(svr_model.predict(X_validate), color= 'black', label= 'predicted')
    
    #plt.plot(dates_train, clf.predict(X_test), color= 'red', label= 'RBF model')
    #plt.plot(dates_test, scaler.inverse_transform(svr_rbf.predict(dates_test)), color= 'green', label= 'RBF model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    """
    if scaler:
        y_pred = inverse_scaler_on_df_column(scaler, predict_future(data_train, number_of_lags, svr_model, number_of_days_ahead))[DEFAULT_PRICE_COLUMN_NAME].values
    else:
        y_pred = predict_future(data_train, number_of_lags, svr_model, number_of_days_ahead)[
            DEFAULT_PRICE_COLUMN_NAME].values
    y_test = data_test[DEFAULT_PRICE_COLUMN_NAME].values[:number_of_days_ahead]

    #print('on train values **************')
    #print( calculate_metrics(y, svr_model.predict(X)))

    print(f'on {number_of_days_ahead} future values using recursive prediction')

    print(calculate_metrics(y_test, y_pred))


    if scaler:
        plt.figure(figsize=(12, 6))
        # plt.plot(data_test[DEFAULT_PRICE_COLUMN_NAME], color='blue', label= 'y')
        plt.plot(data_all[DEFAULT_PRICE_COLUMN_NAME], color='blue', label='dane historyczne')
        plt.plot(inverse_scaler_on_df_column(scaler, predict_future(data_train, number_of_lags, svr_model, number_of_days_ahead))[DEFAULT_PRICE_COLUMN_NAME], color='red', label='prognozy na zbiorze testowym')
        plt.plot(data_train.index[number_of_lags:], scaler.inverse_transform(svr_model.predict(X).reshape(-1, 1)), color='orange', label='prognozy na zbiorze uczącym i walidacyjnym')

        # plt.plot(dates_train, clf.predict(X_test), color= 'red', label= 'RBF model')
        # plt.plot(dates_test, scaler.inverse_transform(svr_rbf.predict(dates_test)), color= 'green', label= 'RBF model')
        plt.xlabel('Data')
        plt.ylabel('Cena')
        plt.title('Porównanie notowań historycznych oraz wartości prognozowanych modelem SVR')
        plt.legend()
        # plt.savefig('D://Studia//W8//Praca magisterska//30lag-30days-prediction.png')
        plt.show()
    else:
        plt.figure(figsize=(12, 6))
        # plt.plot(ethereum_test[DEFAULT_PRICE_COLUMN_NAME], color='blue', label= 'y')
        plt.plot(data_all[DEFAULT_PRICE_COLUMN_NAME], color='blue', label='dane historyczne')
        plt.plot(predict_future(data_train, number_of_lags, svr_model, number_of_days_ahead)[DEFAULT_PRICE_COLUMN_NAME],
                 color='red', label='prognozy na zbiorze testowym')
        plt.plot(data_train.index[number_of_lags:], svr_model.predict(X), color='orange', label='prognozy na zbiorze uczącym i walidacyjnym')

        # plt.plot(dates_train, clf.predict(X_test), color= 'red', label= 'RBF model')
        # plt.plot(dates_test, scaler.inverse_transform(svr_rbf.predict(dates_test)), color= 'green', label= 'RBF model')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        # plt.savefig('D://Studia//W8//Praca magisterska//30lag-30days-prediction.png')
        plt.show()

    '''
    plt.figure()
    close_px = data_all[DEFAULT_PRICE_COLUMN_NAME]
    mavg = close_px.rolling(window=30).mean()
    plt.figure(figsize=(10,6))
    close_px.plot(label='Bitcoin')
    mavg.plot(label='mavg')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    '''
    if scaler:
        plt.figure(figsize=(12, 6))
        # plt.plot(ethereum_test[DEFAULT_PRICE_COLUMN_NAME], color='blue', label= 'y')
        plt.plot(data_all[DEFAULT_PRICE_COLUMN_NAME], color='blue', label='historical data')
        #plt.plot(predict_future(data_train, number_of_lags, svr_model, number_of_days_ahead)[DEFAULT_PRICE_COLUMN_NAME],
        #            color='red', label='prognozy na zbiorze testowym')

        plt.plot(data_train.index[number_of_lags:], scaler.inverse_transform(svr_model.predict(X).reshape(-1, 1)), color='orange', label='predictions on training and validation sets')


        data_all_scaled = apply_scaler_df_column(scaler, data_all, standardized_column_name=DEFAULT_PRICE_COLUMN_NAME)
        data_all_with_lags = create_df_with_lags(data_all_scaled, number_of_lags, column=DEFAULT_PRICE_COLUMN_NAME)
        X_all, y_all = create_numpy_arrays_with_lags(data_all_with_lags, main_column=DEFAULT_PRICE_COLUMN_NAME)

        plt.plot(data_test.index, scaler.inverse_transform(svr_model.predict(X_all).reshape(-1, 1))[-data_test.size:], color='red', label='predictions on the test set')
        # plt.plot(dates_train, clf.predict(X_test), color= 'red', label= 'RBF model')
        # plt.plot(dates_test, scaler.inverse_transform(svr_rbf.predict(dates_test)), color= 'green', label= 'RBF model')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        # plt.savefig('D://Studia//W8//Praca magisterska//30lag-30days-prediction.png')
        plt.show()
        print('one day ahead predictions on future values **************')

        print(calculate_metrics(data_test[DEFAULT_PRICE_COLUMN_NAME].values, scaler.inverse_transform(svr_model.predict(X_all).reshape(-1, 1))[-data_test.size:].ravel()))
    else:
        plt.figure(figsize=(12, 6))
        # plt.plot(ethereum_test[DEFAULT_PRICE_COLUMN_NAME], color='blue', label= 'y')
        plt.plot(data_all[DEFAULT_PRICE_COLUMN_NAME], color='blue', label='historical data')
        # plt.plot(predict_future(data_train, number_of_lags, svr_model, number_of_days_ahead)[DEFAULT_PRICE_COLUMN_NAME],
        #            color='red', label='prognozy na zbiorze testowym')
        plt.plot(data_train.index[number_of_lags:], svr_model.predict(X), color='orange',
                 label='predictions on training and validation sets')
        #data_all_scaled = apply_scaler_df_column(scaler, data_all, standardized_column_name=DEFAULT_PRICE_COLUMN_NAME)
        data_all_with_lags = create_df_with_lags(data_all, number_of_lags, column=DEFAULT_PRICE_COLUMN_NAME)
        X_all, y_all = create_numpy_arrays_with_lags(data_all_with_lags, main_column=DEFAULT_PRICE_COLUMN_NAME)

        plt.plot(data_test.index, svr_model.predict(X_all)[-data_test.size:],
                 color='red', label='predictions on the test set')
        # plt.plot(dates_train, clf.predict(X_test), color= 'red', label= 'RBF model')
        # plt.plot(dates_test, scaler.inverse_transform(svr_rbf.predict(dates_test)), color= 'green', label= 'RBF model')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        # plt.savefig('D://Studia//W8//Praca magisterska//30lag-30days-prediction.png')
        plt.show()
        print('one day ahead predictions on future values **************')

        print(calculate_metrics(data_test[DEFAULT_PRICE_COLUMN_NAME].values, svr_model.predict(X_all)[-data_test.size:]))


perform_SVR_on_data(STELLAR_FILE, number_of_lags=90, use_scaler=1, C=10, gamma=1e-2, number_of_days_ahead_in_recursive_forecast=14)