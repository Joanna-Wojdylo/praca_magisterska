import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from data.data_preparation import *
from models.metrics_utils import calculate_metrics

from consts import BITCOIN_FILE, ETHEREUM_FILE, LITECOIN_FILE, RIPPLE_FILE, TETHER_FILE, CHAINLINK_FILE, NEM_FILE, \
    STELLAR_FILE


def predict_future_LSTM(known_dataframe: pd.DataFrame, lag: int, model, days_ahead: int):
    """
    known_dataframe in form of date(index) - value(column), column named as default 'prices'
    """
    X = create_numpy_arrays_with_lags(create_df_with_lags(known_dataframe, lag))[0]
    predicted_y = []
    indexes = []
    last_index = known_dataframe.index[-1]
    for i in range(0, days_ahead):
        X_reshaped = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        new_prediction = model.predict(X_reshaped[-1:])[0, 0, 0]  # jeden dzien do przodu
        predicted_y.append(new_prediction)
        X = np.concatenate((X, np.append(new_prediction, X[-1][:-1]).reshape(1, -1)))
        last_index = last_index + pd.Timedelta(days=1)
        indexes.append(last_index)

    predicted_dataframe = pd.DataFrame({DEFAULT_PRICE_COLUMN_NAME: predicted_y, 'date': indexes})
    predicted_dataframe = predicted_dataframe.set_index('date')
    predicted_dataframe[DEFAULT_PRICE_COLUMN_NAME] = predicted_dataframe[DEFAULT_PRICE_COLUMN_NAME].astype(float)
    return predicted_dataframe


def perform_LSTM_on_data(data_file: str, number_of_lags: int, use_scaler: int,
                         number_of_days_ahead_in_recursive_forecast: int = 14):
    number_of_days_ahead = number_of_days_ahead_in_recursive_forecast

    data_all = import_data(os.path.join(os.curdir, RAW_DIR, data_file))
    # print('number of days: ', data_all.shape)
    data_all = interpolate_missing_values(data_all)
    data_all = set_date_column_as_index(data_all)
    data_all = choose_and_rename_column(data_all, 'Close', DEFAULT_PRICE_COLUMN_NAME)
    data_train, data_test = split_into_train_and_test_by_ratio(data_all, 0.9)
    # print('number of train days: ', data_train.shape)
    # print('number of test days: ', data_test.shape)
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
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    model = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True, input_shape=(1, number_of_lags)))
    # model.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))

    # # Adding a third LSTM layer and some Dropout regularisation
    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))
    #
    # # Adding a fourth LSTM layer and some Dropout regularisation
    # model.add(LSTM(units=50))
    # model.add(Dropout(0.2))

    # Adding the output layer
    model.add(Dense(units=1))

    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    model.summary()

    if scaler:
        y_pred = inverse_scaler_on_df_column(scaler, predict_future_LSTM(data_train, number_of_lags, model,
                                                                         number_of_days_ahead))[
            DEFAULT_PRICE_COLUMN_NAME].values
    else:
        y_pred = predict_future_LSTM(data_train, number_of_lags, model, number_of_days_ahead)[
            DEFAULT_PRICE_COLUMN_NAME].values
    y_test = data_test[DEFAULT_PRICE_COLUMN_NAME].values[:number_of_days_ahead]

    # print('on train values **************')
    # print( calculate_metrics(y, model.predict(X)))

    print(f'on {number_of_days_ahead} future values using recursive prediction')

    print(calculate_metrics(y_test, y_pred))

    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    if scaler:
        plt.figure(figsize=(12, 6))
        plt.plot(data_all[DEFAULT_PRICE_COLUMN_NAME], color='blue', label='historical data')
        plt.plot(inverse_scaler_on_df_column(scaler, predict_future_LSTM(data_train, number_of_lags, model,
                                                                         number_of_days_ahead))[
                     DEFAULT_PRICE_COLUMN_NAME],
                 color='red', label='predictions on the test set')
        plt.plot(data_train.index[number_of_lags:],
                 scaler.inverse_transform(model.predict(X).reshape(X.shape[0], ).reshape(-1, 1)),
                 color='orange', label='predictions on training and validation sets')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(data_all[DEFAULT_PRICE_COLUMN_NAME], color='blue', label='historical data')
        plt.plot(
            predict_future_LSTM(data_train, number_of_lags, model, number_of_days_ahead)[DEFAULT_PRICE_COLUMN_NAME],
            color='red', label='predictions on the test set')
        plt.plot(data_train.index[number_of_lags:], model.predict(X).reshape(X.shape[0], ), color='orange',
                 label='predictions on training and validation sets')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    if scaler:
        plt.figure(figsize=(12, 6))
        plt.plot(data_all[DEFAULT_PRICE_COLUMN_NAME], color='blue', label='historical data')
        plt.plot(data_train.index[number_of_lags:],
                 scaler.inverse_transform(model.predict(X).reshape(X.shape[0], ).reshape(-1, 1)),
                 color='orange', label='predictions on training and validation sets')

        data_all_scaled = apply_scaler_df_column(scaler, data_all, standardized_column_name=DEFAULT_PRICE_COLUMN_NAME)
        data_all_with_lags = create_df_with_lags(data_all_scaled, number_of_lags, column=DEFAULT_PRICE_COLUMN_NAME)
        X_all, y_all = create_numpy_arrays_with_lags(data_all_with_lags, main_column=DEFAULT_PRICE_COLUMN_NAME)
        X_all = np.reshape(X_all, (X_all.shape[0], 1, X_all.shape[1]))

        plt.plot(data_test.index,
                 scaler.inverse_transform(model.predict(X_all).reshape(X_all.shape[0], ).reshape(-1, 1))[
                 -data_test.size:],
                 color='red', label='predictions on the test set')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        print('one day ahead predictions on future values **************')

        print(calculate_metrics(data_test[DEFAULT_PRICE_COLUMN_NAME].values,
                                scaler.inverse_transform(model.predict(X_all).reshape(X_all.shape[0], ).reshape(-1, 1))[
                                -data_test.size:].ravel()))
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(data_all[DEFAULT_PRICE_COLUMN_NAME], color='blue', label='historical data')

        plt.plot(data_train.index[number_of_lags:], model.predict(X).reshape(X.shape[0], ), color='orange',
                 label='predictions on training and validation sets')
        data_all_with_lags = create_df_with_lags(data_all, number_of_lags, column=DEFAULT_PRICE_COLUMN_NAME)
        X_all, y_all = create_numpy_arrays_with_lags(data_all_with_lags, main_column=DEFAULT_PRICE_COLUMN_NAME)
        X_all = np.reshape(X_all, (X_all.shape[0], 1, X_all.shape[1]))

        plt.plot(data_test.index, model.predict(X_all).reshape(X_all.shape[0], )[-data_test.size:],
                 color='red', label='predictions on the test set')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        print('one day ahead predictions on future values **************')

        print(
            calculate_metrics(data_test[DEFAULT_PRICE_COLUMN_NAME].values,
                              model.predict(X_all).reshape(X_all.shape[0], )[-data_test.size:]))


if __name__ == "__main__":
    perform_LSTM_on_data(STELLAR_FILE, number_of_lags=90, use_scaler=1, number_of_days_ahead_in_recursive_forecast=147)
