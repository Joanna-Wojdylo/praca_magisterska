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
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_validate = np.reshape(X_validate, (X_validate.shape[0], 1, X_validate.shape[1]))
    model = Sequential()
    # Adding the first LSTM layer
    # Two LSTM layers with 50 units, output layer of 1 neuron
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

    if use_scaler == 0:
        train_metrics = calculate_metrics(y_train, model.predict(X_train).reshape(X_train.shape[0], ))
        validate_metrics = calculate_metrics(y_validate, model.predict(X_validate).reshape(X_validate.shape[0], ))
    else:
        train_metrics = calculate_metrics(scaler.inverse_transform(y_train.reshape(-1, 1)), scaler.inverse_transform(model.predict(X_train).reshape(X_train.shape[0], ).reshape(-1, 1)))
        validate_metrics = calculate_metrics(scaler.inverse_transform(y_validate.reshape(-1, 1)), scaler.inverse_transform(model.predict(X_validate).reshape(X_validate.shape[0], ).reshape(-1, 1)))

    return train_metrics, validate_metrics


def compare_models():
    columns_naming = ['data', 'lags used', 'transformation', 'RMSE', 'MAE', 'MAPE', 'R2']
    train_table = []
    validate_table = []
    test_table = []
    for data_file in [TETHER_FILE]:
        for number_of_lags in [7, 14, 21, 30, 90, 150]:
            print(f'******************* DOING {number_of_lags} LAGS *********************')
            for use_scaler in [0, 1, 2]:
                if use_scaler == 0:
                    scaler_text = 'Without transformation'
                elif use_scaler == 1:
                    scaler_text = 'Standardized'
                else:
                    scaler_text = 'Normalized into (-1,1)'
                train_metrics, validate_metrics = explore_model(data_file, number_of_lags, use_scaler)
                train_table.append([data_file, number_of_lags, scaler_text] + train_metrics)
                validate_table.append([data_file, number_of_lags, scaler_text] + validate_metrics)
    train_df = pd.DataFrame.from_records(train_table)
    train_df.columns = columns_naming
    train_df.to_csv('data/processed/train_df_scores.csv', index=False)
    validate_df = pd.DataFrame.from_records(validate_table)
    validate_df.columns = columns_naming
    validate_df.to_csv('data/processed/validate_df_scores.csv', index=False)


compare_models()
#explore_model(BITCOIN_FILE, 10, use_scaler=2)