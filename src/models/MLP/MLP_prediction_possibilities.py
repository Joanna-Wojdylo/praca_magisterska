import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from data.data_preparation import *
from models.metrics_utils import calculate_metrics, predict_future
from consts import BITCOIN_FILE, ETHEREUM_FILE, LITECOIN_FILE, RIPPLE_FILE, TETHER_FILE, STELLAR_FILE


def prediction_possibilities_rolling_window(number_of_lags: int, file: str, use_scaler: int, hidden_layer_sizes:tuple, max_iter:int ):
    data_file = file
    data_all = import_data(os.path.join(os.curdir, RAW_DIR, data_file))
    # print('number of days: ', data_all.shape)
    data_all = interpolate_missing_values(data_all)
    data_all = set_date_column_as_index(data_all)
    data_all = choose_and_rename_column(data_all, 'Close', DEFAULT_PRICE_COLUMN_NAME)
    data_train, data_test = split_into_train_and_test_by_ratio(data_all, 0.9)
    if use_scaler == 0:
        scaler = None
    elif use_scaler == 1:
        scaler = fit_standard_scaler_to_df_column(data_train)
        data_train = apply_scaler_df_column(scaler, data_train, standardized_column_name=DEFAULT_PRICE_COLUMN_NAME)
    elif use_scaler == 2:
        scaler = fit_minmax_scaler_to_df_column(data_train)
        data_train = apply_scaler_df_column(scaler, data_train, DEFAULT_PRICE_COLUMN_NAME, DEFAULT_PRICE_COLUMN_NAME)
    else:
        print('Invalid use scaler parameter. Valid options: 0 (None), 1 (Standard scaler), 2 (Minmax scaler)')
        return None

    data_train_with_lags = create_df_with_lags(data_train, number_of_lags, column=DEFAULT_PRICE_COLUMN_NAME)
    X, y = create_numpy_arrays_with_lags(data_train_with_lags, main_column=DEFAULT_PRICE_COLUMN_NAME)

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=1)

    mlp_model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    mlp_model.fit(X_train, y_train)

    if use_scaler == 0:
        data_all_with_lags = create_df_with_lags(data_all, number_of_lags, column=DEFAULT_PRICE_COLUMN_NAME)
        X_all, y_all = create_numpy_arrays_with_lags(data_all_with_lags, main_column=DEFAULT_PRICE_COLUMN_NAME)
        metrics_list = []
        metrics_list.append(calculate_metrics(data_test[DEFAULT_PRICE_COLUMN_NAME].values, mlp_model.predict(X_all)[-data_test.size:]))
        for time_horizon in range(2, 101):
            horizon_scores = []
            for iteration in range(data_test.size - time_horizon + 1):
                temporary_known_data = data_all.iloc[:data_train.size + iteration]
                y_pred = predict_future(temporary_known_data, number_of_lags, mlp_model, time_horizon)[DEFAULT_PRICE_COLUMN_NAME].values
                y_test = data_all.iloc[temporary_known_data.size:temporary_known_data.size+time_horizon][DEFAULT_PRICE_COLUMN_NAME].values
                metrics = calculate_metrics(y_test, y_pred)
                horizon_scores.append(metrics)
            horizon_scores = pd.DataFrame.from_records(horizon_scores)
            metrics_list.append(list(horizon_scores.mean().values))
    else:
        data_all_scaled = apply_scaler_df_column(scaler, data_all, standardized_column_name=DEFAULT_PRICE_COLUMN_NAME)
        data_all_with_lags = create_df_with_lags(data_all_scaled, number_of_lags, column=DEFAULT_PRICE_COLUMN_NAME)
        X_all, y_all = create_numpy_arrays_with_lags(data_all_with_lags, main_column=DEFAULT_PRICE_COLUMN_NAME)
        metrics_list = []
        metrics_list.append(calculate_metrics(data_test[DEFAULT_PRICE_COLUMN_NAME].values, scaler.inverse_transform(mlp_model.predict(X_all).reshape(-1, 1))[-data_test.size:].ravel()))
        for time_horizon in range(2, 101):
            horizon_scores = []
            for iteration in range(data_test.size - time_horizon + 1):
                temporary_known_data = data_all_scaled.iloc[:data_train.size + iteration]
                y_pred = inverse_scaler_on_df_column(scaler, predict_future(temporary_known_data, number_of_lags, mlp_model, time_horizon))[
                    DEFAULT_PRICE_COLUMN_NAME].values
                # y_test = data_test[DEFAULT_PRICE_COLUMN_NAME].values[:time_horizon]
                y_test = data_all.iloc[temporary_known_data.size:temporary_known_data.size + time_horizon][
                    DEFAULT_PRICE_COLUMN_NAME].values
                metrics = calculate_metrics(y_test, y_pred)
                horizon_scores.append(metrics)
            horizon_scores = pd.DataFrame.from_records(horizon_scores)
            metrics_list.append(list(horizon_scores.mean().values))

    metrics_df = pd.DataFrame.from_records(metrics_list)
    metrics_df.columns = ['RMSE', 'MAE', 'MAPE', 'R2']
    metrics_df.index += 1  # shifting index to start with 1 day prediction, not 0
    metrics_df.plot(y=['RMSE', 'MAE'])
    # plt.title('Przebieg wielkości błędu RMSE w zalezności od liczby prognozowanych dni')
    plt.xlabel('Forecast horizon')
    plt.ylabel('Prediction errors')
    plt.grid()
    plt.legend()
    plt.show()
    return metrics_df


metrics_df = prediction_possibilities_rolling_window(90, STELLAR_FILE, 1, hidden_layer_sizes=(50,50,50), max_iter=250)
print(metrics_df.head(50))