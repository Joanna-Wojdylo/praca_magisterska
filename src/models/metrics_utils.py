import numpy as np
from sklearn import metrics

import pandas as pd

from data.data_preparation import create_df_with_lags, create_numpy_arrays_with_lags
from consts import DEFAULT_PRICE_COLUMN_NAME


def predict_future(known_dataframe: pd.DataFrame, lag: int,  model, days_ahead: int):
    """
    known_dataframe in form of date(index) - value(column), column named as default 'prices'
    """
    X = create_numpy_arrays_with_lags(create_df_with_lags(known_dataframe, lag))[0]
    predicted_y = []
    indexes = []
    last_index = known_dataframe.index[-1]
    for i in range(0, days_ahead):
        new_prediction = model.predict(X[-1].reshape(1, -1))  # jeden dzien do przodu
        predicted_y.append(new_prediction)
        X = np.concatenate((X, np.append(new_prediction, X[-1][:-1]).reshape(1, -1)))
        last_index = last_index + pd.Timedelta(days=1)
        indexes.append(last_index)

    predicted_dataframe = pd.DataFrame({DEFAULT_PRICE_COLUMN_NAME: predicted_y, 'date': indexes})
    predicted_dataframe = predicted_dataframe.set_index('date')
    predicted_dataframe[DEFAULT_PRICE_COLUMN_NAME] = predicted_dataframe[DEFAULT_PRICE_COLUMN_NAME].astype(float)
    return predicted_dataframe


def calculate_mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


def calculate_metrics(actual, pred):
    mse = metrics.mean_squared_error(actual, pred)
    # print("MSE:", mse)
    rmse = np.sqrt(metrics.mean_squared_error(actual, pred))
    # print("RMSE:", rmse)
    mae = metrics.mean_absolute_error(actual, pred)
    # print("MAE:", mae)
    explained_variance_score = metrics.explained_variance_score(actual, pred)
    # print("Explained variance score: ", explained_variance_score)
    mape = calculate_mape(actual, pred)
    # print("MAPE: ", mape)
    r2 = metrics.r2_score(actual, pred)
    # print("R2:", r2)

    correlation_matrix = np.corrcoef(actual, pred)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2
    # print("R2 numpy:", r_squared)

    # sum of square of residuals
    # ssr = np.sum((pred - actual) ** 2)
    #  total sum of squares
    # sst = np.sum((actual - np.mean(actual)) ** 2)
    # R2 score
    # r2_score = 1 - (ssr / sst)
    # print("R2 =1-ssr/sst:", r2_score)

    residuals = actual - pred
    residuals_mean = np.mean(residuals)
    residuals_std = np.std(residuals)
    # print('Residuals mean: ', residuals_mean)
    # print('Residuals sd: ', residuals_std)
    return [rmse, mae, mape, r2]
