import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from data.data_preparation import *
from models.SVR.SVR_model import predict_future

number_of_lags = 150
number_of_days_ahead = 150


bitcoin_all = import_data(os.path.join(os.curdir, RAW_DIR, BITCOIN_FILE))
bitcoin_all = remove_missing_values(bitcoin_all)
bitcoin_all = set_date_column_as_index(bitcoin_all)
bitcoin_all = choose_and_rename_column(bitcoin_all, 'Close', DEFAULT_PRICE_COLUMN_NAME)
bitcoin_train, bitcoin_test = split_into_train_and_test_by_ratio(bitcoin_all, 0.9)

bitcoin_train_with_lags = create_df_with_lags(bitcoin_train, number_of_lags)
X, y = create_numpy_arrays_with_lags(bitcoin_train_with_lags)

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=1)
#svr_model = fit_best_SVR_model_using_GridSearch(X_train, y_train)
svr_model = SVR(kernel='rbf', C=1000, gamma=1e-09)
svr_model.fit(X_train, y_train)


"""
plt.figure(figsize=(12,6))
plt.plot(y_validate, color='blue', label='y')
plt.plot(svr_model.predict(X_validate), color= 'black', label= 'predicted')

#plt.plot(dates_train, clf.predict(X_test), color= 'red', label= 'RBF model')
#plt.plot(dates_test, scaler.inverse_transform(svr_rbf.predict(dates_test)), color= 'green', label= 'RBF model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
"""

plt.figure(figsize=(12, 6))
# plt.plot(bitcoin_test[DEFAULT_PRICE_COLUMN_NAME], color='blue', label= 'y')
plt.plot(bitcoin_all[DEFAULT_PRICE_COLUMN_NAME], color='blue', label='dane historyczne')
plt.plot(predict_future(bitcoin_train, number_of_lags, svr_model, number_of_days_ahead)[DEFAULT_PRICE_COLUMN_NAME], color='red', label='predicted_test')
plt.plot(bitcoin_train.index[number_of_lags:], svr_model.predict(X), color='orange', label='predicted_train')

# plt.plot(dates_train, clf.predict(X_test), color= 'red', label= 'RBF model')
# plt.plot(dates_test, scaler.inverse_transform(svr_rbf.predict(dates_test)), color= 'green', label= 'RBF model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
# plt.savefig('D://Studia//W8//Praca magisterska//30lag-30days-prediction.png')
plt.show()
