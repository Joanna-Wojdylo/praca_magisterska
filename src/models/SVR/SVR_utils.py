from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


def fit_best_SVR_model_using_GridSearch(X, y):
    param_grid = {'C': [0.1, 1, 10, 50, 100, 500, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 1e-08, 1e-09, 1e-10],
                  'kernel': ['rbf']}

    svr_rbf = GridSearchCV(SVR(), param_grid, cv=5, refit=True, verbose=0)

    # fitting the model from grid search
    svr_rbf.fit(X, y)
    # print(f'Chosen parameters: {svr_rbf.best_params_}')
    return svr_rbf

"""
def stardardize_data(vector):
    scaler = StandardScaler().fit(vector)
    vector_scaled = scaler.transform(vector)
    return vector_scaled, scaler
"""



    


