from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor


def fit_best_MLP_model_using_GridSearch(X, y):
    param_grid = {"hidden_layer_sizes": [(10,), (50,), (10, 10), (50, 50), (10, 10, 10), (50, 50, 50)],
                  "max_iter": [250, 500, 1000, 2000]}

    mlp_model = GridSearchCV(MLPRegressor(), param_grid, cv=5, refit=True, verbose=0)

    # fitting the model from grid search
    mlp_model.fit(X, y)
    # print(f'Chosen parameters: {mlp_model.best_params_}')
    return mlp_model

"""
def stardardize_data(vector):
    scaler = StandardScaler().fit(vector)
    vector_scaled = scaler.transform(vector)
    return vector_scaled, scaler
"""



    


