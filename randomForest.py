
#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklego.meta import ZeroInflatedRegressor

import numpy as np

def findBestParams(x, y):

    print('Generating best set of hyperparameters for the model. . .')
    n_estimators = [int(x) for x in np.linspace(start = 5, stop = 100, num = 10)]

    max_depth = [2, 4]

    min_samples_split = [2, 4, 6]

    min_samples_leaf = [1, 2, 4]

    max_leaf_nodes = [2, 5, 10]

    max_features = ['auto', 'sqrt']

    param_grid = {'n_estimators': n_estimators,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'max_leaf_nodes': max_leaf_nodes,
                  'max_features': max_features
                  }

    model = RandomForestRegressor()

    rf_grid = GridSearchCV(estimator = model, param_grid = param_grid, refit = False, verbose = 0, n_jobs = -1)
    
    rf_grid.fit(x, y)
    
    print('Best set of hyperparameters obtained: ' + str(rf_grid.best_params_) + '\n' + 'Accuracy obtained: ' + "{:.2f}".format(rf_grid.best_score_) + '\n\n') 

    return(rf_grid.best_params_)

def gerarResultadoRandomForest(x_treino, x_teste, y_treino, optimize = False):

    if(optimize):

        best_hyperparams = findBestParams(x_treino, y_treino)

        model = RandomForestRegressor(n_estimators = best_hyperparams['n_estimators'],
                                    max_depth = best_hyperparams['max_depth'],
                                    min_samples_leaf = best_hyperparams['min_samples_leaf'],
                                    min_samples_split = best_hyperparams['min_samples_split'],
                                    max_leaf_nodes = best_hyperparams['max_leaf_nodes'],
                                    max_features = best_hyperparams['max_features']
                                    )

    else:
        
        model = RandomForestRegressor()

    model.fit(x_treino, y_treino)

    res = model.predict(x_teste)
    
    return(res)
    
def gerarResultadoZIRRandomForest(x_treino, x_teste, y_treino, optimize = False):
    
    if(optimize):

        best_hyperparams = findBestParams(x_treino, y_treino)

        zir = ZeroInflatedRegressor(
            
            classifier = RandomForestClassifier(n_estimators = best_hyperparams['n_estimators'],
                                    max_depth = best_hyperparams['max_depth'],
                                    min_samples_leaf = best_hyperparams['min_samples_leaf'],
                                    min_samples_split = best_hyperparams['min_samples_split'],
                                    max_leaf_nodes = best_hyperparams['max_leaf_nodes'],
                                    max_features = best_hyperparams['max_features']
                                    ),

            regressor = RandomForestRegressor(n_estimators = best_hyperparams['n_estimators'],
                                    max_depth = best_hyperparams['max_depth'],
                                    min_samples_leaf = best_hyperparams['min_samples_leaf'],
                                    min_samples_split = best_hyperparams['min_samples_split'],
                                    max_leaf_nodes = best_hyperparams['max_leaf_nodes'],
                                    max_features = best_hyperparams['max_features']
                                    )
        )

    else:

        zir = ZeroInflatedRegressor(classifier = RandomForestClassifier(), regressor = RandomForestRegressor())

    zir.fit(x_treino, y_treino)

    res = zir.predict(x_teste)

    return(res)
