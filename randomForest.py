
#%%
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklego.meta import ZeroInflatedRegressor

import numpy as np

#%% essa função serve para gerar os melhores parametros para o regressor Random Forest

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
    
    print('Best set of hyperparameters obtained: ' + str(rf_grid.best_params_)) 

    return(rf_grid.best_params_)

# %% aqui é criado o modelo, treinado e retorna o resultado predito
def gerarResultadoRandomForest(x_treino, x_teste, y_treino, optimize = False):

    #%% caso queira a otimização de parâmetros
    if(optimize):

        best_hyperparams = findBestParams(x_treino, y_treino)

        model = RandomForestRegressor(n_estimators = best_hyperparams['n_estimators'],
                                    max_depth = best_hyperparams['max_depth'],
                                    min_samples_leaf = best_hyperparams['min_samples_leaf'],
                                    min_samples_split = best_hyperparams['min_samples_split'],
                                    max_leaf_nodes = best_hyperparams['max_leaf_nodes'],
                                    max_features = best_hyperparams['max_features']
                                    )

    #%% senão, serão default
    else:
        
        model = RandomForestRegressor()

    # %% o modelo é treinado
    model.fit(x_treino, y_treino)

    # %% e aqui, a predição
    res = model.predict(x_teste)
    
    return(res)

# %% essa função utiliza a versão Zero-Inflated do Random Forest   
def gerarResultadoZIRRandomForest(x_treino, x_teste, y_treino, optimize = False):
    
    if(optimize):

        best_hyperparams = findBestParams(x_treino, y_treino)

        # %% o modelo utiliza um classificador para checar se um valor é ou não 0
        zir = ZeroInflatedRegressor(
            
            classifier = RandomForestClassifier(n_estimators = best_hyperparams['n_estimators'],
                                    max_depth = best_hyperparams['max_depth'],
                                    min_samples_leaf = best_hyperparams['min_samples_leaf'],
                                    min_samples_split = best_hyperparams['min_samples_split'],
                                    max_leaf_nodes = best_hyperparams['max_leaf_nodes'],
                                    max_features = best_hyperparams['max_features']
                                    ),
        
            # %% e o regressor para prever os valores que não são 0
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

    # %% o treinamento do modelo
    zir.fit(x_treino, y_treino)

    # e a predição
    res = zir.predict(x_teste)

    return(res)
