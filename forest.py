
#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn import metrics

from case import gerarDados 

def findBestParams(x, y):

    stringAcc = []

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

    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.33)
    
    rf_grid = GridSearchCV(estimator = model, param_grid = param_grid, refit = False, verbose = 0, n_jobs = -1)
    
    rf_grid.fit(x_treino, y_treino)

    str1 = 'Best set of hyperparameters obtained: ' + str(rf_grid.best_params_) + '\n'
    str2 = 'Accuracy obtained: ' + "{:.2f}".format(rf_grid.best_score_) + '\n\n'
    
    print(str1) 
    print(str2)

    stringAcc.append(str1)
    stringAcc.append(str2)

    return(rf_grid.best_params_, stringAcc)

#%%

x_treino = x_teste = y_treino = []

x_treino, x_teste, y_treino = gerarDados(returnValue = True)

#%%

x, y = gerarDados(returnValue = False)

#%%
#best_hyperparams, accuracy = findBestParams(x, y)

#%%
'''model = RandomForestRegressor(n_estimators = best_hyperparams['n_estimators'],
                                   max_depth = best_hyperparams['max_depth'],
                                   min_samples_leaf = best_hyperparams['min_samples_leaf'],
                                   min_samples_split = best_hyperparams['min_samples_split'],
                                   max_leaf_nodes = best_hyperparams['max_leaf_nodes'],
                                   max_features = best_hyperparams['max_features']
                                   )
'''
model = RandomForestRegressor()

#%%
model.fit(x_treino, y_treino)

#%%
resultado = model.predict(x_teste)

# %%

# %%

from sklego.meta import ZeroInflatedRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

zir = ZeroInflatedRegressor(
    classifier=RandomForestClassifier(),

    regressor=RandomForestRegressor()
)

# Easy fit!
zir.fit(x_treino, y_treino)

res = zir.predict(x_teste)

# %%
import pandas as pd

import datetime

#period = pd.date_range('2020-09-01', periods = 30, freq = 'D')

clusters = []

period = []

date = datetime.datetime(2020, 8, 31).date()

for i in range(300):
    
    if(i % 10 == 0):
        clusters += ['A', 'B', 'C', 'D', 'E', 'F', 'J', 'K', 'L', 'M']
        date+= datetime.timedelta(days=1)
    
    period.append(date)

df = pd.DataFrame()

df['DATA'] = period

df['CLUSTER'] = clusters

df['Volume'] = x_teste['Volume']

df['Dropsize'] = x_teste['Dropsize']

df['Remessas'] = res

df.to_excel('resultado.xlsx')

# %%
