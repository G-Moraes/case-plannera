#%%
from sklearn.linear_model import LinearRegression, Ridge

import numpy as np

def gerarResultadoLinearRegression(x_treino, x_teste, y_treino):

    lm_model = LinearRegression()

    lm_model.fit(x_treino[['Volume', 'Dropsize']], y_treino)

    res = lm_model.predict(x_teste[['Volume', 'Dropsize']])

    slope = lm_model.coef_
    intercept = lm_model.intercept_

    line = (np.dot(x_teste[['Volume', 'Dropsize']], slope) + intercept)

    return (res, line)

def gerarResultadoRidgeRegression(x_treino, x_teste, y_treino):

    rid_model = Ridge()

    rid_model.fit(x_treino[['Volume', 'Dropsize']], y_treino)

    res = rid_model.predict(x_teste[['Volume', 'Dropsize']])

    slope = rid_model.coef_

    intercept = rid_model.intercept_
    
    line = (np.dot(x_teste[['Volume', 'Dropsize']], slope) + intercept)

    return (res, line)
# %%
