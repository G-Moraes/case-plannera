#%%
from sklearn.linear_model import LinearRegression, Ridge

import numpy as np

def gerarResultadoLinearRegression(x_treino, x_teste, y_treino, n = 1):

    lm_model = LinearRegression()

    features = ['Volume', 'Dropsize', 'Dia', 'Mês', 'Ano', 'é_dia_normal', 'é_feriado',
       'é_fim_de_semana', 'Cluster_A', 'Cluster_B', 'Cluster_C', 'Cluster_D',
       'Cluster_E', 'Cluster_F', 'Cluster_J', 'Cluster_K', 'Cluster_L',
       'Cluster_M']

    print('Features utilizadas: {}'.format(features[:n]))

    lm_model.fit(x_treino[features[0:n]], y_treino)

    res = lm_model.predict(x_teste[features[:n]])
    
    return res

def gerarResultadoRidgeRegression(x_treino, x_teste, y_treino, n = 1):

    rid_model = Ridge()

    features = ['Volume', 'Dropsize', 'Dia', 'Mês', 'Ano', 'é_dia_normal', 'é_feriado',
       'é_fim_de_semana', 'Cluster_A', 'Cluster_B', 'Cluster_C', 'Cluster_D',
       'Cluster_E', 'Cluster_F', 'Cluster_J', 'Cluster_K', 'Cluster_L',
       'Cluster_M']
    
    print('Features utilizadas: {}'.format(features[:n]))

    rid_model.fit(x_treino[features[:n]], y_treino)

    res = rid_model.predict(x_teste[features[:n]])

    return res
# %%
