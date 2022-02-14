#%%
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

from sklearn import preprocessing

from sklearn.linear_model import PoissonRegressor

from sklearn.ensemble import RandomForestClassifier

from sklego.meta import ZeroInflatedRegressor

#%% alguns modelos exigem uma normalização no dataset para ser executado
#e o poisson é um deles

def escalarTreinoTeste(X_treino, X_teste):

    # %% as variaveis de X são normalizadas pela fórmula (x - u) / s, onde
    # x = o próprio vetor
    # u = a média 
    # s = o desvio padrão

    X_treino_scaler = preprocessing.StandardScaler().fit(X_treino)

    X_treino_scaled = X_treino_scaler.transform(X_treino)

    X_teste_scaler = preprocessing.StandardScaler().fit(X_teste)

    X_teste_scaled = X_teste_scaler.transform(X_teste)

    return (X_treino_scaled, X_teste_scaled)

# %% a implementação da Distribuição de Poisson
def gerarResultadoPoisson(X_treino, X_teste, y_treino):

    # %% normalização do dataset
    X_treino_scaled, X_teste_scaled = escalarTreinoTeste(X_treino, X_teste)

    # %% criação do modelo
    poissonModel = PoissonRegressor()
    
    # %% treinamento
    poissonModel.fit(X_treino_scaled, y_treino)

    # %% e a predição
    res = poissonModel.predict(X_teste_scaled)

    return (res)

#%% implementação de Poisson com Zero-Inflated, utilizando Random Forest
# como classificador
def gerarResultadoZIRPoisson(X_treino, X_teste, y_treino):

    # %% normalização do dataset
    X_treino_scaled, X_teste_scaled = escalarTreinoTeste(X_treino, X_teste)

    # %% criação do modelo
    zir = ZeroInflatedRegressor(classifier = RandomForestClassifier(),
                                regressor = PoissonRegressor())

    # %% treinamento
    zir.fit(X_treino_scaled, y_treino)

    # %% e a predição
    res = zir.predict(X_teste_scaled)

    return(res)
