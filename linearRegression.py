#%%
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklego.meta import ZeroInflatedRegressor

from sklearn import preprocessing

features = ['Volume', 'Dropsize', 'Dia', 'Mês', 'Ano', 'é_dia_normal', 'é_feriado',
       'é_fim_de_semana', 'Cluster_A', 'Cluster_B', 'Cluster_C', 'Cluster_D',
       'Cluster_E', 'Cluster_F', 'Cluster_J', 'Cluster_K', 'Cluster_L',
       'Cluster_M']

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

def gerarResultadoLinearRegression(X_treino, X_teste, y_treino, n = 1):

    aux_x_treino = X_treino.drop(features[n+1:], axis = 1)

    aux_x_teste = X_teste.drop(features[n+1:], axis = 1)

    X_treino_scaled, X_teste_scaled = escalarTreinoTeste(aux_x_treino, aux_x_teste)

    lm_model = LinearRegression(positive = True)

    print(f'Features utilizadas: {features[:n]}')

    lm_model.fit(X_treino_scaled, y_treino)

    res = lm_model.predict(X_teste_scaled)
    
    return res

def gerarResultadoZIR_LR(X_treino, X_teste, y_treino, n = 1):

    aux_x_treino = X_treino.drop(features[n+1:], axis = 1)

    aux_x_teste = X_teste.drop(features[n+1:], axis = 1)

    X_treino_scaled, X_teste_scaled = escalarTreinoTeste(aux_x_treino, aux_x_teste)

    zir_lr_model = ZeroInflatedRegressor(classifier = RandomForestClassifier(), regressor = LinearRegression(positive = True))

    print(f'Features utilizadas: {features[:n]}')

    zir_lr_model.fit(X_treino_scaled, y_treino)

    # e a predição
    res = zir_lr_model.predict(X_teste_scaled)
    
    return res
# %%
