#%%
from pickletools import string1
from sklearn.metrics import mean_squared_error, mean_absolute_error

#%% gera as métricas para um mês, sem distinguir cada cluster
def gerarMetricaGeral(y_teste, res, save = False, nome = None):

    strList = []

    MSE = mean_squared_error(y_teste, res)

    RMSE = mean_squared_error(y_teste, res, squared = False)

    MAE = mean_absolute_error(y_teste, res)

    strList.append(f'Mean Squared Error: {MSE}')
    strList.append(f'Root Mean Squared Error: {RMSE}')
    strList.append(f'Mean Absolute Error: {MAE}')

    print(strList[0])
    print(strList[1])
    print(strList[2] + '\n')

    # %% caso queira, pode salvar as medidas como um .txt    
    if(save and nome):

        f = open(nome + '.txt', 'w+')

        f.write('\n'.join(strList))
        
        f.close()

# gera as métricas individuais de cada cluster
def gerarMetricasClusters(X_teste, y_teste, res, save = False, nome = None):

    clusters = ['A', 'B', 'C', 'D', 'E', 'F', 'J', 'K', 'L', 'M']

    indexes = X_teste.index

    metricasSave = {}
    strList = []

    for i in clusters:
        
        clusterIndexes = indexes[X_teste['Cluster_' + i] == 1].tolist()

        MSE = mean_squared_error(y_teste[clusterIndexes], res[clusterIndexes])

        RMSE = mean_squared_error(y_teste[clusterIndexes], res[clusterIndexes], squared = False)

        MAE = mean_absolute_error(y_teste[clusterIndexes], res[clusterIndexes])

        strList.append(f'No Cluster {i}')
        strList.append(f'Mean Squared Error: {MSE}')
        strList.append(f'Root Mean Squared Error: {RMSE}')
        strList.append(f'Mean Absolute Error: {MAE}\n')

        print(strList[0])
        print(strList[1])
        print(strList[2])
        print(strList[3])

        if(save):

            metricasSave[i] = '\n'.join(strList)

        strList = []
            
    # %% caso queira, pode salvar as medidas como um .txt 
    if(save and nome):

        f = open(nome + '.txt', 'w+')

        for i in metricasSave:

            f.write(metricasSave[i] + '\n')
            
        f.close()
# %%
