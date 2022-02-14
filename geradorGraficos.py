#%%
from multiprocessing.sharedctypes import Value
import seaborn as sns
import matplotlib.pyplot as plt

from linearRegression import gerarResultadoLinearRegression, gerarResultadoRidgeRegression
from randomForest import gerarResultadoRandomForest, gerarResultadoZIRRandomForest

from geradorExcel import gerarExcel
from tratamentoDados import gerarDados

plt.figure(num=1, figsize=(8, 6))

sns.set(rc={'figure.figsize':(11.7,8.27)})

from random import randint

def gerarGraficoMensal(x_treino, x_teste, y_treino, y_teste, resultado, comecoMes, fimMes, saveImage = False, name = None):

    plt.clf()

    sns.lineplot(x = x_teste['Dia'], y = resultado, label = 'Remessa prevista')

    sns.lineplot(label = 'Remessa atual', x = x_treino['Dia'][comecoMes:fimMes], y = y_teste).set_title('Mês {}/{}'.format(x_teste['Mês'][comecoMes], x_teste['Ano'][comecoMes]))

    if(saveImage and name):
        plt.savefig(name + '.png')

def gerarGraficoLinear(x_teste, y_teste, resultado, saveImage = False, name = None):

    plt.clf()

    plt.scatter(x_teste['Volume'], y_teste, s = 3)
    plt.plot(x_teste['Volume'], resultado, color = 'r', linewidth = 3)


    plt.xlabel("Volume")
    plt.ylabel("Remessas Previstas")

    if(saveImage and name):
        plt.savefig(name + '.png')

def gerarIndexMesAleatorio(x_treino):
    
    while(True):
        
        dia = randint(0, (x_treino.shape[0] - 311))
        if (x_treino['Dia'][dia] == 1) and x_treino['Cluster_A'][dia] == 1:
            break

    comecoMes = dia

    for i in range(comecoMes, x_treino.shape[0]):

        if(x_treino['Mês'][i] != x_treino['Mês'][dia] or (i == x_treino.shape[0] -1)):
            
            fimMes = i
            break

    return(comecoMes, fimMes)

def gerarIndexMesEspecifico(x_treino, mes, ano):

    if(ano != 2019 and ano != 2020):
        raise ValueError("Ano indicado não existe no conjunto de dados!")

    if((ano == 2019 and mes not in range(10, 13)) or (ano == 2020 and mes not in range(1, 9))):
        raise ValueError("Mês indicado não existe!")

    for i, value in enumerate(x_treino['Mês']):
        
        if((value == mes) and (x_treino['Ano'][i] == ano)):
            
            comecoMes = i
            break

    for i in range(comecoMes, x_treino.shape[0]):
        
        if(x_treino['Mês'][i] != x_treino['Mês'][comecoMes] or (i == x_treino.shape[0] -1)):
            
            fimMes = i
            break
    
    return (comecoMes, fimMes)
# %%

def gerarGraficoMensalClusterEspecifico(x_treino, x_teste, y_treino, y_teste, resultado, comecoMes, specific = False, cluster = None, saveImage = False, name = None):

    plt.clf()

    indexes = x_teste.index

    if (specific and cluster):

        clusterIndexes = indexes[x_teste['Cluster_' + cluster] == 1].tolist()

        sns.lineplot(x = x_teste['Dia'][clusterIndexes], y = resultado[clusterIndexes], label = 'Remessa prevista')

        sns.lineplot(label = 'Remessa atual', x = x_treino['Dia'][clusterIndexes], y = y_teste).set_title('Mês {}/{}'.format(x_teste['Mês'][comecoMes], x_teste['Ano'][comecoMes]))

    else:

        clusters = ['A', 'B', 'C', 'D', 'E', 'F', 'J', 'K', 'L', 'M']

        fig, axs = plt.subplots(nrows = 5, ncols = 2, squeeze = False)

        for index, value in enumerate(clusters):

            clusterIndexes = indexes[x_teste['Cluster_' + value] == 1].tolist()
            
            sns.lineplot(legend = False, x = x_teste['Dia'][clusterIndexes], y = resultado[clusterIndexes], label = 'Remessa prevista', ax = axs[index%5][index//5])

            sns.lineplot(legend = False, x = x_treino['Dia'][clusterIndexes], y = y_teste, label = 'Remessa atual', ax = axs[index%5][index//5]).set_title('Cluster {}'.format(value))  

        handles, labels = axs[0][0].get_legend_handles_labels()
        
        fig.legend(handles, labels, loc='upper right')
        
        fig.suptitle('Mês {}/{}'.format(x_teste['Mês'][comecoMes], x_teste['Ano'][comecoMes]))

        fig.subplots_adjust(hspace = 1.2, wspace = 0.3)

    if(saveImage and name):
        plt.savefig(name + '.png')

# %%
