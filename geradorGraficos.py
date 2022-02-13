#%%
from multiprocessing.sharedctypes import Value
import seaborn as sns
import matplotlib.pyplot as plt

from linearRegression import gerarResultadoLinearRegression, gerarResultadoRidgeRegression
from randomForest import gerarResultadoRandomForest, gerarResultadoZIRRandomForest

from geradorExcel import gerarExcel
from tratamentoDados import gerarDados

from random import randint

def gerarGraficoMensal(x_treino, x_teste, y_treino, resultado, comecoMes, fimMes, saveImage = False, name = None):

    sns.lineplot(x = x_teste['Dia'], y = resultado, label = 'Remessa prevista')

    sns.lineplot(label = 'Mês analisado', x = x_treino['Dia'][comecoMes:fimMes], y = y_treino[comecoMes:fimMes]).set_title('Mês {}/{}'.format(x_treino['Mês'][comecoMes], x_treino['Ano'][comecoMes]))

    if(saveImage and name):
        plt.savefig(name + '.png')

def gerarGraficoLinear(x_teste, resultado, line, saveImage = False, name = None):

    plt.scatter(x_teste['Volume'], resultado, s = 3)
    plt.plot(x_teste['Volume'], line, color='r', linewidth = 1)

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
    
#%%

x_treino, x_teste, y_treino = gerarDados(returnValue = True)

#%%
resultadoRF = gerarResultadoRandomForest(x_treino, x_teste, y_treino)

#%%

comecoMes, fimMes = gerarIndexMesAleatorio(x_treino)

gerarGraficoMensal(x_treino, x_teste, y_treino, resultadoRF, comecoMes, fimMes)

#%%
gerarExcel(resultadoRF, x_teste, 'RandomForest')

#%%
resultadoZIR = gerarResultadoZIRRandomForest(x_treino, x_teste, y_treino)

#%%

comecoMes, fimMes = gerarIndexMesAleatorio(x_treino)

gerarGraficoMensal(x_treino, x_teste, y_treino, resultadoZIR, comecoMes, fimMes)

#%%
resultadoLR, lineLR = gerarResultadoLinearRegression(x_treino, x_teste, y_treino)

#%%

gerarGraficoLinear(x_teste, resultadoLR, lineLR)

#%%
comecoMes, fimMes = gerarIndexMesAleatorio(x_treino)

gerarGraficoMensal(x_treino, x_teste, y_treino, resultadoLR, comecoMes, fimMes)

#%%
resultadoRR, lineRidge = gerarResultadoRidgeRegression(x_treino, x_teste, y_treino)

#%%
gerarGraficoLinear(x_teste, resultadoRR, lineRidge)

#%%
comecoMes, fimMes = gerarIndexMesAleatorio(x_treino)

gerarGraficoMensal(x_treino, x_teste, y_treino, resultadoRR, comecoMes, fimMes)