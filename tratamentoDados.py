#%%
import pandas as pd
import openpyxl

import numpy as np

import datetime as dt

#classe para acertar os feriados
from pandas.tseries.holiday import(
    AbstractHolidayCalendar, Holiday
)

class BrasilFeriados(AbstractHolidayCalendar):
    rules = [
        Holiday('Independência do Brasil', month = 9, day = 7),
        Holiday('Nª Senhora Aparecida', month = 10, day = 12),
        Holiday('Dia de Finados', month = 11, day = 2),
        Holiday('Proclamação da República', month = 11, day = 15),
        Holiday('Natal', month = 12, day = 25),
        Holiday('Ano Novo', month = 1, day = 1),
        Holiday('Sexta-Feira Santa', year = 2020, month = 4, day = 10),
        Holiday('Tiradentes', month = 4, day = 21),
        Holiday('Dia do Trabalho', month = 5, day = 1)
    ]

def gerarFinsDeSemana(totalDatas):

    datas = totalDatas

    #caso sábado seja dia de trabalho
    #isWeekend = ((pd.DatetimeIndex(datas).dayofweek) == 5).astype(int)

    #caso sábado seja fim de semana
    isWeekend = ((pd.DatetimeIndex(datas).dayofweek) // 5 == 1).astype(int)

    return isWeekend

def gerarFeriados(totalDatas):

    feriados = BrasilFeriados().holidays(start = totalDatas[0], end = totalDatas.iloc[-1])

    isHoliday = np.where(totalDatas.isin(feriados), 1, 0)

    return isHoliday

def gerarDiasNormais(df):

    #gera indices de todos os dias que não são fins de semana nem feriados    
    isWorkingDay = np.where((df['é_fim_de_semana'] == 1) | (df['é_feriado'] == 1), 0, 1)

    return isWorkingDay

def separarColunaData(df, datas):

    #separa as datas em dia, mês e ano

    aux = df

    ano = []
    mes = [] 
    dia = []

    for data in datas:
        
        ano.append(data.year)
        mes.append(data.month)
        dia.append(data.day)

    aux['Ano'] = ano
    aux['Mês'] = mes
    aux['Dia'] = dia

    return aux

def gerarDados(trainTestSplit = False):

    #%% leitura do arquivo dados é o dataset histórico, e o objetivo é o plano do mês 09/2020

    xls = pd.ExcelFile('Plannera - Case de Data Science.xlsx')

    dados = pd.read_excel(xls, 'Dados Históricos')

    objetivo = pd.read_excel(xls, 'Plano de Volume')

    #%% gerar os fins de semana

    dados['é_fim_de_semana'] = gerarFinsDeSemana(dados['DataEntrega'])

    objetivo['é_fim_de_semana'] = gerarFinsDeSemana(objetivo['DATA'])

    #%%gerar os feriados

    dados['é_feriado'] = gerarFeriados(dados['DataEntrega'])

    objetivo['é_feriado'] = gerarFeriados(objetivo['DATA'])

    #%% gerar os dias normais de trabalho

    dados['é_dia_normal'] = gerarDiasNormais(dados)

    objetivo['é_dia_normal'] = gerarDiasNormais(objetivo)

    #%% criar as colunas "dia", "mes" e "ano"  

    dados = separarColunaData(dados, dados['DataEntrega'])

    objetivo = separarColunaData(objetivo, objetivo['DATA'])

    #%% dropando a coluna de Data dos datasets 

    dados.drop(columns = 'DataEntrega', inplace = True)

    objetivo.drop(columns = 'DATA', inplace = True)

    #%% Reorganizando o dataset
    
    #%% gera a média de toda a coluna 'Dropsize', excluindo os valores 0
    
    meanDropsize = (dados['Dropsize'].loc[dados['Dropsize'] != 0]).mean()

    #%% cria a coluna 'Dropsize' no dataset objetivo, já que não existia antes, 
    # e atribui a média de dropsize dos dados históricos a mesma
    
    objetivo['Dropsize'] = meanDropsize
    
    #%% atribui o valor médio de 'Dropsize' a somente os elementos do dataset
    #objetivo que correspondem a um dia normal de trabalho

    indexes = objetivo.index
    
    meanIndexes = indexes[objetivo['é_dia_normal'] != 1].tolist()

    objetivo.loc[meanIndexes, 'Dropsize'] = 0

    #%% renomeia a coluna "Cluster" para ficar padronizado com o arquivo "Plano de Volume"

    objetivo = objetivo.rename(columns = {'CLUSTER': 'Cluster'})

    #%% reordena as colunas feature de ambos os datasets para ficarem padrão

    dados = dados.reindex(sorted(dados.columns), axis=1)

    objetivo = objetivo.reindex(sorted(objetivo.columns), axis=1)

    #%% gerando os dummies de colunas categóricas

    dados = pd.get_dummies(dados)

    objetivo = pd.get_dummies(objetivo)

    # %% gerando x e y

    # caso queira gerar treino e objetivo como treino e teste:

    if trainTestSplit:

        y_treino = dados['Remessas']

        x_treino = dados.drop(columns = 'Remessas')

        x_teste = objetivo

        return (x_treino, x_teste, y_treino)

    # caso queira somente o conjunto de dados históricos para testar um mês aleatório 

    else:
        
        x = dados.drop(columns = 'Remessas')

        y = dados['Remessas']

        return (x, y)

# %% função feita para remover um mês do dataset de dados históricos, criar um outro dataset
# de treino com o mês removido, e reindexar o antigo dataset sem o mês removido

def gerarMesTeste(x, y, mes, ano):

    #%% checagem para ver se o mês e o ano indicado fazem parte do dataset histórico

    if(ano != 2019 and ano != 2020):
        raise ValueError("Ano indicado não existe no conjunto de dados!")

    if((ano == 2019 and mes not in range(10, 13)) or (ano == 2020 and mes not in range(1, 9))):
        raise ValueError("Mês indicado não existe!")


    # %% aqui é para determinar o indice do começo do mês indicado para servir como teste
    for i, value in enumerate(x['Mês']):
        
        if((value == mes) and (x['Ano'][i] == ano)):
            
            comecoMes = i
            break
    
    # %% agora o índice do fim do mês
    for i in range(comecoMes, x.shape[0]):
        
        if(x['Mês'][i] != x['Mês'][comecoMes] or (i == x.shape[0] -1)):
            
            fimMes = i
            break

    x_teste = x.iloc[comecoMes:fimMes]

    y_teste = y.iloc[comecoMes:fimMes]

    x_treino = x.drop(x_teste.index, axis = 0)

    y_treino = y.drop(y_teste.index, axis = 0)

    # %% resetando os indices do dataset histórico para não ficar um buraco, e tambem
    # resetando o indice do dataset novo com o mês teste

    x_treino = x_treino.reset_index(drop = True)
    
    y_treino = y_treino.reset_index(drop = True)

    x_teste = x_teste.reset_index(drop = True)
    
    y_teste = y_teste.reset_index(drop = True)

    return(x_treino, x_teste, y_treino, y_teste)

# %%
