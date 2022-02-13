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
    
    isWorkingDay = np.where((df['é_fim_de_semana'] == 1) | (df['é_feriado'] == 1), 0, 1)

    return isWorkingDay

def separarColunaData(df, datas):

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

def gerarDados(returnValue):

    #%% leitura do arquivo

    xls = pd.ExcelFile('Plannera - Case de Data Science.xlsx')

    dados = pd.read_excel(xls, 'Dados Históricos')

    objetivo = pd.read_excel('Plano de Volume.xlsx')

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
    
    meanDropsize = (dados['Dropsize'].loc[dados['Dropsize'] != 0]).mean()

    objetivo['Dropsize'] = meanDropsize
    
    indexes = objetivo.index
    
    meanIndexes = indexes[objetivo['é_dia_normal'] != 1].tolist()

    objetivo.loc[meanIndexes, 'Dropsize'] = 0

    objetivo = objetivo.rename(columns = {'CLUSTER': 'Cluster'})

    dados = dados.reindex(sorted(dados.columns), axis=1)

    objetivo = objetivo.reindex(sorted(objetivo.columns), axis=1)

    #%% gerando os  dummies de colunas categóricas

    dados = pd.get_dummies(dados)

    objetivo = pd.get_dummies(objetivo)

    # %% gerando x e y

    if returnValue == True:

        y_treino = dados['Remessas']

        x_treino = dados.drop(columns = 'Remessas')

        x_teste = objetivo

        return (x_treino, x_teste, y_treino)

    else:

        x = dados.drop(columns = 'Remessas')

        y = dados['Remessas']

        return (x, y)

# %%
