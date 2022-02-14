#%%
import datetime
import pandas as pd

def gerarExcel(res, x_teste, mes, ano, nome):
    clusters = []

    period = []

    date = datetime.datetime(ano, mes, 1).date()

    date-= datetime.timedelta(days=1)

    # = datetime.datetime(2020, 8, 31).date()

    if(ano != 2019 and ano != 2020):
        raise ValueError("Ano indicado não existe no conjunto de dados!")

    if((ano == 2019 and mes not in range(10, 13)) or (ano == 2020 and mes not in range(1, 9))):
        raise ValueError("Mês indicado não existe!")

    for i in range(x_teste.shape[0]):
        
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

    strName = 'resultado' + nome + '.xlsx'

    df.to_excel(strName)
# %%
