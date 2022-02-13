import datetime
import pandas as pd

def gerarExcel(res, x_teste, nome):
    clusters = []

    period = []

    date = datetime.datetime(2020, 8, 31).date()

    for i in range(300):
        
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