from polygon import RESTClient
import datetime as dt
import pandas as pd
from APIKey import getPolygonAPIkey

def GetData(stocks):
    rawData = list()
    client = RESTClient(getPolygonAPIkey())
    for stock in stocks:
        rawData.append(pd.DataFrame(client.get_aggs(
        ticker      = stock,
        multiplier  = 1,
        timespan    = 'hour',
        from_       = '2023-08-01',
        to          = '2023-08-01')))
        
    return rawData

def CheckData():
    pass

def HandleData():
    pass

def main():
    stocks = ["AAPL"]
    rawData = GetData(stocks)
    print(len(rawData[0]))

if __name__=="__main__":
    main()



"""# create client with API key 
client = RESTClient(getPolygonAPIkey())

def cleanDataSingle(dataFrame):
    #Convert UNIX timestamps to milliseconds
    dataFrame['Date'] = dataFrame['timestamp'].apply(
                          lambda x: pd.to_datetime(x*1000000)) 
    #Clean up dataframe
    dataFrame = dataFrame.set_index('Date')
    dataFrame.drop(columns = ['vwap', 'otc', 'timestamp', 'transactions'], axis=1, inplace=True)
    return dataFrame

def cleanDataAll(dataFrame, volumeMin):
    for x in dataFrame.index:
        if dataFrame.loc[x, "volume"] <  volumeMin:
            dataFrame.drop(x, inplace = True)
    dataFrame.drop(columns = ['vwap', 'otc', 'timestamp', 'transactions'], axis=1, inplace=True)
    dataFrame.reset_index(drop=True, inplace=True)
    
    
data = pd.DataFrame(client.get_grouped_daily_aggs(getConfigurations()[0][1]))
cleanDataAll(data, getConfigurations()[1][1])
    """