from polygon import RESTClient
from datetime import datetime
import datetime as dt
import pandas as pd
from APIKey import getPolygonAPIkey

# Future: 
# Add csv for each stock passed
# If CSV exists skip queries
# Add delays to ensure API does not timeout

# Queries API for stocks in list argument
# Returns list of dataframes for each stock
def GetData(stocks):
    rawData = list()
    client = RESTClient(getPolygonAPIkey())
    for stock in stocks:
        rawData.append(
            pd.DataFrame(
                client.get_aggs(
                    ticker      = stock,
                    multiplier  = 1,
                    timespan    = 'day',
                    from_       = datetime(2024, 8, 1, 00, 00, 00),
                    to          = datetime(2024, 8, 30, 00, 00, 00))))
        
    return rawData

# Implement in future
# Returns False
def CheckData(Stock):
    return False

# Deletes unused columns and sets timestamps as index
# Returns list of cleaned dataframes 
def HandleData(rawData):
    cleanData = list()
    for dataFrame in rawData:
        #Convert UNIX timestamps to milliseconds
        dataFrame['Date'] = dataFrame['timestamp'].apply(
                              lambda x: pd.to_datetime(x*1000000)) 
        #Clean up dataframe
        dataFrame = dataFrame.set_index('Date')
        dataFrame.drop(columns = ['vwap', 'otc', 'timestamp', 'transactions'], axis=1, inplace=True)
        cleanData.append(dataFrame)
    return cleanData

def main():
    stocks = ["AAPL"]
    rawData = GetData(stocks)
    cleanData = HandleData(rawData)

    for stock in cleanData:
        if not CheckData(stock):
            stock.to_csv('test.csv', mode = 'a', index = True, header = False)

if __name__=="__main__":
    main()
