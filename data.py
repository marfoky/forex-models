import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator


def apply_indicators(df):
    
    # Calculate the moving averages
    df['MA_daily'] = df['close'].rolling(window=50).mean()
    df['MA_weekly'] = df['close'].rolling(window=200).mean()

    # Calculate the Relative Strength Index (RSI)
    rsi_indicator = RSIIndicator(df['close'])
    df['RSI'] = rsi_indicator.rsi()

    # Calculate Bollinger Bands
    bollinger = BollingerBands(df['close'])
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()

    # Calculate MACD
    macd_indicator = MACD(df['close'])
    df['MACD'] = macd_indicator.macd()

    # Remove nas
    data = df.dropna()
    
    return data

def get_data(symbol):

    # Getting data on the 15 minutes timeframe
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()

    # set the symbol and timefram
    timeframe = mt5.TIMEFRAME_M15   # for one-minute bars

    #dates for retrieving the history
    date_from = datetime(2016, 12, 15)
    today = datetime.today()

    # get the history
    history = mt5.copy_rates_range(symbol, timeframe, date_from, today)

    if history is not None and len(history) > 0:
        # create DataFrame out of the obtained data
        rates_frame = pd.DataFrame(history).drop(['spread','real_volume'], axis =1)

        # convert time in seconds into the datetime format
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    else:
        print("No data for the requested period")

    # terminate the connection to the MetaTrader 5 terminal
    mt5.shutdown()
    
    #adding indicators to retrieved data
    data = apply_indicators(rates_frame)
    
    
    return data