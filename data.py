#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[61]:


# df.to_csv('btcusd.csv', index=False)


# # Getting data from the metatrader server

# In[3]:


def get_data(symbol):

    # Getting data on the 1 hour timeframe
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()

    # set the symbol and timefram
    timeframe = mt5.TIMEFRAME_H1   # for one-minute bars

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
    
    data =rates_frame

    return data

data =get_data('XAUUSD')


# In[4]:


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
    df['BBW'] = df['BB_High'] - df['BB_Low']

    # Calculate MACD
    macd_indicator = MACD(df['close'])
    df['MACD'] = macd_indicator.macd()

    
    return data


# In[5]:


def indicator_signal(df):
    # RSI rules
    df['RSI_signal'] = 0
    df.loc[df['RSI'] > 70, 'RSI_signal'] = 1
    df.loc[df['RSI'] < 30, 'RSI_signal'] = -1
    # Bollinger Bands rules
    df['BB_signal'] = 0
    df.loc[df['close'] > df['BB_High'], 'BB_signal'] = 1
    df.loc[df['close'] < df['BB_Low'], 'BB_signal'] = -1
    # MACD rules
    df['MACD_signal'] = 0
    df.loc[df['MACD'] > 0, 'MACD_signal'] = 1
    df.loc[df['MACD'] < 0, 'MACD_signal'] = -1
    # Moving Averages rules
    df['MA_signal'] = 0
    df.loc[df['close'] > df['MA_daily'], 'MA_signal'] = 1
    df.loc[df['close'] < df['MA_daily'], 'MA_signal'] = -1
    
    #engulfing signal 
    df['Engulfing'] = 0
    for i in range(1, len(df)):
        # Bullish engulfing condition
        if df.loc[i-1, 'open'] > df.loc[i-1, 'close'] and df.loc[i, 'open'] < df.loc[i, 'close'] and df.loc[i-1, 'open'] < df.loc[i, 'close'] and df.loc[i-1, 'close'] > df.loc[i, 'open']:
            df.loc[i, 'Engulfing'] = 1

        # Bearish engulfing condition
        elif df.loc[i-1, 'open'] < df.loc[i-1, 'close'] and df.loc[i, 'open'] > df.loc[i, 'close'] and df.loc[i-1, 'open'] > df.loc[i, 'close'] and df.loc[i-1, 'close'] < df.loc[i, 'open']:
            df.loc[i, 'Engulfing'] = -1
    
    # Remove nas
#     data = df.dropna()
    return data


# In[6]:


def detect_consolidations_and_breakouts(df):
    # initialize new columns
    df['Consolidation'] = 0
    df['Breakout'] = 0

    consolidation_start = None

    # calculate BBW and ATR
    df['BBW'] = df['BB_High'] - df['BB_Low']
    df['HL'] = df['high'] - df['low']
    df['HPC'] = abs(df['high'] - df['close'].shift())
    df['LPC'] = abs(df['low'] - df['close'].shift())
    df['TR'] = df[['HL', 'HPC', 'LPC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df.drop(['HL', 'HPC', 'LPC', 'TR'], axis=1, inplace=True)  # remove temporary columns

    # threshold values for BBW and ATR 
    bbw_threshold = df['BBW'].quantile(0.2)  # for example, the 20th percentile
    atr_threshold = df['ATR'].quantile(0.2)  # for example, the 20th percentile

    for i in range(1, len(df)):
        # check for start of consolidation
        if (df.loc[i, 'BBW'] < bbw_threshold and
            df.loc[i, 'ATR'] < atr_threshold and
            (df.loc[i, 'RSI'] > 30 and df.loc[i, 'RSI'] < 70)):
            consolidation_start = i
            df.loc[i, 'Consolidation'] = 1

        # check for end of consolidation
        elif consolidation_start is not None:
            max_high_during_consolidation = df.loc[consolidation_start:i, 'high'].max()
            min_low_during_consolidation = df.loc[consolidation_start:i, 'low'].min()
            avg_volume_during_consolidation = df.loc[consolidation_start:i, 'tick_volume'].mean()
            current_volume = df.loc[i, 'tick_volume']

            if df.loc[i, 'high'] > max_high_during_consolidation and current_volume > avg_volume_during_consolidation:
                df.loc[i, 'Breakout'] = 1  # upward breakout
                consolidation_start = None  # reset for the next consolidation

            elif df.loc[i, 'low'] < min_low_during_consolidation and current_volume > avg_volume_during_consolidation:
                df.loc[i, 'Breakout'] = -1  # downward breakout
                consolidation_start = None  # reset for the next consolidation

    return df


# In[10]:


def detect_consolidations_and_breakouts(df, min_consolidation_length=3):
    # initialize new columns
    df['Consolidation'] = 0
    df['Breakout'] = 0

    consolidation_start = None
    consolidation_length = 0

    # calculate BBW and ATR
    df['BBW'] = df['BB_High'] - df['BB_Low']
    df['HL'] = df['high'] - df['low']
    df['HPC'] = abs(df['high'] - df['close'].shift())
    df['LPC'] = abs(df['low'] - df['close'].shift())
    df['TR'] = df[['HL', 'HPC', 'LPC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df.drop(['HL', 'HPC', 'LPC', 'TR'], axis=1, inplace=True)  # remove temporary columns

    # threshold values for BBW and ATR 
    bbw_threshold = df['BBW'].quantile(0.2)  # for example, the 20th percentile
    atr_threshold = df['ATR'].quantile(0.2)  # for example, the 20th percentile

    for i in range(1, len(df)):
        # check for start of consolidation
        if (df.loc[i, 'BBW'] < bbw_threshold and
            df.loc[i, 'ATR'] < atr_threshold and
            (df.loc[i, 'RSI'] > 30 and df.loc[i, 'RSI'] < 70)):
            if consolidation_start is None:
                consolidation_start = i
            consolidation_length += 1
            df.loc[i, 'Consolidation'] = 1

        # check for end of consolidation
        elif consolidation_start is not None and consolidation_length >= min_consolidation_length:
            max_high_during_consolidation = df.loc[consolidation_start:i, 'high'].max()
            min_low_during_consolidation = df.loc[consolidation_start:i, 'low'].min()
            avg_volume_during_consolidation = df.loc[consolidation_start:i, 'tick_volume'].mean()
            current_volume = df.loc[i, 'tick_volume']

            if df.loc[i, 'high'] > max_high_during_consolidation: #and current_volume > avg_volume_during_consolidation:
                df.loc[i, 'Breakout'] = 1  # upward breakout
                consolidation_start = None  # reset for the next consolidation
                consolidation_length = 0

            elif df.loc[i, 'low'] < min_low_during_consolidation:# and current_volume > avg_volume_during_consolidation:
                df.loc[i, 'Breakout'] = -1  # downward breakout
                consolidation_start = None  # reset for the next consolidation
                consolidation_length = 0

        else:
            consolidation_start = None  # reset for the next consolidation
            consolidation_length = 0

    return df


# In[14]:


def add_consolidation_breakout_indicator(df):
    df = df.copy()

    # initialize new columns
    df['Consolidation_wv'] = 0
    df['Breakout_wv'] = 0

    consolidation_start = None

    for i in range(2, len(df)):
        # check for start of consolidation
        if abs(df.loc[i, 'high'] - df.loc[i, 'low']) <= abs(df.loc[i-1, 'high'] - df.loc[i-1, 'low']) and abs(df.loc[i-1, 'high'] - df.loc[i-1, 'low']) <= abs(df.loc[i-2, 'high'] - df.loc[i-2, 'low']):
            consolidation_start = i
            df.loc[i, 'Consolidation_wv'] = 1

        # check for end of consolidation
        elif consolidation_start is not None:
            max_high_during_consolidation = df.loc[consolidation_start:i, 'high'].max()
            min_low_during_consolidation = df.loc[consolidation_start:i, 'low'].min()

            if df.loc[i, 'high'] > max_high_during_consolidation or df.loc[i, 'low'] < min_low_during_consolidation:
                df.loc[i, 'Breakout_wv'] = 1
                consolidation_start = None  # reset for the next consolidation

    return df.dropna()


# In[15]:


data =apply_indicators(data)
data =indicator_signal(data)
datam = detect_consolidations_and_breakouts(data)
data.to_csv('xauusd.csv', index=False)


# In[16]:


datam


# In[135]:


# Make sure your 'time' column is a datetime
datam['time'] = pd.to_datetime(datam['time'])

# Set 'time' as index for the DataFrame because mplfinance requires Date index for plotting
datam.set_index('time', inplace=True)

# Rename the 'tick_volume' column to 'volume'
datam.rename(columns={'tick_volume': 'volume'}, inplace=True)

# Create a dictionary for market colors
mc = mpf.make_marketcolors(up='green',down='red',wick='inherit',edge='inherit')

# Create a style based on the market colors
s = mpf.make_mpf_style(marketcolors=mc)

# Create the plot
mpf.plot(datam, type='candle', style=s, title='Candlestick Plot', volume=True)

# Add colored bars based on 'Consolidation' column
for i in range(len(datam)):
    if datam['Consolidation'].iloc[i] == 1:
        plt.bar(datam.index[i], datam['high'].iloc[i], color='r')

plt.show()


# In[ ]:




