import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator

import numpy as np
from sklearn.preprocessing import MinMaxScaler
# In[61]:


# df.to_csv('btcusd.csv', index=False)


# # Getting data from the metatrader server

# In[3]:


def get_data(symbol):
    # Getting data on the 1 hour timeframe
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # set the symbol and timefram
    timeframe = mt5.TIMEFRAME_H1  # for one-minute bars

    # dates for retrieving the history
    date_from = datetime(2019, 12, 15)
    today = datetime.today()

    # get the history
    history = mt5.copy_rates_range(symbol, timeframe, date_from, today)

    if history is not None and len(history) > 0:
        # create DataFrame out of the obtained data
        rates_frame = pd.DataFrame(history).drop(['spread', 'real_volume'], axis=1)

        # convert time in seconds into the datetime format
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    else:
        print("No data for the requested period")

    # terminate the connection to the MetaTrader 5 terminal
    mt5.shutdown()

    data = rates_frame

    return data



# In[4]:


def apply_indicators(df):
    print('data in indicators  ', df)
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

    return df

def add_required_indicators(data):
    data['RSI'] = ta.rsi(data.close, length=15)
    data['EMAF'] = ta.ema(data.close, length=20)
    data['EMAM'] = ta.ema(data.close, length=100)
    data['EMAS'] = ta.ema(data.close, length=150)

    data['Target'] = data['close'] - data.open
    data['Target'] = data['Target'].shift(-1)

    data['TargetClass'] = [1 if data.Target[i] > 0 else 0 for i in range(len(data))]

    data['TargetNextClose'] = data['close'].shift(-1)

    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    data.drop(['tick_volume', 'close', 'time'], axis=1, inplace=True)
    pd.set_option('display.max_columns', None)
    return data

def scale_data(data_set):
    sc = MinMaxScaler(feature_range=(0, 1))
    data_set_scaled = sc.fit_transform(data_set)
    return data_set_scaled


def prepare_realtime_data(new_data_set_scaled,  backcandles=30, feature_count=8):
    """
    Prepare real-time data for making predictions based on how the training data was preprocessed.

    Parameters:
    - new_data_set: DataFrame or 2D array-like, the new real-time data
    - sc: trained MinMaxScaler object
    - backcandles: int, the number of past records to use for each sequence
    - feature_count: int, the number of features in the data set

    Returns:
    - X_new: 3D numpy array, preprocessed real-time data suitable for predictions
    """

    # Step 1: Use the same MinMaxScaler to scale new data
    # Step 2: Create sequences (like rolling window) of last 'backcandles' records
    X_new = []

    for j in range(feature_count):  # Number of feature columns
        X_new.append([])
        for i in range(backcandles, new_data_set_scaled.shape[0]):
            X_new[j].append(new_data_set_scaled[i - backcandles:i, j])

    # Step 3: Reshape data
    X_new = np.moveaxis(X_new, [0], [2])

    return X_new
