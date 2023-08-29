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

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# Normalize the data
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# df.to_csv('btcusd.csv', index=False)


# # Getting data from the metatrader server

# In[3]:


def get_data(symbol, date_time=datetime.today()):
    # Getting data on the 1-hour timeframe
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # set the symbol and timefram
    timeframe = mt5.TIMEFRAME_H1  # for one-minute bars

    # dates for retrieving the history
    date_from = datetime(2019, 12, 15)
    date_to = date_time

    # get the history
    history = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

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


def get_HnL_Predictions(data):
    # Create a copy of the data to avoid modifying the original DataFrame
    data_copy = data.copy()

    data_copy['hightarget'] = data_copy['high'].shift(-1)
    data_copy['lowtarget'] = data_copy['low'].shift(-1)

    # Drop rows containing NaN values
    data_copy.dropna(inplace=True)

    X = data_copy[['open', 'high', 'low', 'close', 'tick_volume']]
    y = data_copy[['hightarget', 'lowtarget']]

    # Initialize the scalers
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    # Fit and transform the scalers
    scaler_x.fit(X)
    scaler_y.fit(y)

    X_scaled = scaler_x.transform(X)
    y_scaled = scaler_y.transform(y)

    # Transform only the necessary columns for new_data_scaled
    new_data_scaled = scaler_x.transform(data_copy.tail(30)[['open', 'high', 'low', 'close', 'tick_volume']])
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

    # Initialize the model architecture
    model = Net()

    # Load the saved state dictionary into the model
    model.load_state_dict(torch.load('high_and_lows.pth'))

    model.eval()
    with torch.no_grad():
        future_pred_scaled = model(new_data_tensor).numpy()

    # Inverse transform to get original values
    new_HnL_prediction = scaler_y.inverse_transform(future_pred_scaled)

    return new_HnL_prediction

def getModelPredictions(data):
    # Create a copy of the data to avoid modifying the original DataFrame
    data_copy = data.copy()

    data_copy['hightarget'] = data_copy['high'].shift(-1)
    data_copy['lowtarget'] = data_copy['low'].shift(-1)

    # Drop rows containing NaN values
    data_copy.dropna(inplace=True)

    X = data_copy[['open', 'high', 'low', 'close', 'tick_volume']]
    y = data_copy[['hightarget', 'lowtarget']]

    model = load_model('trained_model.keras')

