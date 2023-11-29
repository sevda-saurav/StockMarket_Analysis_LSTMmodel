import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
from datetime import datetime
from keras.models import load_model

import streamlit as st 


st.title('Stock Market Prediction')
user_input = st.text_input('Enter Stock Ticker', 'TATAPOWER.NS')

#  Date selection
start = start_date = st.date_input('Select Start date', datetime(2010, 1, 1))
end = end_date = st.date_input('Select End date', datetime.now())

y_symbols = [user_input]   # ['TATAPOWER.NS']
data = pdr.get_data_yahoo(y_symbols,start,end)

# Describing Data
st.subheader('Data Discription')
st.write(data.describe())

# Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.xlabel('Year')
plt.ylabel('Closing Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(data.Close, 'g')
plt.xlabel('Year')
plt.ylabel('Closing Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100 , 'r')
plt.plot(ma200, 'b')
plt.plot(data.Close, 'g')
plt.xlabel('Year')
plt.ylabel('Closing Price')
st.pyplot(fig)

# spliting our data into testing and training

data_tarining = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70): int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_tarining)
# we already trained our model that's why we don't need train spliting model
# load my model 
model = load_model('keras_model.h5')

# Testing 
# for predicting the data for future we need past 100 day data which we can get from testing data
if not data_testing.empty:
    last_100_days = data_tarining.tail(100)
    final_df = pd.concat([last_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test=[]
    y_test=[]

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100 : i])
        y_test.append(input_data[i,0])

    x_test,y_test = np.array(x_test), np.array(y_test)

# Making predictions
    y_predicted = model.predict(x_test)

    scaler = scaler.scale_
    scale_factor = 1/scaler[0]
    y_predicted = y_predicted*scale_factor
    y_test = y_test*scale_factor


# Final graph

    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
else : 
    st.warning("Data testing is empty. Please check your data.")