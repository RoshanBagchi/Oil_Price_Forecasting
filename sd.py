# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:38:23 2023

@author: 91976
"""
#import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
import datetime as dt
import yfinance as yf
import pandas_ta as ta
from plotly.subplots import make_subplots
from datetime import timedelta
from datetime import date
import numpy as np
 
start = '2000-01-01'
end = '2023-06-28'

st.title('oil Price Prediction')

user_input = st.text_input('oil Stock Ticker', 'ASIANPAINT.NS')

stock_info = yf.Ticker(user_input).info 
# stock_info.keys() for other properties you can explore
company_name = stock_info['shortName']
st.subheader(company_name)
#market_price = stock_info['regularMarketPrice']
previous_close_price = stock_info['regularMarketPreviousClose']
#st.write('market price : ', market_price)
st.write('previous close price : ', previous_close_price)

df=yf.download('CL=F', start='2000-1-1', end='2023-6-28').reset_index(drop=False)

# describing data

st.subheader('Data from 2000-2023')
#df= df.reset_index()
st.write(df.tail(10))
st.write(df.describe())
# Force lowercase (optional)
df.columns = [x.lower() for x in df.columns]

st.subheader("Prediction of oil Price")

# splitting date into training and testing 
data_training= pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['close'][int(len(df)*0.70): int(len(df))])

print("training data: ",data_training.shape)
print("testing data: ", data_testing.shape)


# scaling of data using min max scaler (0,1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


# loading of the saved model

import pickle
loaded_model = pickle.load(open('C:/Users\91976/Desktop/PROJECT DEPLOYMENT/trained_model.sav','rb'))

#testing part
past_100_days = data_training.tail(100)

final_df= past_100_days.append(data_testing, ignore_index =True)

input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)    


y_predicted = loaded_model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]

y_predicted = y_predicted * scale_factor

y_test = y_test* scale_factor


# final Graph
st.subheader("Predictions vs Original")
fig2= plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

   






st.subheader('Stock Price Prediction by Date')

df1=df.reset_index()['close']
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
#datemax="24/06/2022"
datemax=dt.datetime.strftime(dt.datetime.now() - timedelta(1), "%d/%m/%Y")
datemax =dt.datetime.strptime(datemax,"%d/%m/%Y")
x_input=df1[:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


date1 = st.date_input("Enter Date in this format yyyy-mm-dd")

result = st.button("Predict")
#st.write(result)
if result:
	from datetime import datetime
	my_time = datetime.min.time()
	date1 = datetime.combine(date1, my_time)
	#date1=str(date1)
	#date1=dt.datetime.strptime(time_str,"%Y-%m-%d")

	nDay=date1-datemax
	nDay=nDay.days

	date_rng = pd.date_range(start=datemax, end=date1, freq='D')
	date_rng=date_rng[1:date_rng.size]
	lst_output=[]
	n_steps=x_input.shape[1]
	i=0

	while(i<=nDay):
    
	    if(len(temp_input)>n_steps):
        	  #print(temp_input)
        	    x_input=np.array(temp_input[1:]) 
        	    print("{} day input {}".format(i,x_input))
        	    x_input=x_input.reshape(1,-1)
        	    x_input = x_input.reshape((1, n_steps, 1))
        		#print(x_input)
        	    yhat = loaded_model.predict(x_input, verbose=0)
        	    print("{} day output {}".format(i,yhat))
        	    temp_input.extend(yhat[0].tolist())
        	    temp_input=temp_input[1:]
        	    #print(temp_input)
        	    lst_output.extend(yhat.tolist())
        	    i=i+1
	    else:
        	    x_input = x_input.reshape((1, n_steps,1))
        	    yhat = loaded_model.predict(x_input, verbose=0)
        	    print(yhat[0])
        	    temp_input.extend(yhat[0].tolist())
        	    print(len(temp_input))
        	    lst_output.extend(yhat.tolist())
        	    i=i+1
	res =scaler.inverse_transform(lst_output)
#output = res[nDay-1]

	output = res[nDay]

	st.write("*Predicted Price for Date :*", date1, "*is*", np.round(output[0], 2))
	st.success('The Price is {}'.format(np.round(output[0], 2)))

	#st.write("predicted price : ",output)

	predictions=res[res.size-nDay:res.size]
	print(predictions.shape)
	predictions=predictions.ravel()
	print(type(predictions))
	print(date_rng)
	print(predictions)
	print(date_rng.shape)

	@st.cache
	def convert_df(df):
   		return df.to_csv().encode('utf-8')
	df = pd.DataFrame(data = date_rng)
	df['Predictions'] = predictions.tolist()
	df.columns =['Date','Price']
	st.write(df)
	csv = convert_df(df)
	st.download_button(
   		"Press to Download",
   		csv,
  		 "file.csv",
   		"text/csv",
  		 key='download-csv'
	)
	#visualization

	fig =plt.figure(figsize=(10,6))
	xpoints = date_rng
	ypoints =predictions
	plt.xticks(rotation = 90)
	plt.plot(xpoints, ypoints)
	st.pyplot(fig)
