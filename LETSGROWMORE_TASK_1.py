#!/usr/bin/env python
# coding: utf-8

# # LetsGrowMore Task1
# 
# #TASK1: Stock Market Prediction And Forecasting Using Stacked LSTM
# NAME : Pratiksha Hemraj Salunke
# 
# Dataset : https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv

# In[1]:


from google . colab import drive


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# Reading the Dataset

# In[3]:


data = pd.read_csv("https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv")
data.head()


# Data sorting

# In[4]:


data['Date']=pd.to_datetime(data['Date'])
print(type(data.Date[0]))


# In[5]:


df=data.sort_values(by='Date')
df.head()


# Visualization

# In[6]:


plt.plot(df['Close'])


# In[7]:


dff=df['Close']
dff


# MIN MAX scaler

# In[8]:


scaler=MinMaxScaler(feature_range=(0,1))
dff=scaler.fit_transform(np.array(dff).reshape(-1,1))
dff


# Spliting The Dataset

# In[9]:


training_size=int(len(dff)*0.70)
test_size=len(dff)-training_size
train_data,test_data=dff[0:training_size,:],dff[training_size:len(dff),:1]


# convert an array of values into a dataset Matrixconvert an array of values into a dataset Matrix

# In[10]:


def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


# Spliting Data into Train and Test

# In[11]:


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[12]:


print(X_train.shape), print(y_train.shape)


# In[13]:


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# Creating the LSTM Model

# In[14]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()


# In[15]:


model.fit(X_train,y_train,validation_split=0.1,epochs=60,batch_size=64,verbose=1)


# In[16]:


model.fit(X_train,y_train,validation_split=0.1,epochs=60,batch_size=64,verbose=1)


# In[21]:


test_predict=model.predict(X_test)


# In[22]:


test_predicted=scaler.inverse_transform(test_predict)
test_predicted


# Calculating performance

# In[23]:


performance = math.sqrt(mean_squared_error(ytest,test_predict))
performance

