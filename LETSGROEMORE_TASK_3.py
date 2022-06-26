#!/usr/bin/env python
# coding: utf-8

# LETSGROWMORE_TASK_3
# 
# Author :  Pratiksha Hemraj Salunke
# 
# Task 3 : Exploratory Data Analysis - Retail.
# 
# Perform 'Exploratory Data Analysis' on dataset 'SampleSuperstore'.As a business manager, try to find out the weak areas where you can work to make more profit. What all business problems you can derive by exploring the data?

# Importing Libraries

# In[3]:


#Import the neccesary libraries
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


# In[4]:


from google.colab import drive
drive.mount('/content/drive')


# In[5]:


df = pd.read_csv('/content/drive/MyDrive/Share SampleSuperstore.csv')


# In[6]:


df.head()  #view first 5 rows of the dataset


# Exploratory Data analysis

# In[7]:


df.shape    #returns the no of rows and columns


# In[8]:


df.info()    #Basic summary about the data


# In[9]:


df.isnull().sum()  #checking whether any null values are present


# In[10]:


df.duplicated().sum()


# In[11]:


df.drop_duplicates(inplace=True)


# In[12]:


df.columns


# In[13]:


df.nunique() #gives the count of unique values present in the particular column


# In[14]:


df.drop(columns='Postal Code',axis=1,inplace=True)


# In[15]:


df.describe() #Statistical summary of data


# In[16]:


df['Ship Mode'].value_counts().to_frame()


# In[17]:


df['Segment'].value_counts().to_frame()


# In[18]:


df['Country'].value_counts().to_frame()


# In[19]:


df['Region'].value_counts().to_frame()


# In[20]:


a = pd.DataFrame(df.groupby('Ship Mode')['Sales'].sum().sort_values(ascending=False))
a.reset_index(inplace=True)
a.columns=['Ship Mode','Sales']

b = pd.DataFrame(df.groupby('Ship Mode')['Profit'].sum().sort_values(ascending=False))
b.reset_index(inplace=True)
b.columns=['Ship Mode','Profit']

fig = make_subplots(rows=1,cols=2,subplot_titles=("Ship Mode vs Sales","Ship Mode vs Profit", ))
fig.add_trace(go.Bar(x=a['Ship Mode'], y=a['Sales'],marker=dict(color=[1,2,3,4,5])),1, 1)
fig.add_trace(go.Bar(x=b['Ship Mode'], y=b['Profit'],marker=dict(color=[1,2,3,4,5])),1, 2)

fig.update_xaxes(title_text="Ship Mode", row=1, col=1)
fig.update_xaxes(title_text="Ship Mode", row=1, col=2)
fig.update_yaxes(title_text="Sales", row=1, col=1)
fig.update_yaxes(title_text="Profit",row=1, col=2)

fig.update_layout(showlegend=False)


# In[21]:


a = pd.DataFrame(df.groupby('Segment')['Profit'].sum().sort_values(ascending=False))
a.reset_index(inplace=True)
a.columns=['Segment','Profit']

b = pd.DataFrame(df.groupby('Segment')['Sales'].sum().sort_values(ascending=False))
b.reset_index(inplace=True)
b.columns=['Segment','Sales']

fig = make_subplots(rows=1,cols=2,subplot_titles=("Segment vs Sales","Segment vs Profit"))
fig.add_trace(go.Bar(x=b['Segment'],y=b['Sales'],marker=dict(color=[1,2,3])),1, 1)
fig.add_trace(go.Bar(x=a['Segment'], y=a['Profit'],marker=dict(color=[1,2,3])),1, 2)
fig.update_layout(showlegend=False)

fig.update_xaxes(title_text="Segment", row=1, col=1)
fig.update_xaxes(title_text="Segment", row=1, col=2)
fig.update_yaxes(title_text="Sales", row=1, col=1)
fig.update_yaxes(title_text="Profit",row=1, col=2)
fig.update_layout(showlegend=False)


# In[22]:


df.groupby('Category')['Sub-Category'].value_counts().to_frame()


# In[23]:


Furniture = pd.DataFrame(df[df['Category'] == 'Furniture']['Sub-Category'].value_counts())
Furniture.reset_index(inplace=True)
Furniture.columns = ['Furniture','Count']
Office_Supplies = pd.DataFrame(df[df['Category'] == 'Office Supplies']['Sub-Category'].value_counts())
Office_Supplies.reset_index(inplace=True)
Office_Supplies.columns = ['Office_Supplies','Count']
Technology = pd.DataFrame(df[df['Category'] == 'Technology']['Sub-Category'].value_counts())
Technology.reset_index(inplace=True)
Technology.columns = ['Technology','Count']


# In[24]:


fig = make_subplots(rows=1,cols=3,subplot_titles=("Furniture","Office Supplies", "Technology"))
fig.add_trace(go.Bar(x=Furniture['Furniture'], y=Furniture['Count'],marker=dict(color=[1,2,3,4,5])),1, 1)
fig.add_trace(go.Bar(x=Office_Supplies['Office_Supplies'], y=Office_Supplies['Count'],marker=dict(color=[1,2,3,4,5,6,7,8,9])),1, 2)
fig.add_trace(go.Bar(x=Technology['Technology'], y=Technology['Count'],marker=dict(color=[1,2,3,4,5])),1, 3)

fig.update_yaxes(title_text="Count",row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count",row=1, col=3)
fig.update_layout(showlegend=False)


# In[25]:


a = pd.DataFrame(df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False))
a.reset_index(inplace=True)
a.columns=['Sub-Category','Sales']
fig = px.bar(a,y=a['Sales'],x=a['Sub-Category'],title='Sub-Category vs Sales',color_discrete_sequence=['DarkCyan'])


# In[26]:


data = ['Sales','Quantity','Profit','Discount','State','Category','Sub-Category','Segment']
data=df[data]
data=data.sort_values(by='Profit',ascending=False)
data
df1 = pd.pivot_table(data,index=['Category','Sub-Category'])
df1


# In[27]:


data.pivot_table(values='Profit',index='Segment',columns='Discount',aggfunc='median')


# In[28]:


a = pd.DataFrame(df.groupby('Sub-Category')['Profit'].sum().sort_values(ascending=False))
a.reset_index(inplace=True)
a.columns=['Sub - category','Profit']
fig = px.bar(a,y=a['Profit'],x=a['Sub - category'],title='Sub-Category vs Profit',color_discrete_sequence=['DarkCyan'])
fig.show()


# In[29]:


a = pd.DataFrame(df.groupby('Region')['Profit'].sum().sort_values(ascending=False))
a.reset_index(inplace=True)
a.columns=['Region','Profit']
fig = px.bar(a,y=a['Profit'],x=a['Region'],title='Region vs Profit',color_discrete_sequence=['DarkCyan'],width=600,height=500)
fig.show()


# In[30]:


a = pd.DataFrame(df.groupby('State')['Sales'].sum().sort_values(ascending=False))
a.reset_index(inplace=True)
a.columns=['State','Sales']
fig = px.bar(a,y=a['Sales'],x=a['State'],title='State vs Sales',color_discrete_sequence=['DarkCyan'])
fig.show()


# In[31]:


a = pd.DataFrame(df.groupby('State')['Profit'].sum().sort_values(ascending=False))
a.reset_index(inplace=True)
a.columns=['State','Profit']
fig = px.bar(a,y=a['Profit'],x=a['State'],title='State vs Profit',color_discrete_sequence=['DarkCyan'])
fig.show()


# In[32]:


a = pd.DataFrame(df.groupby('State')['Discount'].sum().sort_values(ascending=False)).head(20)
a.reset_index(inplace=True)
a.columns=['State','Discount']
fig = px.bar(a,y=a['Discount'],x=a['State'],title='State vs Discount',color_discrete_sequence=['darkcyan'])
fig.show()


# In[33]:


a = pd.DataFrame(df.groupby('City')['Profit'].sum().sort_values(ascending=False).head(5))
a.reset_index(inplace=True)
a.columns=['City','Profit']

b = pd.DataFrame(df.groupby('City')['Profit'].sum().sort_values(ascending=False).tail(5))
b.reset_index(inplace=True)
b.columns=['City','Profit']

fig = make_subplots(rows=1,cols=2,subplot_titles=("Top 5 cities with max profit","Top 5 cities with min profit"))
fig.add_trace(go.Bar(x=a['City'],y=a['Profit'],marker=dict(color=[1,2,3,4,5])),1, 1)
fig.add_trace(go.Bar(x=b['City'], y=b['Profit'],marker=dict(color=[1,2,3,4,5])),1, 2)
fig.update_layout(showlegend=False)

fig.update_xaxes(title_text="Segment", row=1, col=1)
fig.update_xaxes(title_text="Segment", row=1, col=2)
fig.update_yaxes(title_text="Profit", row=1, col=1)
fig.update_yaxes(title_text="Profit",row=1, col=2)
fig.update_layout(showlegend=False)


# Conclusion
# 
# Problem Statement : Find out weak areas where you can work to make profit and what all business problem can be derived by exploring data.
# 
# Standard Class in ShipMode has recorded the highest profit and Same Day has recorded the lowest profit.
# 
# There are 3 segments selling products they are Consumer, Corporate & Home Office where Consumer segment has recorded maximum profit followed by Corporate whereas Home Offices recorded minimum profit.
# 
# In United States the products are sold where West region has recorded maximum profit followed by East and lowest being recorded in Central region.
# 
# Top 5 most sold products Sub-Category wise are Phones, Chairs, Storage, Tables & Binders.
# 
# Top 5 least sold products Sub-Category wise are Fasteners, Labels, Envelopes, Art & Supplies.
# 
# When the discount given on a product is beyond 20% then company is getting a loss instead of gainning profit.
# 
# Maximum profit is gained by Copiers, Phones, Accessories ,Paper, Binders whereas Tables has recorded maximim loss followed by Bookcases & Supplies.Hence discount given on these products can be reduced to increase profit.
# 
# Maximum Sales are from states California, New York & Minimum sales are from North Dakota, West Virginia.
# 
# State California & New Yok has recorded the maximum profit whereas Texas, Ohio, Pennsylvania in these states products has occured loss. So discount given in these states can be reduced to increase profit.
# 
# As maximum sales are in states California, NewYork so sales can be increased in these areas to gain profit and In technology category company is getting benefitted so increase in sales of these category can increase profit.
