#!/usr/bin/env python
# coding: utf-8

# # RETAIL PRICE OPTIMIZATION

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from matplotlib.pyplot import figure
from matplotlib import colors

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = 'plotly_white'

from datetime import datetime


# ## Get The Data

# In[2]:


url = '/Users/yasemincingoz/Desktop/UCSC/Practice:Projects/PYTHON/Retail_Price/retail_price.csv'
df = pd.read_csv(url)

df.head()


# ## Check The Data

# In[3]:


df.info(verbose=True, memory_usage=True)


# In[4]:


# Loop over each column, and use the isnull() method in Pandas to calculate the fraction of missing values
for col in df.columns:
    missing_value = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, missing_value))
    
#There is no missing values


# In[5]:


# Check if there is ant duplicated values
df[df.duplicated(keep=False)]

#There is no duplicated values


# ## Understand The Data

# In[6]:


df.shape

#676 entries, 30 columns


# In[7]:


df.describe()


# In[8]:


df.dtypes


# In[9]:


df.rename(columns={'qty':'quantity', 'month_year':'date', 's':'seasonality'}, inplace=True)
df.head(5)


# In[10]:


# Drop unncessary column(s)

df = df.drop(['product_id'], axis=1)
df.head()


# # Explotary Data Analysis
# 

# In[11]:


# DISTRIBUTION OF TOTAL PRICE

plt.figure(figsize=(12,5))
plt.grid(False)
sns.set_style('white')
sns.histplot(data=df, x='total_price', color='cornflowerblue',bins = 15).set(title = 'Distribution of Total Price')


# In[12]:


# AVERAGE TOTAL PRICE BY PRODUCT CATEGORIES

df1 = df.groupby(['product_category_name'])['total_price'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 5))
plt.bar(df1.index, df1.values, color='skyblue')

plt.grid(False)
plt.title('Product Categories By Average Price')
plt.xlabel('Categories')
plt.ylabel('Total Price')
plt.xticks(rotation=45)

plt.show()

# sns.barplot(x=df1.index, y=df1.values, palatte='pastel', ci=None)


# In[13]:


# RELATIONSHIP BETWEEN QUANTITY AND TOTAL PRICE

fig = px.scatter(df, 
                 x='quantity', 
                 y='total_price', 
                 title='Quantity vs Total Price', trendline="ols")
fig.show()


# In[14]:


# DISTRIBUTION OF UNIT PRICE

# Create a box plot using Plotly Express (px.box)

fig = px.box(df, y='unit_price', title='Box Plot of Unit Price')

# Update box color
box_color = 'lightblue'
fig.update_traces(marker=dict(color=box_color, outliercolor=box_color), selector=dict(type='box'))

# Update layout to remove grid lines
#fig.update_layout(yaxis=dict(showgrid=False), xaxis=dict(showgrid=False))

fig.show()


# In[15]:


# TOTAL PRICE BY WEEKDAY

#weekday = number of weekday in that month

fig = px.box(df, x='weekday', y='total_price', title='Total Price by Weekday')

box_color = 'green'
fig.update_traces(marker=dict(color=box_color, outliercolor=box_color), selector=dict(type='box'))
fig.update_layout(xaxis_title='Weekday', yaxis_title='Total Price')
fig.show()


# In[16]:


# TOTAL PRICE BY HOLIDAY

# weekend=number of weekend in that month

fig = px.box(df, x='holiday', y='total_price', title='Total Price By Holiday')

box_color = 'green'
fig.update_traces(marker=dict(color=box_color, outliercolor=box_color), selector=dict(type='box'))
fig.update_layout(xaxis_title='Holiday', yaxis_title='Total Price')
fig.show()


# In[17]:


plt.figure(figsize=(12, 8))

df1 = df.groupby(['product_category_name'])['quantity'].count().sort_values(ascending=False)

# Create a horizontal bar plot using Seaborn
sns.barplot(x=df1.values, y=df1.index)
sns.set_palette("YlGnBu")
plt.title('Product Categories by Quantity')
plt.xlabel('Quantity Sold')
plt.ylabel('Product Category')
plt.xticks(rotation=0)  # Horizontal x-axis labels for better readability

plt.show()


# ## TIME SERIES ANALYSIS

# In[18]:


# date dtype is object. It should be datetime
df['date']= pd.to_datetime(df['date'])


# In[19]:


df['day'] = df['date'].dt.day
df.head()


# In[20]:


# Create a set of colors
colors = ['#4F6272', '#B7C3F3', '#DD7596', '#8EB897']
df.groupby(['year'])['total_price'].mean().plot.pie(wedgeprops = { 'linewidth' : 1, 'edgecolor' : 'white' }, colors=colors)

# Set the title for the pie plot and adjust the layout
plt.title('Average Total Price by Year', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Total Price')  # Equal aspect ratio ensures the pie plot is circular


# In[21]:


plt.figure(figsize=(20,10))
sns.relplot(data=df, kind="line", x='month', y='total_price', hue='year', palette='bright')

plt.title("Total Price by Month")
plt.xlabel('Month')
plt.ylabel('Total Price')
plt.show()


# # CORRELATION

# In[22]:


# NUMERICAL COLUMNS

numerical_col = df.select_dtypes(exclude=['object']).columns
numerical_col


# In[23]:


categorical_col = df.select_dtypes(include=['object']).columns
categorical_col


# In[24]:


# Correlation heatmap of Numerical Features

plt.figure(figsize=(18,7))
sns.heatmap(df[numerical_col].corr())


# In[25]:


df.corr()[['total_price']].sort_values(by='total_price', ascending=False)


# In[26]:


df.corr()[['unit_price']].sort_values(by='unit_price', ascending=False)

#There is high correlation between unit price and lag price.


# In[27]:


#Analyzing competitors’ pricing strategies is essential in optimizing retail prices. 
#Monitoring and benchmarking against competitors’ prices can help identify opportunities to price competitively, 
#either by pricing below or above the competition, depending on the retailer’s positioning and strategy. 
#The average competitor price difference by product category(COMPETITOR_1)

df['comp1_price_diff'] = df['unit_price'] - df['comp_1']
avg_pricediff_by_category = df.groupby('product_category_name')['comp1_price_diff'].mean().reset_index()

fig = px.bar(avg_pricediff_by_category, 
             x='product_category_name', 
             y='comp1_price_diff', 
             title='Average Competitor Price Difference by Product Category',
             color='product_category_name',
             hover_name="product_category_name", color_discrete_sequence=[
                 px.colors.qualitative.Alphabet[6],
                 px.colors.qualitative.Alphabet[11],
                 px.colors.qualitative.Plotly[2],
                 px.colors.qualitative.Plotly[7],
                 px.colors.qualitative.G10[5]])
fig.update_layout(
    xaxis_title='Product Category',
    yaxis_title='Average Competitor1 Price Difference')
fig.show()


# In[28]:


df['comp2_price_diff'] = df['unit_price'] - df['comp_2']
avg_pricediff_by_category = df.groupby('product_category_name')['comp2_price_diff'].mean().reset_index()

fig = px.bar(avg_pricediff_by_category, 
             x='product_category_name', 
             y='comp2_price_diff', 
             title='Average Competitor Price Difference by Product Category',
             color='product_category_name',
             hover_name="product_category_name", color_discrete_sequence=[
                 px.colors.qualitative.Alphabet[6],
                 px.colors.qualitative.Alphabet[11],
                 px.colors.qualitative.Plotly[2],
                 px.colors.qualitative.Plotly[7],
                 px.colors.qualitative.G10[5]])
fig.update_layout(
    xaxis_title='Product Category',
    yaxis_title='Average Competitor2 Price Difference')
fig.show()


# In[29]:


df['comp3_price_diff'] = df['unit_price'] - df['comp_1']
avg_pricediff_by_category = df.groupby('product_category_name')['comp3_price_diff'].mean().reset_index()

fig = px.bar(avg_pricediff_by_category, 
             x='product_category_name', 
             y='comp3_price_diff', 
             title='Average Competitor Price Difference by Product Category',
             color='product_category_name',
             hover_name="product_category_name", color_discrete_sequence=[
                 px.colors.qualitative.Alphabet[6],
                 px.colors.qualitative.Alphabet[11],
                 px.colors.qualitative.Plotly[2],
                 px.colors.qualitative.Plotly[7],
                 px.colors.qualitative.G10[5]])
fig.update_layout(
    xaxis_title='Product Category',
    yaxis_title='Average Competitor3 Price Difference')
fig.show()


# In[30]:


# PRICE OPTIMIZATION WITH MACHINE LEARNING MODEL

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# In[31]:


X = df[['quantity', 'unit_price', 'comp_1', 
          'product_score', 'comp1_price_diff']]
y = df['total_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[32]:


# Make predictions and have a look at the predicted retail prices and the actual retail prices

y_pred = model.predict(X_test)

fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', 
                         marker=dict(color='blue'), 
                         name='Predicted vs. Actual Retail Price'))
fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], 
                         mode='lines', 
                         marker=dict(color='red'), 
                         name='Ideal Prediction'))
fig.update_layout(
    title='Predicted vs. Actual Retail Price',
    xaxis_title='Actual Retail Price',
    yaxis_title='Predicted Retail Price'
)
fig.show()


# In[ ]:




