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

df.head(10)


# ## Check The Data

# In[3]:


df.info(verbose=True, memory_usage=True)


# In[4]:


for col in df.columns:
    missing_value = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, missing_value))
    
#There is no missing values


# In[5]:


# Check if there is any duplicated values

df[df.duplicated(keep=False)]

#There is no duplicated values


# ## Understand The Data

# In[6]:


df.shape

#676 entries, 30 columns


# In[7]:


#Descriptive statistics of the data

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


# In[11]:


# Highest total price and product category

df.sort_values('total_price', ascending=False).head(10)


# In[12]:


# Minumum total price 

df.sort_values('total_price', ascending=True).head(10)


# # Explotary Data Analysis
# 

# In[13]:


# DISTRIBUTION OF NUMERIC VALUES


# In[14]:


k=1
plt.figure(figsize=(12,12))

for i in df.select_dtypes('int'):
    plt.subplot(9,3,k)
    sns.histplot(df[i], kde=True, color='darkblue')
    plt.title(i)
    k+=1
   
plt.tight_layout()
plt.show()


# In[15]:


# DISTRIBUTION OF TOTAL PRICE

plt.figure(figsize=(12,5))
plt.grid(False)
sns.set_style('white')

sns.histplot(data=df, x='total_price', color='cornflowerblue',bins = 15).set(title = 'Distribution of Total Price')


# In[16]:


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


# In[46]:


# RELATIONSHIP BETWEEN QUANTITY AND TOTAL PRICE

fig = px.scatter(df, 
                 x='quantity', 
                 y='total_price',
                 title='Quantity vs Total Price', trendline="ols", trendline_color_override="red")
fig.show()


# In[47]:


# DISTRIBUTION OF UNIT PRICE

fig = px.box(df, y='unit_price', title='Distribution of Unit Price')

# Update box color
box_color = 'lightblue'
fig.update_traces(marker=dict(color=box_color, outliercolor=box_color), selector=dict(type='box'))

# Update layout to remove grid lines
#fig.update_layout(yaxis=dict(showgrid=False), xaxis=dict(showgrid=False))

fig.update_layout(yaxis_title='Unit Price')

fig.show()


# In[56]:


# PRODUCT CATEGORIES BY QUANTITY

plt.figure(figsize=(12, 8))

df1 = df.groupby(['product_category_name'])['quantity'].count().sort_values(ascending=False)
category_colors = sns.color_palette("mako", len(df1))

# Create a horizontal bar plot using Seaborn
sns.barplot(x=df1.values, y=df1.index, palette=category_colors)

sns.set_palette("BuPu")
plt.title('Product Categories by Quantity')
plt.xlabel('Quantity Sold')
plt.ylabel('Product Category')
plt.xticks(rotation=0)  

plt.show()


# In[58]:


# CUSTOMERS PER MONTH BY YEARS

plt.figure(figsize=(12,5))

sns.barplot(data=df, x='month', y='customers', hue='year', palette='Blues').set(title = 'Customers per Month')
plt.grid(False)
sns.set_style('white')


# ## TIME SERIES ANALYSIS

# In[60]:


# TOTAL PRICE BY WEEKDAY

# weekday = number of weekdays in that month

fig = px.box(df, x='weekday', y='total_price', title='Total Price by Weekday')

box_color = 'teal'
fig.update_traces(marker=dict(color=box_color, outliercolor=box_color), selector=dict(type='box'))
fig.update_layout(xaxis_title='Weekday', yaxis_title='Total Price')
fig.show()


# In[59]:


fig = px.violin(df, x='weekday', y='total_price', title='Average Total Price by Weekday', color='weekday',
               color_discrete_sequence=["lightgreen", "teal", "lightblue", "darkblue"])
fig.update_layout(xaxis_title='Weekday', yaxis_title='Average Total Price')
fig.show()


# In[23]:


# TOTAL PRICE BY HOLIDAY

# holiday=number of holidays in that month

fig = px.box(df, x='holiday', y='total_price', title='Total Price By Holiday')

box_color = 'skyblue'
fig.update_traces(marker=dict(color=box_color, outliercolor=box_color), selector=dict(type='box'))
fig.update_layout(xaxis_title='Holiday', yaxis_title='Total Price')
fig.show()


# In[24]:


# TOTAL PRICE BY WEEKEND

# weekend = number of weekend in that month

fig = px.box(df, x='weekend', y='total_price', title='Total Price By Weekend')

box_color = 'darkblue'
fig.update_traces(marker=dict(color=box_color, outliercolor=box_color), selector=dict(type='box'))
fig.update_layout(xaxis_title='Weekend', yaxis_title='Total Price')
fig.show()


# In[61]:


# date dtype is object. It should be datetime

df['date']= pd.to_datetime(df['date'])


# In[26]:


df['day'] = df['date'].dt.day
df.head()


# In[63]:


# Create a set of colors
colors=['teal', 'lightblue']
df.groupby(['year'])['total_price'].mean().plot.pie(wedgeprops = { 'linewidth' : 1, 'edgecolor' : 'white' }, colors=colors)

# Set the title for the pie plot and adjust the layout
plt.title('Average Total Price by Year', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Total Price')  


# In[66]:


# TOTAL PRICE BY MONTH

plt.figure(figsize=(20,10))
sns.relplot(data=df, kind="line", x='month', y='total_price', hue='year', palette='bright')

plt.title("Total Price by Month")
plt.xlabel('Month')
plt.ylabel('Total Price')
plt.show()


# In[68]:


# LAG PRICE TRENDS

df1 = df.groupby(['month'])['lag_price'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(df1['month'], df1['lag_price'], color='darkblue', linestyle='-')

plt.xlabel('Month')
plt.ylabel('Lag Price')
plt.title('Lag Price Trends Over Time')

plt.grid(False)
plt.show()


# ## Comparison with Competitors

# #### Product Price

# In[35]:


#Analyzing competitors’ pricing strategies is essential in optimizing retail prices. 
#Monitoring and benchmarking against competitors’ prices can help identify opportunities to price competitively, 
#either by pricing below or above the competition, depending on the retailer’s positioning and strategy. 
#The average competitor price difference by product category(COMPETITOR_1)

df['comp1_price_diff'] = df['unit_price'] - df['comp_1']

fig = px.bar(df, 
             x='product_category_name', 
             y='comp1_price_diff', 
             title='Competitor Price Difference by Product Category',
             color='product_category_name',
             hover_name="product_category_name", color_discrete_sequence=px.colors.sequential.Viridis)
fig.update_layout(
    xaxis_title='Product Category',
    yaxis_title='Competitor1 Price Difference')
fig.show()


# In[36]:


df['comp2_price_diff'] = df['unit_price'] - df['comp_2']
#avg_pricediff_by_category = df.groupby('product_category_name')['comp2_price_diff'].mean().reset_index()

fig = px.bar(df, 
             x='product_category_name', 
             y='comp2_price_diff', 
             title='Competitor Price Difference by Product Category',
             color='product_category_name',
             hover_name="product_category_name", color_discrete_sequence=px.colors.sequential.Viridis)
fig.update_layout(
    xaxis_title='Product Category',
    yaxis_title='Competitor2 Price Difference')
fig.show()


# In[37]:


df['comp3_price_diff'] = df['unit_price'] - df['comp_1']
#avg_pricediff_by_category = df.groupby('product_category_name')['comp3_price_diff'].mean().reset_index()

fig = px.bar(df, 
             x='product_category_name', 
             y='comp3_price_diff', 
             title='Competitor Price Difference by Product Category',
             color='product_category_name',
             hover_name="product_category_name", color_discrete_sequence=px.colors.sequential.Viridis)
fig.update_layout(
    xaxis_title='Product Category',
    yaxis_title='Competitor3 Price Difference')
fig.show()


# #### Shipping Price

# In[84]:


df['comp1_freight_diff'] = df['freight_price'] - df['fp1']
df['comp2_freight_diff'] = df['freight_price'] - df['fp2']
df['comp3_freight_diff'] = df['freight_price'] - df['fp3']

for i in range (1,4):
    comp = f"comp{i}_freight_diff"
    fig = px.bar(x=df['product_category_name'], y=df[comp],
                title = f"Competitor{i} Shipping Price Difference",
                labels = {
                        'x':'Product Category',
                        'y': f"Competitor{i} Shipping Price"
                }
    )
    fig.show()


# ## CORRELATION

# In[70]:


# NUMERICAL COLUMNS

numerical_col = df.select_dtypes(exclude=['object']).columns
numerical_col


# In[71]:


categorical_col = df.select_dtypes(include=['object']).columns
categorical_col


# In[72]:


# Correlation heatmap of Numerical Features

plt.figure(figsize=(18,7))
sns.heatmap(df[numerical_col].corr())


# In[73]:


df.corr()[['total_price']].sort_values(by='total_price', ascending=False)


# In[74]:


df.corr()[['lag_price']].sort_values(by='lag_price', ascending=False)

# There is high correlation between unit price and lag price.


# ## FEATURE ENGINEERING

# In[39]:


# Feature Engineering is the cruical step in the process of preparing data for machine learning models.
# The goal of the feature engineering is to transform raw data into a format that can be better understood by the model,
# allowing it to make more accurate predictions or classifications.


# ### Linear Regression

# In[40]:


# Liner Regression can be used to model the relationship between product prices and vaious factors affecting them.


# In[41]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[42]:


# split the data into training and test datasets

X = df[['quantity', 'unit_price', 'comp_1', 
          'product_score', 'comp1_price_diff']]
y = df['total_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[43]:


# Create a model from the training dataset

linearModel = LinearRegression()
linearModel.fit(X_train, y_train)


# In[44]:


# Validate the model with the test dataset. 
# The score() method passes the x_test and y_test variables of the test dataset to the model, which returns an R² value
# R² value that scores the model.

linearModel.score(X_test, y_test)

# 0.82 indicates that about %82  of the change in the dependent variable can be attributed to the independent variable.
# But it doesn't tell you whether the model's predictions are accurate or whether the model meets the specific requirements 


# In[45]:


# Use the model to make predicitons

y_predicted = linearModel.predict(X_test)

fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test, y=y_predicted, mode='markers', 
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




