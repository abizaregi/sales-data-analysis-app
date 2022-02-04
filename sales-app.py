import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from datetime import datetime

st.title('Sales Data Analysis - Application')

st.write("""
***App to view sales data per Channel and per Month-Year***
""")

st.sidebar.header("**User Input Features**")
df = pd.read_csv('df-sales.csv', sep=";")

# Sidebar
sorted_channel_unique = sorted(df['Channel'].unique())
sorted_month_unique = sorted(df['Month-Year'].unique())
selected_channel = st.sidebar.multiselect('Channel', sorted_channel_unique, sorted_channel_unique)
selected_month = st.sidebar.multiselect('Month-Year', sorted_month_unique, sorted_month_unique)

# Filtering data
df_selected = df[(df['Channel'].isin(selected_channel)) & (df['Month-Year'].isin(selected_month))]
st.header('Display Data Selected in Sidebar')
st.write('Data Shape: ' + str(df_selected.shape[0]) + ' rows and ' + str(df_selected.shape[1]) + ' columns.')
st.dataframe(df_selected)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="sales-data.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected), unsafe_allow_html=True)

df_selected['Profit'] = df_selected['UnitPrice']-df_selected['UnitCost']
df_selected['Revenue'] = df_selected['Profit']*df_selected['OrderQuantity']

st.write("""
#### Distribution of Channel
""")
sns.countplot(x=df_selected['Channel'])
st.pyplot()

df_price = df_selected.groupby('Channel').sum()[['OrderQuantity','UnitPrice']]
df_price.sort_values(by=['UnitPrice'], ascending=False).head()
st.dataframe(df_price)

st.write("""
#### Unit Price per Channel ###
""")
price = df_selected.groupby('Channel').sum()['UnitPrice'].round(2)
price.plot(kind='bar', x='Channel', y='UnitPrice', figsize=(10,5), color='darkblue')
plt.xticks(rotation=0)
ax = plt.axes()
ax.yaxis.grid(linestyle='--')
plt.legend()
plt.show()
st.pyplot()

df_series = df_selected.groupby('Month-Year').sum()[['OrderQuantity','UnitPrice']]
df_series = df_series.iloc[:-1]
st.dataframe(df_series.sort_values(by=['UnitPrice'], ascending=False).head(10)))

st.write("""
#### Unit Price per Month-Year ###
""")
sales = df_selected.groupby('Month-Year').sum()['UnitPrice'].round(2)
sales.plot(kind='line', x='Month-Year', y='UnitPrice', figsize=(10,5), color='darkblue', linewidth=3)
plt.legend()
plt.grid()
st.pyplot()

st.write("""
#### Unit Price per Day ###
""")
plt.figure(figsize=(20,12))
df_selected['UnitPrice'].plot()
st.pyplot()

st.write("""
#### Quantity Ordered per Channel
""")
channel1 = df_selected.groupby('Channel').sum()['OrderQuantity'].round(2)
st.bar_chart(channel1)
st.write("""
#### Quantity Ordered per Month-Year
""")
series1 = df_selected.groupby('Month-Year').sum()['OrderQuantity'].round(2)
st.line_chart(series1)

st.write("""
#### Unit Price and Unit Cost per Channel
""")
channel2 = df_selected.groupby('Channel').sum()[['UnitCost','UnitPrice']].round(2)
st.bar_chart(channel2)
st.write("""
#### Unit Price and Unit Cost per Month-Year
""")
series2 = df_selected.groupby('Month-Year').sum()[['UnitCost','UnitPrice']].round(2)
st.line_chart(series2)

st.write("""
#### Profit and Revenue per Channel
""")
channel3 = df_selected.groupby('Channel').sum()[['Profit','Revenue']].round(2)
st.bar_chart(channel3)
st.write("""
#### Profit and Revenue per Month-Year
""")
series3 = df_selected.groupby('Month-Year').sum()[['Profit','Revenue']].round(2)
st.line_chart(series3)

st.write("""
Sum of Profit: 
""", df_selected['Profit'].sum(), """Sum of Revenue""", df_selected['Revenue'].sum())
