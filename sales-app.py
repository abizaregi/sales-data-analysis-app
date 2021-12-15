import streamlit as st
import pandas as pd
import base64

st.title('Sales Data Analysis - Application')

st.write("""
***App to view sales data per Channel and per Month-Year***
""")

st.sidebar.header("***User Input Features***")
df = pd.read_csv('df-sales.csv', sep=";")
channel = df.groupby('Channel')

# Sidebar - Channel selection
sorted_channel_unique = sorted(df['Channel'].unique())
sorted_month_unique = sorted(df['Month-Year'].unique())
selected_channel = st.sidebar.multiselect('Channel', sorted_channel_unique, sorted_channel_unique)
selected_month = st.sidebar.multiselect('Month-Year', sorted_month_unique, sorted_month_unique)

# Filtering data
df_selected_channel = df[(df['Channel'].isin(selected_channel))]
df_selected_month = df[(df['Month-Year'].isin(selected_month))]
st.header('Display Data in Selected Channel')
st.write('Data Dimension: ' + str(df_selected_channel.shape[0]) + ' rows and ' + str(df_selected_channel.shape[1]) + ' columns.')
st.dataframe(df_selected_channel.style.highlight_max(axis=0,color='purple'))

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="sales-data.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_channel), unsafe_allow_html=True)

st.write("""
#### Quantity Ordered per Channel
""")
channel1 = df_selected_channel.groupby('Channel').sum()['OrderQuantity'].round(2)
st.bar_chart(channel1)
st.write("""
#### Unit Price per Channel
""")
channel2 = df_selected_channel.groupby('Channel').sum()[['UnitCost','UnitPrice']].round(2)
st.bar_chart(channel2)

df_selected_channel['Profit'] = df_selected_channel['UnitPrice']-df_selected_channel['UnitCost']
df_selected_channel['Revenue'] = df_selected_channel['Profit']*df_selected_channel['OrderQuantity']

st.write("""
#### Profit per Channel
""")
channel3 = df_selected_channel.groupby('Channel').sum()[['Profit','Revenue']].round(2)
st.bar_chart(channel3)
st.write("""
Sum of Profit: 
""", df_selected_channel['Profit'].sum(), """Sum of Revenue""", df_selected_channel['Revenue'].sum())

st.header('Display Data in Selected Month-Year')
st.write('Data Dimension: ' + str(df_selected_channel.shape[0]) + ' rows and ' + str(df_selected_channel.shape[1]) + ' columns.')
st.dataframe(df_selected_month)
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="sales-data.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_channel), unsafe_allow_html=True)

st.write("""
#### Quantity Ordered per Month-Year
""")
series1 = df_selected_month.groupby('Month-Year').sum()['OrderQuantity'].round(2)
st.line_chart(series1)
st.write("""
#### Unit Price per Month-Year
""")
series2 = df_selected_month.groupby('Month-Year').sum()[['UnitCost','UnitPrice']].round(2)
st.line_chart(series2)

df_selected_month['Profit'] = df_selected_channel['UnitPrice']-df_selected_channel['UnitCost']
df_selected_month['Revenue'] = df_selected_channel['Profit']*df_selected_channel['OrderQuantity']

st.write("""
#### Profit per Month-Year
""")
series3 = df_selected_month.groupby('Month-Year').sum()[['Profit','Revenue']].round(2)
st.line_chart(series3)

st.write("""
Sum of Profit: 
""", df_selected_month['Profit'].sum(), """Sum of Revenue""", df_selected_month['Revenue'].sum())
st.sidebar.write("""
Created by **Abizar Egi Mahendra**
""")
