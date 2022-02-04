import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from datetime import datetime
from wordcloud import WordCloud

st.title('Sales Data Analysis - Application')

st.write("""
***App to view sales data per Channel and per Month-Year***
""")

st.sidebar.header("**User Input Features**")
df = pd.read_csv('df-sales.csv', sep=";")

# Sidebar
sorted_month_unique = sorted(df['Month-Year'].unique())
selected_month = st.sidebar.multiselect('Month-Year', sorted_month_unique, sorted_month_unique)

# Filtering data
df_selected = df[df['Month-Year'].isin(selected_month)]
st.header('Display Data Selected in Sidebar')
st.write('Data Shape: ' + str(df_selected.shape[0]) + ' rows and ' + str(df_selected.shape[1]) + ' columns.')
st.dataframe(df_selected)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="sales-data.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected), unsafe_allow_html=True)

st.write("""
#### Most Product Sold (by Count) ###
""")
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color='white').generate(' '.join(df_selected['Product']))
plt.subplots(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
st.pyplot()

df_sales = df_selcted.groupby('Product').sum()[['Quantity Ordered', 'Price Total']]
st.dataframe(df_sales.sort_values(by=['Price Total'], ascending=False).head())

df_month_year = df_selected.groupby('Month-Year').sum()[['Quantity Ordered', 'Price Total']]
df_month_year = df_month_year.iloc[:-1]
st.dataframe(df_month_year.sort_values(by=['Price Total'], ascending=False))

st.write('''### Price Total per Month-Year''')
sales = df_selected.groupby('Month-Year').sum()['Price Total'].round(2)
sales.plot(kind='line', x='Month-Year', y='Price Total', figsize=(12,8))
plt.legend()
plt.grid()

df_selected['Purchase Address City'] = df_selected['Purchase Address'].apply(lambda x: x.split(',')[1][1:])

def cityProduct(city):
    return ' ,'.join(df['Product'][df['Purchase Address City'] == city].value_counts()[:3].index)

df_city = df_selected.groupby('Purchase Address City').sum()[['Quantity Ordered', 'Price Total']].sort_values(by='Price Total', ascending=False)
df_city['Top 3 Product'] = list(map(cityProduct, df_city.index))
st.dataframe(df_city.head())

Qty = df_selected.groupby('Product').sum()['Quantity Ordered'].sort_values(ascending=False).head()
Qty = pd.DataFrame(Qty)
st.dataframe(Qty)


df_selected['Order Date'] = df_selected['Order Date'].dt.date.astype(np.str)
df_string_date = df_selected.groupby('Order Date').sum().iloc[:-1]

st.write('''### Sales per Days''')
plt.figure(figsize=(10, 8))
df_string_date['Price Total'].plot()
plt.ylabel('Pendapatan ($)')
plt.grid()
plt.show()
st.pyplot()
