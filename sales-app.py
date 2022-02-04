import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from datetime import datetime
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from tensorflow import keras

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
st.set_option('deprecation.showPyplotGlobalUse', False)

df_sales = df_selected.groupby('Product').sum()[['Quantity Ordered', 'Price Total']]
st.dataframe(df_sales.sort_values(by=['Price Total'], ascending=False).head())

df_month_year = df_selected.groupby('Month-Year').sum()[['Quantity Ordered', 'Price Total']]
df_month_year = df_month_year.iloc[:-1]
st.dataframe(df_month_year.sort_values(by=['Price Total'], ascending=False))

st.write('''### Price Total per Month-Year''')
sales = df_selected.groupby('Month-Year').sum()['Price Total'].round(2)
sales.plot(kind='line', x='Month-Year', y='Price Total', figsize=(12,8))
plt.legend()
plt.grid()
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

df_selected['Purchase Address City'] = df_selected['Purchase Address'].apply(lambda x: x.split(',')[1][1:])

def cityProduct(city):
    return ' ,'.join(df['Product'][df['Purchase Address City'] == city].value_counts()[:3].index)

df_city = df_selected.groupby('Purchase Address City').sum()[['Quantity Ordered', 'Price Total']].sort_values(by='Price Total', ascending=False)
df_city['Top 3 Product'] = list(map(cityProduct, df_city.index))
st.dataframe(df_city.head())

Qty = df_selected.groupby('Product').sum()['Quantity Ordered'].sort_values(ascending=False).head()
Qty = pd.DataFrame(Qty)
st.dataframe(Qty)

df = pd.read_csv('Sales Harian 2019.csv', sep=",")
st.write('''
Revenue per Days''')
plt.figure(figsize=(10, 8))
df['Price Total'].plot()
plt.ylabel('Revenue ($)')
plt.grid()
plt.show()
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

x = df.drop('Price Total', axis=1)
y = df['Price Total']
x_train, x_test, y_train, y_test = train_test_split(x, y)

def evaluate(y_pred, y_true):
    print(f'MAE : {mean_absolute_error(y_true, y_pred)}')
    print(f'MAE : {mean_squared_error(y_true, y_pred)}')

lr = LinearRegression()
lr.fit(x_train, y_train)
evaluate(lr.predict(x_test), y_test)

df_predict = pd.read_csv('Harian Bulan Januari.csv').drop('Unnamed: 0', axis=1)
df_result = pd.Series(lr.predict(df_predict), pd.date_range(start='1-1-2020', end='31-01-2020', freq='1D'))

def split_sequence(sequence, n_steps=3):
    sequence = list(sequence)
    X, Y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        Y.append(seq_y)
    def reshape(d):
        d = np.array(d)
        d = np.reshape(d,(d.shape[0], d.shape[1],1))
        return d
    return reshape(X), np.array(Y)

train_data = df['Price Total'].iloc[:250]
test_data = df['Price Total'].iloc[250:]

x_train, y_train = split_sequence(train_data)
x_test, y_test = split_sequence(test_data)

model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(3,1,), activation='relu', return_sequences=True),
    keras.layers.LSTM(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(loss='mse', optimizer='adam')
st.write(model.summary())

plt.figure(figsize=(10,8))
plt.plot(model.predict(x_test), label='Prediction')
plt.plot(y_test, label='Actual')
plt.legend()
plt.grid()
plt.title('Data Prediksi vs Data Actual')
plt.xlabel('Waktu')
plt.ylabel('Pendapatan ($)')
plt.savefig('../Output/Demonstrasi Prediksi NN Model')
plt.show()
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

def predict_future(shift_count):
    def reshape(three):
        return np.array(three).reshape(1,3,1)
    array = list(df['Price Total']) + []
    now = len(df['Price Total'])-3
    last = len(df['Price Total'])
    for _ in range(shift_count):
        converted = reshape(array[now:last])
        array.append(model.predict(converted)[0][0])
        now += 1
        last += 1
    return array

future_prediction = predict_future(30)

plt.figure(figsize=(10,5))
plt.plot(np.arange(29,60), future_prediction[-31:], '--', label='Prediksi')
plt.plot(np.arange(30), df['Price Total'][-30:], label='Data Aktual')
plt.title('Prediksi Pendapatan dalam 30 hari ke depan')
plt.grid()
plt.savefig('../Output/Prediksi Dengan NN Model')
plt.legend()
st.set_option('deprecation.showPyplotGlobalUse', False)


