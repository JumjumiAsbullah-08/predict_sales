import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    return df

def preprocess_data(df):
    st.write("Data sebelum preprocessing:")
    st.write(df.head())
    st.write(df.info())

    # Encoding categorical variables
    df = pd.get_dummies(df, columns=['NAMABARANG', 'NAMAOUTLET', 'NAMASALESMAN'], drop_first=True)
    st.write("Setelah encoding kategorikal:")
    st.write(df.head())

    # Converting dates
    reference_date = pd.Timestamp('2023-01-01')
    if 'TGL_INVC' in df.columns:
        df['TGL_INVC'] = pd.to_datetime(df['TGL_INVC'], errors='coerce')
        df['TGL_INVC'] = (df['TGL_INVC'] - reference_date).dt.days
    else:
        st.error("'TGL_INVC' tidak ada dalam dataset.")
        return pd.DataFrame()

    if 'EXP_DATE' in df.columns:
        df['EXP_DATE'] = pd.to_datetime(df['EXP_DATE'], errors='coerce')
        df['EXP_DATE'] = (df['EXP_DATE'] - reference_date).dt.days
    else:
        st.error("'EXP_DATE' tidak ada dalam dataset.")
        return pd.DataFrame()

    st.write("Setelah konversi tanggal ke days:")
    st.write(df[['TGL_INVC', 'EXP_DATE']].head())

    # Removing NaN values
    df = df.dropna()
    st.write(f"Jumlah data setelah dropna: {len(df)}")
    st.write(df.head())
    
    # Remove infinite values if any
    df.replace([float('inf'), -float('inf')], np.nan, inplace=True)
    df = df.dropna()

    st.write(f"Jumlah data setelah dropna dan replace inf: {len(df)}")
    st.write(df.head())
    
    return df

def create_dummy_data(df, target_date):
    """Creates dummy data for prediction based on the last available data."""
    reference_date = pd.Timestamp('2023-01-01')
    dummy = df.sample(n=1, random_state=42).copy()  # Sample one row of data
    dummy['TGL_INVC'] = (pd.Timestamp(f'{target_date}-01') - reference_date).days
    dummy['EXP_DATE'] = dummy['TGL_INVC'] + 30  # Adjust as needed, e.g., assume 30 days after TGL_INVC
    dummy = dummy.drop(['QTYSALES'], axis=1)  # Drop target variable, as it's what we're predicting
    return dummy

def train_and_predict(df, target_date):
    df = preprocess_data(df)

    if df.empty:
        st.error("Data tidak cukup untuk melatih model setelah preprocessing.")
        return None, None, None, None

    X = df.drop(['QTYSALES'], axis=1)
    y = df['QTYSALES']
    
    if len(X) == 0 or len(y) == 0:
        st.error("Tidak ada data fitur atau target setelah preprocessing.")
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if X_train.empty or X_test.empty:
        st.error("Pembagian data tidak cukup. Coba dengan data yang lebih besar.")
        return None, None, None, None

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Creating dummy data for the target prediction period
    df_target = create_dummy_data(df, target_date)

    if df_target.empty:
        st.warning("Tidak ada data untuk periode yang dipilih.")
        return None, mse, mae, r2

    predictions = model.predict(df_target)
    
    return predictions, mse, mae, r2

st.title('Prediksi Penjualan di Apotek')

st.sidebar.header('Masukkan Parameter Prediksi')
file_path = st.sidebar.file_uploader('Upload File Excel', type=['xlsx'])

if file_path is not None:
    df = load_data(file_path)
    st.subheader('Data yang Dimuat')
    st.write(df)
    
    bulan_tahun_options = ['Semua Bulan dan Tahun'] + pd.to_datetime(df['TGL_INVC']).dt.to_period('M').astype(str).unique().tolist()
    start_date = '2023-09-01'
    end_date = '2024-08-31'
    tahun_prediksi_options = pd.date_range(start=start_date, end=end_date, freq='M').strftime('%Y-%m').tolist()
    
    bulan_tahun_prediksi = st.sidebar.selectbox('Bulan dan Tahun Data Set', bulan_tahun_options)
    tahun_prediksi = st.sidebar.selectbox('Bulan dan Tahun Prediksi', ['Semua Bulan dan Tahun'] + tahun_prediksi_options)
    
    nama_barang = st.sidebar.selectbox('Nama Barang', [''] + list(df['NAMABARANG'].unique()))
    nama_salesman = st.sidebar.selectbox('Nama Salesman', [''] + list(df['NAMASALESMAN'].unique()))
    
    if st.sidebar.button('Prediksi'):
        if tahun_prediksi == 'Semua Bulan dan Tahun':
            st.warning("Silakan pilih Bulan dan Tahun Prediksi.")
        else:
            predictions, mse, mae, r2 = train_and_predict(df, tahun_prediksi)
            if predictions is not None:
                st.write(f"Prediksi QTYSALES untuk {tahun_prediksi}: {predictions.sum():.2f}")
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"Mean Absolute Error: {mae:.2f}")
                st.write(f"R2 Score: {r2:.2f}")
else:
    st.sidebar.text('Mohon upload file Excel untuk memulai prediksi.')
