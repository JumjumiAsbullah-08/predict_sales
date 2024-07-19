import streamlit as st
import pandas as pd
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import plotly.express as px
import matplotlib.pyplot as plt

# Function to preprocess data and train model
def preprocess_data_and_train_model(df):
    # Convert categorical variables to numerical using dummy encoding
    df_encoded = pd.get_dummies(df, columns=['NAMABARANG', 'NAMAOUTLET', 'NAMASALESMAN'])
    
    # Ensure all necessary columns exist in encoded dataset
    necessary_columns = [
        'NAMABARANG_Obat A', 'NAMABARANG_Obat B', 'NAMABARANG_Obat C',
        'NAMAOUTLET_Apotek A', 'NAMAOUTLET_Apotek B', 'NAMAOUTLET_Apotek C',
        'NAMASALESMAN_John Doe', 'NAMASALESMAN_Jane Smith', 'NAMASALESMAN_Mike Johnson'
    ]
    for col in necessary_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Convert datetime columns to numerical representation (days since 2023-01-01)
    reference_date = pd.Timestamp('2023-01-01')
    df_encoded['TGL_INVC'] = (pd.to_datetime(df_encoded['TGL_INVC']) - reference_date).dt.days
    df_encoded['EXP_DATE'] = (pd.to_datetime(df_encoded['EXP_DATE']) - reference_date).dt.days
    
    # Split data into X (features) and y (target)
    X = df_encoded.drop(['TRS_TYPE'], axis=1)
    y = df_encoded['TRS_TYPE']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, pos_label='SALES')
    precision = precision_score(y_test, y_pred, pos_label='SALES')
    f1 = f1_score(y_test, y_pred, pos_label='SALES')
    
    metrics = {
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1
    }
    
    # Create DataFrame for metrics
    metrics_df = pd.DataFrame({
        'Metrics': ['Akurasi', 'Recall', 'Presisi', 'F1 Score'],
        'Value': [accuracy * 100, recall * 100, precision * 100, f1 * 100]
    })
    return model, X_train.columns, metrics  # Return model, columns used for training, and metrics

# Function to predict sales and returns
def predict_sales_and_returns(model, input_data, columns_fit):
    # Ensure input_data has the same columns as during training
    for col in columns_fit:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Remove any extra columns not seen during training
    input_data = input_data[columns_fit]
    
    # Convert datetime columns to numerical representation (days since 2023-01-01)
    reference_date = pd.Timestamp('2023-01-01')
    input_data['TGL_INVC'] = (pd.to_datetime(input_data['TGL_INVC']) - reference_date).dt.days
    input_data['EXP_DATE'] = (pd.to_datetime(input_data['EXP_DATE']) - reference_date).dt.days
    
    # Perform prediction using the trained model
    predictions = model.predict(input_data)
    
    return predictions

# Function to calculate total sales and returns
def calculate_total_sales_returns(df, nama_salesman=None, nama_barang=None):
    if nama_salesman:
        sales_data = df[(df['NAMASALESMAN'] == nama_salesman) & (df['TRS_TYPE'] == 'SALES')]
        retur_data = df[(df['NAMASALESMAN'] == nama_salesman) & (df['TRS_TYPE'] == 'RETUR')]
    elif nama_barang:
        sales_data = df[(df['NAMABARANG'] == nama_barang) & (df['TRS_TYPE'] == 'SALES')]
        retur_data = df[(df['NAMABARANG'] == nama_barang) & (df['TRS_TYPE'] == 'RETUR')]
    else:
        sales_data = df[df['TRS_TYPE'] == 'SALES']
        retur_data = df[df['TRS_TYPE'] == 'RETUR']
    
    total_sales = sales_data['QTYSALES'].sum() if not sales_data.empty else 0
    total_retur = retur_data['QTYSALES'].sum() if not retur_data.empty else 0
    total_sales_value = sales_data['VALUEHNA'].sum() if not sales_data.empty else 0
    total_retur_value = retur_data['VALUEHNA'].sum() if not retur_data.empty else 0
    
    return total_sales, total_retur, total_sales_value, total_retur_value

# Streamlit application
st.title('Prediksi Penjualan dan Retur di Apotek')

# Load data from Excel
@st.cache_data  # Cache data loading for better performance
def load_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    return df

# Input form
st.sidebar.header('Masukkan Parameter Prediksi')
file_path = st.sidebar.file_uploader('Upload File Excel', type=['xlsx'])
if file_path is not None:
    df = load_data(file_path)
    
    # Display all loaded data
    st.subheader('Data yang Dimuat')
    st.write(df)

    # Preprocess and train model using loaded data
    model, columns_fit, metrics = preprocess_data_and_train_model(df)
    
    # Define bulan_tahun_prediksi here
    bulan_tahun_options = ['Semua Bulan dan Tahun'] + pd.to_datetime(df['TGL_INVC']).dt.to_period('M').astype(str).unique().tolist()
    bulan_tahun_prediksi = st.sidebar.selectbox('Bulan dan Tahun Prediksi', bulan_tahun_options)
    nama_barang = st.sidebar.selectbox('Nama Barang', [''] + list(df['NAMABARANG'].unique()))
    nama_salesman = st.sidebar.selectbox('Nama Salesman', [''] + list(df['NAMASALESMAN'].unique()))
    
    # Filter data based on input
    if bulan_tahun_prediksi == 'Semua Bulan dan Tahun':
        filtered_data = df[
            (df['NAMABARANG'].isin([nama_barang]) if nama_barang else df['NAMABARANG'].notna()) &
            (df['NAMASALESMAN'].isin([nama_salesman]) if nama_salesman else df['NAMASALESMAN'].notna())
        ]
    else:
        bulan_tahun_prediksi = pd.Period(bulan_tahun_prediksi, freq='M')
        filtered_data = df[
            (pd.to_datetime(df['TGL_INVC']).dt.to_period('M') == bulan_tahun_prediksi) &
            (df['NAMABARANG'].isin([nama_barang]) if nama_barang else df['NAMABARANG'].notna()) &
            (df['NAMASALESMAN'].isin([nama_salesman]) if nama_salesman else df['NAMASALESMAN'].notna())
        ]

    if not filtered_data.empty:
        # Calculate total sales and returns
        total_sales, total_retur, total_sales_value, total_retur_value = calculate_total_sales_returns(filtered_data, nama_salesman, nama_barang)

        # Initialize empty list to store result data
        result_data = []

        # Iterate over unique outlets
        for outlet in filtered_data['NAMAOUTLET'].unique():
            outlet_data = filtered_data[filtered_data['NAMAOUTLET'] == outlet]

            # Calculate total sales and returns for the current outlet
            total_sales_outlet = outlet_data[outlet_data['TRS_TYPE'] == 'SALES']['QTYSALES'].sum()
            total_retur_outlet = outlet_data[outlet_data['TRS_TYPE'] == 'RETUR']['QTYSALES'].sum()

            # Calculate the return percentage
            if total_sales_outlet != 0:
                retur_percentage = (total_retur_outlet / total_sales_outlet) * 100
            else:
                retur_percentage = 0

                        # Determine outlet health status
            if retur_percentage > 5:
                outlet_status = "Tidak Sehat"
            else:
                outlet_status = "Sehat"

            # Append data for the current outlet to result_data list if outlet is "Sehat"
            if outlet_status == "Sehat":
                result_data.append({
                    'Nama Outlet': outlet,
                    # 'Persentase Retur (%)': retur_percentage,
                    'Status Outlet': outlet_status
                })

        # Create a DataFrame from result_data list
        df_result = pd.DataFrame(result_data)

        # Display total sales and returns with badge style
        st.subheader('Total Sales dan Retur')
        if nama_salesman:
            st.success(f'Total Sales oleh {nama_salesman}: {total_sales} Btl')
            st.error(f'Total Retur oleh {nama_salesman}: {total_retur} Btl')
            st.info(f'Nilai Total Sales oleh {nama_salesman}: Rp. {total_sales_value:,.2f}')
            st.info(f'Nilai Total Retur oleh {nama_salesman}: Rp. {total_retur_value:,.2f}')
        elif nama_barang:
            st.success(f'Total Sales untuk Barang {nama_barang}: {total_sales} Btl')
            st.error(f'Total Retur untuk Barang {nama_barang}: {total_retur} Btl')
            st.info(f'Total Sales untuk Barang {nama_barang}: Rp. {total_sales_value:,.2f}')
            st.info(f'Total Retur untuk Barang {nama_barang}: Rp. {total_retur_value:,.2f}')
        else:
            st.success(f"Nilai Total Sales: {total_sales}")
            st.error(f"Nilai Total Retur: {total_retur}")

        # Display table for healthy outlets
        st.subheader('Status Outlet')
        df_result_styled = df_result.style.applymap(lambda status: 'background-color : green' if status == 'Sehat' else 'background-color : red', subset=['Status Outlet'])
        st.write(df_result_styled)

        # Display top salesman and top outlet
        if nama_salesman:
            st.subheader(f'Top Barang untuk {nama_salesman}')
            top_barang = filtered_data.groupby('NAMABARANG')['QTYSALES'].sum().reset_index().sort_values(by='QTYSALES', ascending=False).head(5)
            top_barang = top_barang.rename(columns={'NAMABARANG': 'Nama Barang', 'QTYSALES': 'Jumlah Produk'})
            st.write(top_barang)

        if nama_barang:
            st.subheader(f'Top Salesman untuk Barang {nama_barang}')
            top_salesman_barang = filtered_data.groupby('NAMASALESMAN')['QTYSALES'].sum().reset_index().sort_values(by='QTYSALES', ascending=False)
            top_salesman_barang = top_salesman_barang.rename(columns={'NAMASALESMAN': 'Nama Salesman', 'QTYSALES': 'Jumlah Produk'})
            st.write(top_salesman_barang)

        # Display top 1-5 sales by outlet
        st.subheader('Top 1-5 Sales by Outlet')
        top_sales_outlet = filtered_data[filtered_data['TRS_TYPE'] == 'SALES'].groupby('NAMAOUTLET')['QTYSALES'].agg(['sum', 'mean']).reset_index()
        top_sales_outlet = top_sales_outlet.rename(columns={'NAMAOUTLET': 'Nama Outlet', 'sum': 'Jumlah Produk Sales', 'mean': 'Rata-rata Sales'}).sort_values(by='Jumlah Produk Sales', ascending=False).head(5)
        top_sales_outlet['Rata-rata Sales'] = top_sales_outlet['Rata-rata Sales'].round(0).astype(int)
        st.write(top_sales_outlet)

        # Display top 1-5 retur by outlet
        st.subheader('Top 1-5 Retur by Outlet')
        top_retur_outlet = filtered_data[filtered_data['TRS_TYPE'] == 'RETUR'].groupby('NAMAOUTLET')['QTYSALES'].agg(['sum', 'mean']).reset_index()
        top_retur_outlet = top_retur_outlet.rename(columns={'NAMAOUTLET': 'Nama Outlet', 'sum': 'Jumlah Produk Retur', 'mean': 'Rata-rata Retur'}).sort_values(by='Jumlah Produk Retur').head(5)
        top_retur_outlet['Rata-rata Retur'] = top_retur_outlet['Rata-rata Retur'].round(0).astype(int)
        st.write(top_retur_outlet)

        # Display trends
        st.subheader('Grafik Tren Sales dan Retur')
        sales_chart = alt.Chart(filtered_data[filtered_data['TRS_TYPE'] == 'SALES']).mark_line().encode(
            x='TGL_INVC',
            y='QTYSALES',
            color=alt.value('green')
        ).properties(
            title='Tren Penjualan'
        )

        retur_chart = alt.Chart(filtered_data[filtered_data['TRS_TYPE'] == 'RETUR']).mark_line().encode(
            x='TGL_INVC',
            y='QTYSALES',
            color=alt.value('red')
        ).properties(
            title='Tren Retur'
        )

        # Display Altair charts using st.altair_chart
        st.altair_chart(sales_chart, use_container_width=True)
        st.altair_chart(retur_chart, use_container_width=True)

        # Display metrics as line chart with markers
        # Display metrics as line chart with markers
        st.subheader('Metrik Model')
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])
        metrics_df.reset_index(inplace=True)
        metrics_df = metrics_df.rename(columns={'index': 'Metric'})

        # Plotly Line Chart with Markers
        fig = px.line(metrics_df, x='Metric', y='Score', title='Metrik Model: Akurasi, Recall, Presisi, F1 Score',
                    labels={'Metric': 'Metric', 'Score': 'Score (%)'}, markers=True)

        # Add annotations to the chart
        for i, row in metrics_df.iterrows():
            fig.add_annotation(
                x=row['Metric'],      # X coordinate based on metric
                y=row['Score'],       # Y coordinate based on value
                text=f'{row["Score"]:0f}%',  # Text to display (format the text correctly)
                font=dict(size=10),   # Font size
                yshift=10             # Vertical text position shift
            )

        st.plotly_chart(fig, use_container_width=True)  # Display the chart


    else:
        st.error('Tidak ada data yang cocok dengan kriteria yang dimasukkan.')

else:
    st.sidebar.text('Mohon upload file Excel untuk memulai prediksi.')
