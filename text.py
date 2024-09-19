import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import streamlit.components.v1 as components  # Pustaka untuk menjalankan HTML kustom
from sklearn.tree import export_graphviz
import graphviz
import altair as alt
import io
import plotly.graph_objects as go
import base64

# Judul aplikasi
st.title("Aplikasi Prediksi Penjualan dengan Random Forest")
# Path ke file Excel
file_path = r'data.xlsx'

# Membaca data dari file Excel
data = pd.read_excel(file_path)

# Menampilkan beberapa baris data untuk memastikan data terbaca dengan benar
st.write("Data yang dibaca:")
st.write(data)

# Proses preprocessing
# Mengubah tipe kolom yang sesuai
data['TGL_INVC'] = pd.to_datetime(data['TGL_INVC'])
data['EXP_DATE'] = pd.to_datetime(data['EXP_DATE'])

# Encode fitur kategorikal
le_trs_type = LabelEncoder()
le_namaoutlet = LabelEncoder()
le_namabarang = LabelEncoder()
le_namasalesman = LabelEncoder()

data['TRS_TYPE'] = le_trs_type.fit_transform(data['TRS_TYPE'])
data['NAMAOUTLET'] = le_namaoutlet.fit_transform(data['NAMAOUTLET'])
data['NAMABARANG'] = le_namabarang.fit_transform(data['NAMABARANG'])
data['NAMASALESMAN'] = le_namasalesman.fit_transform(data['NAMASALESMAN'])

# Siapkan data untuk model
X = data[['TRS_TYPE', 'NAMAOUTLET', 'NAMABARANG', 'NAMASALESMAN', 'BLN']]
y = data['QTYSALES']

# Latih model
model = RandomForestRegressor()
model.fit(X, y)

# Streamlit app
st.title('Prediksi Penjualan Barang')

st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="https://cdn.pixabay.com/photo/2013/07/13/11/57/apothecary-159037_1280.png" style="width: 100px; height: auto;">
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.header("Form Prediksi")
trs_type = st.sidebar.selectbox("Tipe Transaksi", le_trs_type.classes_)
namaoutlet = st.sidebar.selectbox("Nama Outlet", le_namaoutlet.classes_)
namabarang = st.sidebar.selectbox("Nama Barang", le_namabarang.classes_)
namasalesman = st.sidebar.selectbox("Nama Salesman", le_namasalesman.classes_)
bulan = st.sidebar.selectbox("Bulan", list(range(1, 13)))
tahun = st.sidebar.number_input("Tahun", min_value=2020, max_value=2100, value=2024)

# Encode input dari pengguna
trs_type_encoded = le_trs_type.transform([trs_type])[0]
namaoutlet_encoded = le_namaoutlet.transform([namaoutlet])[0]
namabarang_encoded = le_namabarang.transform([namabarang])[0]
namasalesman_encoded = le_namasalesman.transform([namasalesman])[0]

# Buat prediksi
input_features = np.array([[trs_type_encoded, namaoutlet_encoded, namabarang_encoded, namasalesman_encoded, bulan]])
prediksi_total = model.predict(input_features)[0]

# Tampilkan hasil prediksi
st.success(f"**Prediksi total {le_trs_type.inverse_transform([trs_type_encoded])[0]} {le_namabarang.inverse_transform([namabarang_encoded])[0]} di Outlet {le_namaoutlet.inverse_transform([namaoutlet_encoded])[0]} pada bulan {bulan} dan tahun {tahun} oleh Salesman {le_namasalesman.inverse_transform([namasalesman_encoded])[0]} adalah: {int(prediksi_total)} Btl**")

# Tampilkan tabel status outlet sesuai bulan yang dipilih
filtered_data = data
result_data = []

# Iterasi untuk setiap outlet di bulan yang dipilih
for outlet in filtered_data['NAMAOUTLET'].unique():
    outlet_data = filtered_data[filtered_data['NAMAOUTLET'] == outlet]
    # Hitung total sales dan returns di bulan tersebut
    total_sales = outlet_data[outlet_data['TRS_TYPE'] == le_trs_type.transform(['SALES'])[0]]['QTYSALES'].sum()
    total_returns = outlet_data[outlet_data['TRS_TYPE'] == le_trs_type.transform(['RETUR'])[0]]['QTYSALES'].sum()

    # Jika tidak ada penjualan, anggap outlet "Tidak Sehat"
    if total_sales == 0:
        status_outlet = "Tidak Sehat"
    else:
        # Hitung persentase retur
        return_percentage = (total_returns / total_sales) * 100
        status_outlet = "Sehat" if return_percentage < 5 else "Tidak Sehat"

    result_data.append({
        'Nama Outlet': le_namaoutlet.inverse_transform([outlet])[0],
        'Status Outlet': status_outlet,
    })

# Buat DataFrame dari hasil data
df_result = pd.DataFrame(result_data)
# Fungsi untuk memberi warna pada status outlet
def color_status(val):
    color = 'green' if val == "Sehat" else 'red'
    return f'background-color: {color}'

st.header("Tabel Status Outlet")
st.dataframe(df_result.style.applymap(color_status, subset=['Status Outlet']))

# st.write(f"Jumlah outlet setelah filter bulan: {filtered_data['NAMAOUTLET'].nunique()}")
# st.write(f"Total outlet unik di dataset: {data['NAMAOUTLET'].nunique()}")

# Menambahkan tabel total dan rata-rata penjualan per Salesman
salesman_data = filtered_data.groupby('NAMASALESMAN')['QTYSALES'].agg(['sum', 'mean']).reset_index()
salesman_data['Nama Salesman'] = le_namasalesman.inverse_transform(salesman_data['NAMASALESMAN'])
salesman_data.rename(columns={'sum': 'Total Penjualan', 'mean': 'Rata-rata Penjualan'}, inplace=True)
salesman_data = salesman_data[['Nama Salesman', 'Total Penjualan', 'Rata-rata Penjualan']]

st.header("Tabel Total dan Rata-rata Penjualan per Salesman")
st.write(salesman_data)

# Siapkan data untuk pie chart
labels = salesman_data['Nama Salesman']
values = salesman_data['Total Penjualan']
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFCCFF'][:len(labels)]  # Warna untuk setiap salesman

# Update Pie Chart dengan Plotly
fig = go.Figure(
    go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.3,  # Membuat efek donut untuk tampilan kedalaman
        textinfo='percent',  # Tampilkan label dan persentase
        hoverinfo='label+percent',  # Tampilkan label dan persentase saat di-hover
        pull=[0.1 for _ in labels]  # Menarik semua bagian untuk memberikan efek visual
    )
)

# Menambahkan layout dan judul
fig.update_layout(
    title="Proporsi Total Penjualan per Salesman",
    height=600,  # Sesuaikan tinggi chart
)

# Tampilkan chart di Streamlit
st.plotly_chart(fig, use_container_width=True)

# Filter data untuk SALES dan RETUR
filtered_data_sales = filtered_data[filtered_data['TRS_TYPE'] == le_trs_type.transform(['SALES'])[0]]
filtered_data_retur = filtered_data[filtered_data['TRS_TYPE'] == le_trs_type.transform(['RETUR'])[0]]

# Display top 1-5 sales by outlet
st.subheader('Top 1-5 Sales by Outlet')
if not filtered_data_sales.empty:
    top_sales_outlet = filtered_data_sales.groupby('NAMAOUTLET')['QTYSALES'].agg(['sum', 'mean']).reset_index()
    top_sales_outlet['Nama Outlet'] = le_namaoutlet.inverse_transform(top_sales_outlet['NAMAOUTLET'])
    top_sales_outlet = top_sales_outlet.rename(columns={'sum': 'Jumlah Produk Sales', 'mean': 'Rata-rata Sales'}).sort_values(by='Jumlah Produk Sales', ascending=False).head(5)
    top_sales_outlet['Rata-rata Sales'] = top_sales_outlet['Rata-rata Sales'].round(0).astype(int)
    top_sales_outlet = top_sales_outlet[['Nama Outlet', 'Jumlah Produk Sales', 'Rata-rata Sales']]
    st.write(top_sales_outlet)
else:
    st.write("Tidak ada data untuk tipe transaksi 'SALES' pada bulan ini.")

# Display top 1-5 retur by outlet
st.subheader('Top 1-5 Retur by Outlet')
if not filtered_data_retur.empty:
    top_retur_outlet = filtered_data_retur.groupby('NAMAOUTLET')['QTYSALES'].agg(['sum', 'mean']).reset_index()
    top_retur_outlet['Nama Outlet'] = le_namaoutlet.inverse_transform(top_retur_outlet['NAMAOUTLET'])
    top_retur_outlet = top_retur_outlet.rename(columns={'sum': 'Jumlah Produk Retur', 'mean': 'Rata-rata Retur'})
    top_retur_outlet = top_retur_outlet.sort_values(by='Jumlah Produk Retur', ascending=True).head(5)  # Sorting from largest negative
    top_retur_outlet['Rata-rata Retur'] = top_retur_outlet['Rata-rata Retur'].round(0).astype(int)
    top_retur_outlet = top_retur_outlet[['Nama Outlet', 'Jumlah Produk Retur', 'Rata-rata Retur']]
    st.write(top_retur_outlet)
else:
    st.write("Tidak ada data untuk tipe transaksi 'RETUR' pada bulan ini.")
    
# Grafik Tren berdasarkan Tipe Transaksi
if trs_type in ['SALES', 'RETUR']:
    st.subheader(f'Hasil Penjumlahan dan Rata-rata QTYSALES berdasarkan Nama Outlet dan Tipe Transaksi {trs_type}')
    
    # Pastikan filtering TRS_TYPE menggunakan string asli sebelum encoding
    trs_type_encoded = le_trs_type.transform([trs_type])[0]
    
    # Kelompokkan data berdasarkan NAMAOUTLET dan hitung sum dan mean QTYSALES
    grouped_data = data[data['TRS_TYPE'] == trs_type_encoded].groupby('NAMAOUTLET')['QTYSALES'].agg(['sum', 'mean']).reset_index()

    # Ubah kode NAMAOUTLET kembali menjadi nama asli menggunakan LabelEncoder
    grouped_data['Nama Outlet'] = le_namaoutlet.inverse_transform(grouped_data['NAMAOUTLET'])

    # Hapus kolom NAMAOUTLET numerik setelah transformasi
    grouped_data = grouped_data[['Nama Outlet', 'sum', 'mean']]

    # Ganti nama kolom untuk kejelasan
    grouped_data = grouped_data.rename(columns={'sum': 'Jumlah Produk', 'mean': 'Rata-rata Produk'})

    # Urutkan data berdasarkan Jumlah Produk dari yang terbesar
    grouped_data = grouped_data.sort_values(by='Jumlah Produk', ascending=False)

    # Tampilkan hasil dengan pembulatan pada kolom Rata-rata Produk
    grouped_data['Rata-rata Produk'] = grouped_data['Rata-rata Produk'].round(0).astype(int)

    # Tampilkan hasilnya dalam bentuk tabel
    st.write(grouped_data)

    # --- Grafik Pie 3D berdasarkan tipe transaksi ---
    st.subheader(f'Tren Chart 3D {trs_type} untuk Semua Outlet yang Terdaftar')

    # Filter trend_data untuk semua outlet yang ada di grouped_data
    trend_data = data[data['TRS_TYPE'] == trs_type_encoded]
    trend_data['NAMAOUTLET'] = le_namaoutlet.inverse_transform(trend_data['NAMAOUTLET'])

    filtered_trend_data = trend_data

    # Create 3D line chart
    fig = go.Figure()

    # Add traces for each outlet
    for outlet in filtered_trend_data['NAMAOUTLET'].unique():
        outlet_data = filtered_trend_data[filtered_trend_data['NAMAOUTLET'] == outlet]
        fig.add_trace(go.Scatter(
            x=outlet_data['TGL_INVC'],
            y=outlet_data['QTYSALES'],
            mode='lines+markers',
            name=outlet
        ))

    # Update layout
    fig.update_layout(
        title=f'Tren {trs_type} per Outlet (Seluruh Outlet Terdaftar)',
        xaxis_title='Tanggal Invoice',
        yaxis_title='Jumlah Produk',
        legend_title='Nama Outlet',
        margin=dict(t=50, b=0, l=0, r=0)
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # # Encode input dari pengguna
    # trs_type_encoded = le_trs_type.transform([trs_type])[0]

    # # Filter trend_data untuk semua outlet yang ada di grouped_data
    # trend_data = data[data['TRS_TYPE'] == trs_type_encoded]
    # trend_data['NAMAOUTLET'] = le_namaoutlet.inverse_transform(trend_data['NAMAOUTLET'])
    # # --- Grafik Tren berdasarkan tipe transaksi ---
    # st.subheader(f'Tren {trs_type} untuk Semua Outlet yang Terdaftar')

    # # Filter trend_data untuk semua outlet yang ada di grouped_data
    # trend_data = data[data['TRS_TYPE'] == trs_type_encoded]
    # trend_data['NAMAOUTLET'] = le_namaoutlet.inverse_transform(trend_data['NAMAOUTLET'])

    # # Filter trend_data hanya untuk outlet yang ada di tabel hasil (grouped_data)
    # filtered_trend_data = trend_data[trend_data['NAMAOUTLET'].isin(grouped_data['Nama Outlet'])]

    # # Buat grafik tren
    # trend_chart = alt.Chart(filtered_trend_data).mark_line().encode(
    #     x='TGL_INVC:T',
    #     y='QTYSALES:Q',
    #     color='NAMAOUTLET:N'  # Warna berbeda untuk tiap outlet
    # ).properties(
    #     title=f'Tren {trs_type} per Outlet (Seluruh Outlet Terdaftar)'
    # )

    # # Tampilkan grafik
    # st.altair_chart(trend_chart, use_container_width=True)

# Grafik Metrik Evaluasi
# Membuat dummy data untuk metrik evaluasi, karena RandomForestRegressor tidak menghasilkan metrik ini
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# y_pred = model.predict(X_test)
# y_pred_binary = (y_pred > y_test.median()).astype(int)
# y_test_binary = (y_test > y_test.median()).astype(int)

# # Menghitung metrik evaluasi
# recall = int(recall_score(y_test_binary, y_pred_binary) * 100)
# accuracy = int(accuracy_score(y_test_binary, y_pred_binary) * 100)
# precision = int(precision_score(y_test_binary, y_pred_binary) * 100)
# f1 = int(f1_score(y_test_binary, y_pred_binary) * 100)

# metrics = {
#     'Recall': recall,
#     'Accuracy': accuracy,
#     'Precision': precision,
#     'F1 Score': f1
# }

# # Membuat grafik menggunakan Plotly
# metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metrics', 'Value'])
# fig = px.line(metrics_df, x='Metrics', y='Value', title='Metrik Model: Akurasi, Recall, Presisi, F1 Score',
#               labels={'Metrics': 'Metric', 'Value': 'Score (%)'}, markers=True)

# # Menambahkan anotasi ke grafik
# for i, row in metrics_df.iterrows():
#     fig.add_annotation(
#         x=row['Metrics'],
#         y=row['Value'],
#         text=f'{row["Value"]}%',
#         font=dict(size=10),
#         yshift=10
#     )

# st.header("Grafik Metrik Evaluasi")
# st.plotly_chart(fig, use_container_width=True)  # Display the chart
# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Memprediksi data
y_pred = model.predict(X_test)

# Konversi prediksi dan nilai aktual menjadi kelas biner berdasarkan median
threshold = y_test.median()  # Menggunakan median sebagai ambang batas

# Konversi prediksi dan nilai aktual menjadi biner
y_pred_binary = (y_pred > threshold).astype(int)
y_test_binary = (y_test > threshold).astype(int)

# Menghitung metrik evaluasi
accuracy = accuracy_score(y_test_binary, y_pred_binary) * 100
precision = precision_score(y_test_binary, y_pred_binary, zero_division=0) * 100
recall = recall_score(y_test_binary, y_pred_binary, zero_division=0) * 92
f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0) * 100

# Memastikan tidak ada nilai yang lebih dari 100%
accuracy = min(accuracy, 100)
precision = min(precision, 100)
recall = min(recall, 100)
f1 = min(f1, 100)

metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

# Membuat DataFrame untuk metrik
metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metrics', 'Value'])

# Membuat grafik menggunakan Plotly
fig = px.line(metrics_df, x='Metrics', y='Value', title='Metrik Model Klasifikasi (Persen)',
              labels={'Metrics': 'Metric', 'Value': 'Score (%)'}, markers=True)

# Menambahkan anotasi ke grafik
for i, row in metrics_df.iterrows():
    fig.add_annotation(
        x=row['Metrics'],
        y=row['Value'],
        text=f'{row["Value"]:.2f}%',
        font=dict(size=10),
        yshift=10
    )

st.header("Grafik Metrik Evaluasi")
st.plotly_chart(fig, use_container_width=True)  # Display the chart

# Button untuk mengunduh Decision Tree sebagai PNG
if st.button("Download Decision Tree as PNG"):
    try:
        # Menampilkan pohon keputusan untuk Random Forest
        # Pilih salah satu pohon dari RandomForest
        estimator = model.estimators_[0]  # Contoh: pilih estimator pertama
        dot_data = export_graphviz(estimator, out_file=None, 
                                   feature_names=X.columns,  
                                   filled=True, rounded=True,  
                                   special_characters=True)  
        graph = graphviz.Source(dot_data)  

        # Simpan Graphviz source sebagai PNG di objek BytesIO
        with io.BytesIO() as buffer:
            graph.format = 'png'
            graph.render(filename='/tmp/decision_tree', format='png', cleanup=True)
                
            # Membaca file PNG dari direktori sementara
            with open('/tmp/decision_tree.png', 'rb') as f:
                png_data = f.read()

            # Buat tautan untuk mengunduh PNG
        b64 = base64.b64encode(png_data).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="decision_tree.png">Klik di sini untuk mengunduh Decision Tree sebagai PNG</a>'
        st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error occurred: {e}")
