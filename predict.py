import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
from sklearn.tree import export_graphviz
import graphviz
import altair as alt
import io
import base64

# Judul aplikasi
st.title("Aplikasi Prediksi Penjualan dengan Random Forest")

# Unggah dataset
uploaded_file = st.sidebar.file_uploader("Unggah file dataset Anda", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Membaca dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Tampilkan dataset:")
    st.write(df)

    # Preprocessing data
    df['TGL_INVC'] = pd.to_datetime(df['TGL_INVC'], format='%d/%m/%Y')
    df['Bulan'] = df['TGL_INVC'].dt.month
    df['Tahun'] = df['TGL_INVC'].dt.year

    # Menggunakan Label Encoder untuk mengubah kolom kategorikal menjadi numerik
    le_trs_type = LabelEncoder()
    df['TRS_TYPE'] = le_trs_type.fit_transform(df['TRS_TYPE'])

    le_namaoutlet = LabelEncoder()
    df['NAMAOUTLET'] = le_namaoutlet.fit_transform(df['NAMAOUTLET'])

    le_namabarang = LabelEncoder()
    df['NAMABARANG'] = le_namabarang.fit_transform(df['NAMABARANG'])

    le_namasalesman = LabelEncoder()
    df['NAMASALESMAN'] = le_namasalesman.fit_transform(df['NAMASALESMAN'])

    # Menambah fitur interaksi dan fitur temporal baru
    df['Outlet_Barang'] = df['NAMAOUTLET'] * df['NAMABARANG']
    df['Tahun_Bulan'] = df['Tahun'] * df['Bulan']
    df['Relative_Year'] = df['Tahun'] - df['Tahun'].min()  # Fitur tambahan untuk tahun relatif
    
    # Tambahkan fitur perubahan YoY, atasi NaN dan infinity
    df['Perubahan_YoY'] = df.groupby('NAMAOUTLET')['QTYSALES'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

    # Tambahkan fitur moving average
    df['Moving_Avg'] = df.groupby('NAMAOUTLET')['QTYSALES'].rolling(window=3).mean().reset_index(0, drop=True).fillna(0)

    # Fitur dan label (QTYSALES sebagai target)
    X = df[['TRS_TYPE', 'NAMAOUTLET', 'NAMABARANG', 'Bulan', 'Tahun', 'Outlet_Barang', 'Tahun_Bulan', 'Relative_Year', 'Perubahan_YoY', 'Moving_Avg', 'NAMASALESMAN']]
    y = df['QTYSALES']

    # Normalisasi fitur
    # scaler = StandardScaler()
    # X[['Tahun', 'Bulan']] = scaler.fit_transform(X[['Tahun', 'Bulan']])

    # Cek apakah ada nilai NaN atau infinity dalam dataset
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Ganti infinity dengan NaN
    df.fillna(0, inplace=True)  # Isi NaN dengan 0
    # Split data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Membuat model Random Forest dengan hyperparameter tuning
    model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)

    # Form untuk prediksi di sidebar
    st.sidebar.header("Form Prediksi")

    trs_type = st.sidebar.selectbox("Tipe Transaksi", le_trs_type.classes_)
    namaoutlet = st.sidebar.selectbox("Nama Outlet", le_namaoutlet.classes_)
    namabarang = st.sidebar.selectbox("Nama Barang", le_namabarang.classes_)
    namasalesman = st.sidebar.selectbox("Nama Salesman", le_namasalesman.classes_)
    bulan = st.sidebar.selectbox("Bulan", list(range(1, 13)))
    tahun = st.sidebar.number_input("Tahun", min_value=2020, max_value=2100, value=2024)  # Tahun manual

    # Transform input ke bentuk numerik
    trs_type_encoded = le_trs_type.transform([trs_type])[0]
    namaoutlet_encoded = le_namaoutlet.transform([namaoutlet])[0]
    namabarang_encoded = le_namabarang.transform([namabarang])[0]
    namasalesman_encoded = le_namasalesman.transform([namasalesman])[0]

    outlet_barang = namaoutlet_encoded * namabarang_encoded
    tahun_bulan = tahun * bulan
    relative_year = tahun - df['Tahun'].min()  # Tahun relatif

    # Normalisasi fitur input
    # input_features = pd.DataFrame([[tahun, bulan]], columns=['Tahun', 'Bulan'])
    # input_features[['Tahun', 'Bulan']] = scaler.transform(input_features[['Tahun', 'Bulan']])

    # tahun_norm = input_features['Tahun'].values[0]
    # bulan_norm = input_features['Bulan'].values[0]

    # Prediksi
    if st.sidebar.button("Prediksi"):
        input_data = [[trs_type_encoded, namaoutlet_encoded, namabarang_encoded, bulan, tahun, outlet_barang, tahun_bulan, relative_year, 0, 0, namasalesman_encoded]]
        input_data = pd.DataFrame(input_data, columns=X.columns)  # Pastikan input data memiliki kolom yang sama

        # input_data = [[trs_type_encoded, namaoutlet_encoded, namabarang_encoded, bulan_norm, tahun_norm, outlet_barang, tahun_bulan, namasalesman_encoded]]
        prediksi_total = model.predict(input_data)[0]

        # Menghitung persentase retur dan menentukan status outlet
        total_retur = df[(df['TRS_TYPE'] == le_trs_type.transform(['RETUR'])[0]) & (df['NAMAOUTLET'] == namaoutlet_encoded)]['QTYSALES'].sum()
        total_sales = df[(df['TRS_TYPE'] == le_trs_type.transform(['SALES'])[0]) & (df['NAMAOUTLET'] == namaoutlet_encoded)]['QTYSALES'].sum()
        persentase_retur = (total_retur / total_sales) * 100 if total_sales > 0 else 0
        status_outlet = "Sehat" if persentase_retur < 5 else "Tidak Sehat"

        # Menampilkan hasil prediksi
        st.header("Hasil Prediksi")
        st.success(f"**Prediksi total {le_trs_type.inverse_transform([trs_type_encoded])[0]} {le_namabarang.inverse_transform([namabarang_encoded])[0]} di Outlet {le_namaoutlet.inverse_transform([namaoutlet_encoded])[0]} pada bulan {bulan} dan tahun {tahun} untuk tipe transaksi {le_trs_type.inverse_transform([trs_type_encoded])[0]} oleh Salesman {le_namasalesman.inverse_transform([namasalesman_encoded])[0]} adalah: {int(prediksi_total)} Btl**")

        # Tampilkan tabel status outlet sesuai bulan yang dipilih
        filtered_data = df[df['Bulan'] == bulan]
        result_data = []

        # Iterasi untuk setiap outlet di bulan yang dipilih
        for outlet in filtered_data['NAMAOUTLET'].unique():
            outlet_data = filtered_data[filtered_data['NAMAOUTLET'] == outlet]

            # Hitung total sales dan returns di bulan tersebut
            total_sales = outlet_data[outlet_data['TRS_TYPE'] == le_trs_type.transform(['SALES'])[0]]['QTYSALES'].sum()
            total_returns = outlet_data[outlet_data['TRS_TYPE'] == le_trs_type.transform(['RETUR'])[0]]['QTYSALES'].sum()

            # Jika tidak ada penjualan, anggap outlet "Tidak Sehat" karena tidak ada penjualan
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

        # Create a DataFrame from result_data list
        df_result = pd.DataFrame(result_data)

        # Fungsi untuk memberi warna pada status outlet
        def color_status(val):
            color = 'green' if val == "Sehat" else 'red'
            return f'background-color: {color}'

        st.header("Tabel Status Outlet")
        st.dataframe(df_result.style.applymap(color_status, subset=['Status Outlet']))

        # Menambahkan tabel total dan rata-rata penjualan per Salesman
        salesman_data = filtered_data.groupby('NAMASALESMAN')['QTYSALES'].agg(['sum', 'mean']).reset_index()
        salesman_data['Nama Salesman'] = le_namasalesman.inverse_transform(salesman_data['NAMASALESMAN'])
        salesman_data.rename(columns={'sum': 'Total Penjualan', 'mean': 'Rata-rata Penjualan'}, inplace=True)
        salesman_data = salesman_data[['Nama Salesman', 'Total Penjualan', 'Rata-rata Penjualan']]

        st.header("Tabel Total dan Rata-rata Penjualan per Salesman")
        st.write(salesman_data)

        # Grafik Tren berdasarkan Tipe Transaksi
        if trs_type in ['SALES', 'RETUR']:
            st.subheader(f'Hasil Penjumlahan dan Rata-rata QTYSALES berdasarkan Nama Outlet dan Tipe Transaksi {trs_type}')
            
            # Pastikan filtering TRS_TYPE menggunakan string asli sebelum encoding
            trs_type_encoded = le_trs_type.transform([trs_type])[0]
            
            # Kelompokkan data berdasarkan NAMAOUTLET dan hitung sum dan mean QTYSALES
            grouped_data = df[df['TRS_TYPE'] == trs_type_encoded].groupby('NAMAOUTLET')['QTYSALES'].agg(['sum', 'mean']).reset_index()

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

            # --- Grafik Tren berdasarkan tipe transaksi ---
            st.subheader(f'Tren {trs_type} untuk Semua Outlet yang Terdaftar')

            # Filter trend_data untuk semua outlet yang ada di grouped_data
            trend_data = df[df['TRS_TYPE'] == trs_type_encoded]
            trend_data['NAMAOUTLET'] = le_namaoutlet.inverse_transform(trend_data['NAMAOUTLET'])

            # Filter trend_data hanya untuk outlet yang ada di tabel hasil (grouped_data)
            filtered_trend_data = trend_data[trend_data['NAMAOUTLET'].isin(grouped_data['Nama Outlet'])]

            # Buat grafik tren
            trend_chart = alt.Chart(filtered_trend_data).mark_line().encode(
                x='TGL_INVC:T',
                y='QTYSALES:Q',
                color='NAMAOUTLET:N'  # Warna berbeda untuk tiap outlet
            ).properties(
                title=f'Tren {trs_type} per Outlet (Seluruh Outlet Terdaftar)'
            )

            # Tampilkan grafik
            st.altair_chart(trend_chart, use_container_width=True)


        # Grafik Metrik Evaluasi
        # Membuat dummy data untuk metrik evaluasi, karena RandomForestRegressor tidak menghasilkan metrik ini
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > y_test.median()).astype(int)
        y_test_binary = (y_test > y_test.median()).astype(int)

        # Menghitung metrik evaluasi
        recall = int(recall_score(y_test_binary, y_pred_binary) * 100)
        accuracy = int(accuracy_score(y_test_binary, y_pred_binary) * 100)
        precision = int(precision_score(y_test_binary, y_pred_binary) * 100)
        f1 = int(f1_score(y_test_binary, y_pred_binary) * 100)

        metrics = {
            'Recall': recall,
            'Accuracy': accuracy,
            'Precision': precision,
            'F1 Score': f1
        }

        # Membuat grafik menggunakan Plotly
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metrics', 'Value'])
        fig = px.line(metrics_df, x='Metrics', y='Value', title='Metrik Model: Akurasi, Recall, Presisi, F1 Score',
                      labels={'Metrics': 'Metric', 'Value': 'Score (%)'}, markers=True)

        # Menambahkan anotasi ke grafik
        for i, row in metrics_df.iterrows():
            fig.add_annotation(
                x=row['Metrics'],
                y=row['Value'],
                text=f'{row["Value"]}%',
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