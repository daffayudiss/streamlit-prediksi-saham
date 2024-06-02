import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime

# Fungsi untuk memprediksi harga di masa depan
def prediksi_harga_masa_depan(ticker, tanggal_akhir, hari=5):
    data = yf.download(ticker, start='2024-04-25', end=tanggal_akhir)
    if data.empty:
        st.error("Data tidak tersedia untuk simbol yang ditentukan dan rentang tanggal. Harap masukkan simbol yang valid.")
        return None
    data['Tutup Sebelumnya'] = data['Close'].shift(1)
    data = data.dropna()
    if data.empty:
        st.error("Data yang cukup tidak tersedia untuk prediksi. Harap pilih rentang tanggal yang berbeda.")
        return None
    X = data[['Tutup Sebelumnya']]
    y = data['Close']
    model = LinearRegression()
    model.fit(X, y)
    penutup_terakhir = data['Close'].iloc[-1]
    tanggal_masa_depan = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=hari, freq='B')
    prediksi = []
    for _ in range(hari):
        prediksi_selanjutnya = model.predict([[penutup_terakhir]])[0]
        prediksi.append(prediksi_selanjutnya)
        penutup_terakhir = prediksi_selanjutnya
    prediksi_masa_depan = pd.DataFrame(data={'Tanggal': tanggal_masa_depan, 'Tutupan Diprediksi': prediksi})
    prediksi_masa_depan.set_index('Tanggal', inplace=True)
    return prediksi_masa_depan

# Aplikasi Streamlit
def utama():
    st.set_page_config(page_title='Aplikasi Prediksi Harga Saham', page_icon="📈")
    st.write("""
    # Aplikasi Prediksi Harga Saham
    Aplikasi ini memprediksi harga penutupan saham untuk 5 hari ke depan menggunakan Regresi Linier.
    """)

    # Sidebar
    st.sidebar.header('Parameter Input Pengguna')
    simbol = st.sidebar.text_input("Masukkan Simbol Saham", 'DCII.JK')
    hari = st.sidebar.slider("Jumlah Hari untuk Prediksi", min_value=1, max_value=30, value=5)
    tanggal_akhir = st.sidebar.date_input("Tanggal Akhir untuk Pengambilan Data", datetime.today())

    # Mendapatkan prediksi
    prediksi = prediksi_harga_masa_depan(simbol, tanggal_akhir, hari)
    if prediksi is None:
        return

    # Menampilkan prediksi
    st.subheader(f'Harga Diprediksi untuk {hari} Hari ke Depan')
    st.write(prediksi)

    # Plot harga aktual dan diprediksi
    st.subheader('Harga Aktual vs. Diprediksi')
    data = yf.download(simbol, start='2024-04-25', end=tanggal_akhir.strftime('%Y-%m-%d'))
    if data.empty:
        st.error("Data tidak tersedia untuk simbol yang ditentukan dan rentang tanggal. Harap masukkan simbol yang valid.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot harga aktual dan diprediksi
    ax.plot(data.index, data['Close'], label='Harga Aktual', color='blue')
    ax.plot(prediksi.index, prediksi['Tutupan Diprediksi'], label=f'Harga Diprediksi ({hari} hari)', linestyle='--', color='green')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga Penutupan')
    ax.set_title(f'Prediksi Harga Saham {simbol} untuk {hari} Hari ke Depan menggunakan Regresi Linier')
    ax.legend(loc='upper left')

    # Menampilkan plot
    st.pyplot(fig)

    # Menampilkan statistik tambahan
    st.subheader('Statistik Tambahan')
    st.write("Rata-rata Harga Aktual:", data['Close'].mean())
    st.write("Rata-rata Harga Diprediksi:", prediksi['Tutupan Diprediksi'].mean())

if __name__ == '__main__':
    utama()
