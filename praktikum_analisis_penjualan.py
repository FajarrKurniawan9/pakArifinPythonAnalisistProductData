# =============================================================================
# TAHAP 2: PERSIAPAN & PEMBERSIHAN DATA (DATA CLEANING)
# =============================================================================

# --- Langkah 2.1: Import semua library yang dibutuhkan ---

import pandas as pd                          # Untuk manipulasi dan analisis data tabular
import matplotlib.pyplot as plt              # Untuk membuat visualisasi/grafik dasar
import matplotlib.ticker as mticker         # Untuk memformat angka pada sumbu grafik
import seaborn as sns                        # Untuk visualisasi statistik yang lebih indah
import datetime as dt                        # Untuk manipulasi tipe data tanggal & waktu
import warnings                              # Untuk menyembunyikan peringatan yang tidak perlu
from sklearn.linear_model import LinearRegression  # Model Regresi Linear dari scikit-learn
from sklearn.model_selection import train_test_split  # Untuk membagi data training & testing
from sklearn.metrics import r2_score         # Untuk mengukur akurasi model (R² Score)

# Menyembunyikan warning yang tidak krusial agar output lebih bersih
warnings.filterwarnings('ignore')

# --- Langkah 2.2: Konfigurasi tampilan global ---

# Mengatur tema visual seaborn agar grafik terlihat profesional
sns.set_theme(style="whitegrid", palette="muted")

# Mengatur ukuran font default pada semua plot matplotlib
plt.rcParams.update({'font.size': 11})

print("=" * 60)
print("  ANALISIS PERFORMA PENJUALAN E-COMMERCE")
print("=" * 60)

# --- Langkah 2.3: Membaca dataset CSV ---

# Tentukan path file CSV Anda di sini
# Jika di Google Colab dan sudah di-upload manual: '/content/data_praktikum_analisis_data.csv'
# Jika di folder yang sama dengan skrip ini: 'data_praktikum_analisis_data.csv'
PATH_FILE = 'data_praktikum_analisis_data.csv'

# pd.read_csv() membaca file CSV dan mengubahnya menjadi DataFrame (tabel data di Python)
df = pd.read_csv(PATH_FILE)

print(f"\n[INFO] Dataset berhasil dimuat!")
print(f"[INFO] Jumlah baris awal  : {df.shape[0]} baris")
print(f"[INFO] Jumlah kolom       : {df.shape[1]} kolom")
print(f"\n--- Preview 5 Baris Pertama ---")
print(df.head())
print(f"\n--- Informasi Tipe Data & Nilai Kosong ---")
print(df.info())
print(f"\n--- Statistik Deskriptif Awal ---")
print(df.describe())

# --- Langkah 2.4: Pembersihan Data ---

print("\n" + "=" * 60)
print("  PROSES PEMBERSIHAN DATA")
print("=" * 60)

# Mencatat jumlah baris sebelum pembersihan untuk laporan
jumlah_sebelum = len(df)

# -- Hapus baris yang nilai Total_Sales-nya kosong (NaN) --
# NaN (Not a Number) adalah nilai kosong/hilang di pandas
# dropna(subset=[...]) hanya menghapus baris di mana kolom yang ditentukan bernilai NaN
df = df.dropna(subset=['Total_Sales'])
print(f"[CLEANING] Baris dengan Total_Sales kosong dihapus.")
print(f"           Sisa setelah dropna: {len(df)} baris")

# -- Hapus anomali: baris dengan Price_Per_Unit <= 0 --
# Harga per unit tidak mungkin nol atau negatif, ini kemungkinan error data
df = df[df['Price_Per_Unit'] > 0]
print(f"[CLEANING] Baris dengan Price_Per_Unit <= 0 dihapus.")
print(f"           Sisa setelah filter harga: {len(df)} baris")

# Menampilkan laporan ringkas hasil pembersihan
jumlah_sesudah = len(df)
print(f"\n[RINGKASAN CLEANING]")
print(f"  Jumlah baris sebelum : {jumlah_sebelum}")
print(f"  Jumlah baris sesudah : {jumlah_sesudah}")
print(f"  Baris yang dihapus   : {jumlah_sebelum - jumlah_sesudah}")

# -- Ubah kolom Order_Date ke format datetime --
# Secara default, pandas membaca tanggal sebagai teks (string)
# pd.to_datetime() mengubahnya menjadi objek datetime agar bisa dihitung selisih harinya
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
print(f"\n[CLEANING] Kolom 'Order_Date' berhasil diubah ke tipe: {df['Order_Date'].dtype}")

# Verifikasi akhir: pastikan tidak ada lagi nilai NaN di kolom krusial
print(f"\n[VERIFIKASI] Jumlah nilai kosong setelah cleaning:")
print(df[['Total_Sales', 'Price_Per_Unit', 'Order_Date']].isnull().sum())


# =============================================================================
# TAHAP 3: IDENTIFIKASI PRODUK "UNDERPERFORMER"
# Visualisasi Scatter Plot: Harga vs Kuantitas
# =============================================================================

print("\n" + "=" * 60)
print("  TAHAP 3: IDENTIFIKASI PRODUK UNDERPERFORMER")
print("=" * 60)

# Menghitung nilai rata-rata (mean) untuk Price_Per_Unit dan Quantity
mean_harga    = df['Price_Per_Unit'].mean()
mean_kuantitas = df['Quantity'].mean()

print(f"  Rata-rata Harga per Unit : Rp {mean_harga:,.0f}")
print(f"  Rata-rata Kuantitas      : {mean_kuantitas:.2f} unit")

# -- Membuat Scatter Plot --
fig, ax = plt.subplots(figsize=(12, 7))

# Menentukan warna tiap titik berdasarkan kategori produk agar mudah dibedakan
kategori_unik = df['Product_Category'].unique()
warna_palet   = sns.color_palette("tab10", n_colors=len(kategori_unik))
peta_warna    = dict(zip(kategori_unik, warna_palet))

# Menggambar titik scatter per kategori agar legend muncul dengan benar
for kategori, warna in peta_warna.items():
    subset = df[df['Product_Category'] == kategori]
    ax.scatter(
        subset['Price_Per_Unit'],
        subset['Quantity'],
        label=kategori,
        color=warna,
        alpha=0.7,   # Transparansi 70% agar titik yang bertumpuk tetap terlihat
        s=80,        # Ukuran titik
        edgecolors='white',
        linewidths=0.5
    )

# -- Menambahkan Garis Rata-rata --
# axvline = garis vertikal pada nilai X tertentu (rata-rata harga)
ax.axvline(
    x=mean_harga,
    color='red',
    linestyle='--',
    linewidth=1.8,
    label=f'Rata-rata Harga: Rp {mean_harga:,.0f}'
)
# axhline = garis horizontal pada nilai Y tertentu (rata-rata kuantitas)
ax.axhline(
    y=mean_kuantitas,
    color='blue',
    linestyle='--',
    linewidth=1.8,
    label=f'Rata-rata Kuantitas: {mean_kuantitas:.1f} unit'
)

# -- Menambahkan anotasi pada 4 kuadran --
# Kuadran kanan-bawah adalah zona "Underperformer" (harga tinggi, kuantitas rendah)
offset_x = (df['Price_Per_Unit'].max() - df['Price_Per_Unit'].min()) * 0.03

ax.text(mean_harga + offset_x, mean_kuantitas + 0.3,
        '⚠ UNDERPERFORMER\n(Harga Tinggi, Qty Rendah)',
        color='red', fontsize=9, fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

ax.text(df['Price_Per_Unit'].min() + offset_x, mean_kuantitas + 0.3,
        '✓ STAR\n(Harga Rendah, Qty Tinggi)',
        color='green', fontsize=9, fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.8))

# -- Memformat sumbu X agar tampil dalam format Rupiah ---
# FuncFormatter mengubah angka menjadi string format yang kita tentukan
formatter = mticker.FuncFormatter(lambda x, _: f'Rp {x/1e6:.1f}Jt')
ax.xaxis.set_major_formatter(formatter)

# -- Label, judul, dan legenda --
ax.set_title('Scatter Plot: Identifikasi Produk Underperformer\n(Harga Per Unit vs Kuantitas Terjual)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Harga Per Unit (Rupiah)', fontsize=12)
ax.set_ylabel('Kuantitas Terjual (Unit)', fontsize=12)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plot_underperformer.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OUTPUT] Grafik disimpan sebagai 'plot_underperformer.png'")

# Menampilkan produk yang teridentifikasi sebagai underperformer di konsol
df_under = df[(df['Price_Per_Unit'] > mean_harga) & (df['Quantity'] < mean_kuantitas)]
print(f"\n[INFO] Jumlah transaksi di kuadran Underperformer: {len(df_under)}")
print(df_under[['Order_ID', 'Product_Category', 'Price_Per_Unit', 'Quantity']].head(10))


# =============================================================================
# TAHAP 4: SEGMENTASI PELANGGAN (RFM ANALYSIS)
# RFM = Recency, Frequency, Monetary
# =============================================================================

print("\n" + "=" * 60)
print("  TAHAP 4: SEGMENTASI PELANGGAN (ANALISIS RFM)")
print("=" * 60)

# -- Menentukan Tanggal Referensi --
# Kita gunakan tanggal 1 hari setelah transaksi terakhir sebagai "hari ini"
# Ini adalah praktik standar dalam RFM agar nilai Recency minimal = 1 hari
tanggal_referensi = df['Order_Date'].max() + pd.Timedelta(days=1)
print(f"  Tanggal referensi (hari ini): {tanggal_referensi.date()}")

# -- Hitung nilai R, F, M per CustomerID --
# groupby('CustomerID') mengelompokkan semua transaksi milik customer yang sama

rfm_df = df.groupby('CustomerID').agg(
    # RECENCY: Selisih hari antara tanggal referensi dan transaksi TERAKHIR customer
    # Semakin kecil angkanya, semakin baru customer berbelanja (lebih baik)
    Recency   = ('Order_Date',   lambda x: (tanggal_referensi - x.max()).days),

    # FREQUENCY: Jumlah total order yang pernah dilakukan customer
    # Semakin besar, semakin sering customer berbelanja (lebih baik)
    Frequency = ('Order_ID',     'count'),

    # MONETARY: Total nilai penjualan dari semua transaksi customer
    # Semakin besar, semakin banyak uang yang dibelanjakan customer (lebih baik)
    Monetary  = ('Total_Sales',  'sum')
).reset_index()  # reset_index() mengubah CustomerID dari index kembali menjadi kolom biasa

print(f"\n  Contoh tabel RFM (5 baris pertama):")
print(rfm_df.head())

# -- Memberikan Skor 1-5 menggunakan pd.qcut --
# pd.qcut membagi data menjadi 5 kelompok (quintile) berdasarkan distribusinya
# labels=[5,4,3,2,1] untuk Recency: nilai KECIL (baru belanja) mendapat skor TINGGI (5)
# labels=[1,2,3,4,5] untuk Frequency & Monetary: nilai BESAR mendapat skor TINGGI (5)

# Recency Score: semakin kecil recency, semakin baik → skor dibalik (5 untuk terkecil)
rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')

# Frequency Score: semakin sering belanja, semakin baik → skor normal (5 untuk terbesar)
rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])

# Monetary Score: semakin besar pengeluaran, semakin baik → skor normal (5 untuk terbesar)
rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# -- Gabungkan skor R, F, M menjadi satu kolom RFM_Group --
# .astype(str) mengubah angka skor menjadi teks agar bisa digabung (concatenate)
# Contoh: R=5, F=3, M=4 → RFM_Group = "534"
rfm_df['RFM_Group'] = (rfm_df['R_Score'].astype(str) +
                        rfm_df['F_Score'].astype(str) +
                        rfm_df['M_Score'].astype(str))

# -- Menghitung RFM Score Total untuk ranking keseluruhan --
# Mengkonversi skor ke integer lalu dijumlahkan
rfm_df['RFM_Score'] = (rfm_df['R_Score'].astype(int) +
                        rfm_df['F_Score'].astype(int) +
                        rfm_df['M_Score'].astype(int))

# -- Segmentasi Pelanggan berdasarkan RFM_Score Total --
def segmentasi_pelanggan(score):
    """Fungsi untuk mengkategorikan pelanggan berdasarkan total RFM Score."""
    if score >= 13:
        return 'Champions'           # Skor tertinggi: pelanggan terbaik
    elif score >= 10:
        return 'Loyal Customers'     # Sering belanja & pengeluaran besar
    elif score >= 7:
        return 'Potential Loyalists' # Potensi untuk menjadi loyal
    elif score >= 4:
        return 'At Risk'             # Dulunya aktif, kini mulai jarang
    else:
        return 'Lost'                # Tidak aktif dalam waktu lama

rfm_df['Segmen'] = rfm_df['RFM_Score'].apply(segmentasi_pelanggan)

print(f"\n  Distribusi Segmen Pelanggan:")
print(rfm_df['Segmen'].value_counts())

# -- Visualisasi distribusi RFM --
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Distribusi Nilai RFM Pelanggan', fontsize=14, fontweight='bold', y=1.02)

# Histogram untuk setiap komponen RFM
for ax, kolom, warna, judul in zip(
    axes,
    ['Recency', 'Frequency', 'Monetary'],
    ['#e74c3c', '#3498db', '#2ecc71'],
    ['Recency (Hari Sejak Belanja Terakhir)', 'Frequency (Jumlah Order)', 'Monetary (Total Belanja)']
):
    ax.hist(rfm_df[kolom], bins=15, color=warna, edgecolor='white', alpha=0.85)
    ax.set_title(judul, fontsize=11, fontweight='bold')
    ax.set_xlabel(kolom)
    ax.set_ylabel('Jumlah Pelanggan')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plot_rfm_distribusi.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OUTPUT] Grafik RFM disimpan sebagai 'plot_rfm_distribusi.png'")

# Menampilkan top 10 pelanggan Champions
print(f"\n  Top 10 Pelanggan 'Champions' (RFM Score Tertinggi):")
top_customers = rfm_df[rfm_df['Segmen'] == 'Champions'].sort_values('RFM_Score', ascending=False)
print(top_customers[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'RFM_Group', 'RFM_Score']].head(10))


# =============================================================================
# TAHAP 5: ANALISIS KONTRIBUSI KATEGORI
# Bar Chart Horizontal: Pendapatan vs Anggaran Iklan per Kategori
# =============================================================================

print("\n" + "=" * 60)
print("  TAHAP 5: ANALISIS KONTRIBUSI KATEGORI PRODUK")
print("=" * 60)

# -- Agregasi data per kategori produk --
# Menjumlahkan Total_Sales dan Ad_Budget untuk setiap Product_Category
kategori_df = df.groupby('Product_Category').agg(
    Total_Pendapatan  = ('Total_Sales', 'sum'),
    Total_Iklan       = ('Ad_Budget',   'sum')
).reset_index()

# -- Menghitung Rasio Efisiensi: berapa rupiah pendapatan per 1 rupiah iklan --
# Semakin tinggi rasio, semakin efisien kategori tersebut menggunakan anggaran iklan
kategori_df['Rasio_Efisiensi'] = kategori_df['Total_Pendapatan'] / kategori_df['Total_Iklan']

# -- Urutkan dari yang paling efisien ke yang kurang efisien --
kategori_df = kategori_df.sort_values('Rasio_Efisiensi', ascending=True)

print("  Tabel Efisiensi Iklan per Kategori:")
print(kategori_df[['Product_Category', 'Total_Pendapatan', 'Total_Iklan', 'Rasio_Efisiensi']].to_string(index=False))

# -- Membuat Bar Chart Horizontal --
fig, ax = plt.subplots(figsize=(13, 7))

# Menentukan posisi bar pada sumbu Y
posisi_y = range(len(kategori_df))
tinggi_bar = 0.35  # Lebar tiap bar (dalam satuan matplotlib)

# Menggambar bar untuk Total Pendapatan (warna hijau)
bar_pendapatan = ax.barh(
    [p + tinggi_bar / 2 for p in posisi_y],
    kategori_df['Total_Pendapatan'],
    height=tinggi_bar,
    label='Total Pendapatan (Sales)',
    color='#27ae60',
    alpha=0.85
)

# Menggambar bar untuk Total Anggaran Iklan (warna oranye)
bar_iklan = ax.barh(
    [p - tinggi_bar / 2 for p in posisi_y],
    kategori_df['Total_Iklan'],
    height=tinggi_bar,
    label='Total Anggaran Iklan',
    color='#e67e22',
    alpha=0.85
)

# -- Menambahkan label nilai di ujung setiap bar --
for bar in bar_pendapatan:
    lebar = bar.get_width()
    ax.text(lebar + lebar * 0.01, bar.get_y() + bar.get_height() / 2,
            f'Rp {lebar/1e6:.1f}Jt', va='center', ha='left', fontsize=9)

for bar in bar_iklan:
    lebar = bar.get_width()
    ax.text(lebar + lebar * 0.01, bar.get_y() + bar.get_height() / 2,
            f'Rp {lebar/1e6:.1f}Jt', va='center', ha='left', fontsize=9)

# -- Menambahkan label Rasio Efisiensi di sebelah kanan --
for i, (_, baris) in enumerate(kategori_df.iterrows()):
    ax.text(ax.get_xlim()[1] * 0.98 if ax.get_xlim()[1] > 0 else kategori_df['Total_Pendapatan'].max() * 1.15,
            i,
            f'Efisiensi: {baris["Rasio_Efisiensi"]:.2f}x',
            va='center', ha='right', fontsize=9, color='navy',
            fontweight='bold')

# -- Mengatur label sumbu Y dengan nama kategori --
ax.set_yticks(list(posisi_y))
ax.set_yticklabels(kategori_df['Product_Category'], fontsize=11)

# -- Memformat sumbu X agar tampil dalam Juta Rupiah --
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'Rp {x/1e6:.0f}Jt'))

# -- Label, judul, legenda --
ax.set_title('Perbandingan Pendapatan vs Anggaran Iklan per Kategori Produk\n(Diurutkan dari Paling Efisien)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Jumlah (Rupiah)', fontsize=12)
ax.set_ylabel('Kategori Produk', fontsize=12)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plot_kontribusi_kategori.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OUTPUT] Grafik disimpan sebagai 'plot_kontribusi_kategori.png'")


# =============================================================================
# TAHAP 6: UJI HIPOTESIS SEDERHANA
# Apakah Anggaran Iklan Tinggi Menghasilkan Penjualan Lebih Tinggi?
# =============================================================================

print("\n" + "=" * 60)
print("  TAHAP 6: UJI HIPOTESIS - PENGARUH ANGGARAN IKLAN")
print("=" * 60)

# -- Menghitung Median (nilai tengah) dari Ad_Budget --
# Menggunakan median (bukan mean) agar lebih robust terhadap outlier/data ekstrem
median_iklan = df['Ad_Budget'].median()
print(f"  Median Anggaran Iklan  : Rp {median_iklan:,.0f}")

# -- Membagi data menjadi dua kelompok berdasarkan median --
# Kelompok "Iklan Tinggi": Ad_Budget >= median
df_iklan_tinggi = df[df['Ad_Budget'] >= median_iklan]
# Kelompok "Iklan Rendah": Ad_Budget < median
df_iklan_rendah = df[df['Ad_Budget'] < median_iklan]

print(f"\n  Jumlah transaksi 'Iklan Tinggi' : {len(df_iklan_tinggi)} transaksi")
print(f"  Jumlah transaksi 'Iklan Rendah' : {len(df_iklan_rendah)} transaksi")

# -- Menghitung rata-rata Total_Sales masing-masing kelompok --
rata_iklan_tinggi = df_iklan_tinggi['Total_Sales'].mean()
rata_iklan_rendah = df_iklan_rendah['Total_Sales'].mean()

print(f"\n  *** HASIL UJI HIPOTESIS ***")
print(f"  Rata-rata Total Sales (Iklan TINGGI) : Rp {rata_iklan_tinggi:>12,.0f}")
print(f"  Rata-rata Total Sales (Iklan RENDAH) : Rp {rata_iklan_rendah:>12,.0f}")

# Menghitung selisih dan persentase perbedaan
selisih       = rata_iklan_tinggi - rata_iklan_rendah
persen_lebih  = (selisih / rata_iklan_rendah) * 100

print(f"\n  Selisih Rata-rata Sales             : Rp {selisih:>12,.0f}")
print(f"  Iklan Tinggi lebih besar sebesar    : {persen_lebih:.1f}%")

# -- Interpretasi hasil --
print(f"\n  *** INTERPRETASI ***")
if rata_iklan_tinggi > rata_iklan_rendah:
    print(f"  ✓ Kelompok dengan anggaran iklan TINGGI memiliki rata-rata penjualan")
    print(f"    {persen_lebih:.1f}% LEBIH TINGGI dibandingkan kelompok iklan rendah.")
    print(f"  ✓ Ini mendukung hipotesis bahwa iklan berpengaruh positif terhadap penjualan.")
else:
    print(f"  ✗ Kelompok dengan anggaran iklan TINGGI tidak menghasilkan penjualan lebih tinggi.")
    print(f"  ✗ Perlu investigasi lebih lanjut mengenai efektivitas strategi iklan.")

# -- Visualisasi perbandingan dua kelompok --
fig, ax = plt.subplots(figsize=(8, 6))

kelompok = ['Iklan Rendah\n(< Median)', 'Iklan Tinggi\n(≥ Median)']
nilai_rata = [rata_iklan_rendah, rata_iklan_tinggi]
warna_bar  = ['#e74c3c', '#27ae60']  # Merah untuk rendah, hijau untuk tinggi

batang = ax.bar(kelompok, nilai_rata, color=warna_bar, edgecolor='white', width=0.5, alpha=0.85)

# Menambahkan label nilai di atas setiap bar
for bar, nilai in zip(batang, nilai_rata):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_height() * 0.02,
            f'Rp {nilai/1e6:.2f}Jt', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_title('Rata-rata Total Sales\nBerdasarkan Kelompok Anggaran Iklan',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Rata-rata Total Sales (Rupiah)', fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'Rp {x/1e6:.1f}Jt'))
ax.axhline(y=(rata_iklan_tinggi + rata_iklan_rendah) / 2, color='gray',
           linestyle=':', linewidth=1.5, label='Rata-rata Keseluruhan')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plot_uji_hipotesis.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OUTPUT] Grafik disimpan sebagai 'plot_uji_hipotesis.png'")


# =============================================================================
# TAHAP 7: REGRESI LINEAR SEDERHANA
# Memprediksi Total_Sales berdasarkan Ad_Budget
# Y = β₀ + β₁·X  (β₀=intercept, β₁=koefisien iklan)
# =============================================================================

print("\n" + "=" * 60)
print("  TAHAP 7: MODEL REGRESI LINEAR SEDERHANA")
print("=" * 60)

# -- Menyiapkan fitur (X) dan target (y) --
# X harus berbentuk 2D array (matriks), sehingga digunakan double bracket [[...]]
# y cukup 1D array (vektor)
X = df[['Ad_Budget']]   # Fitur: Anggaran Iklan (variabel independen)
y = df['Total_Sales']   # Target: Total Penjualan (variabel dependen)

print(f"  Jumlah sampel total      : {len(X)}")

# -- Membagi data menjadi Training Set dan Testing Set --
# test_size=0.2 → 20% data untuk testing, 80% untuk training
# random_state=42 → seed agar hasil pembagian selalu sama (reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"  Jumlah data training     : {len(X_train)} sampel (80%)")
print(f"  Jumlah data testing      : {len(X_test)} sampel (20%)")

# -- Membuat dan Melatih Model Regresi Linear --
model = LinearRegression()

# model.fit() adalah proses "belajar": model mencari nilai β₀ dan β₁
# yang paling baik meminimalkan error antara prediksi dan nilai aktual
model.fit(X_train, y_train)

print(f"\n  [MODEL TERLATIH]")

# -- Mengekstrak dan menampilkan parameter model --
koefisien_iklan = model.coef_[0]    # β₁: koefisien untuk Ad_Budget
intercept       = model.intercept_  # β₀: nilai awal (jika iklan = 0)

print(f"  Persamaan Model  : Total_Sales = {intercept:,.0f} + {koefisien_iklan:.4f} × Ad_Budget")
print(f"\n  Intercept (β₀)   : Rp {intercept:,.0f}")
print(f"  Koefisien Iklan (β₁) : {koefisien_iklan:.4f}")
print(f"\n  Interpretasi β₁  :")
print(f"  → Setiap kenaikan Rp 1 pada Ad_Budget, Total_Sales naik sebesar Rp {koefisien_iklan:.4f}")
print(f"  → Setiap kenaikan Rp 1.000.000 pada iklan, sales naik Rp {koefisien_iklan*1e6:,.0f}")

# -- Evaluasi Model dengan R² Score --
# Prediksi nilai sales pada data testing (data yang belum pernah dilihat model)
y_pred = model.predict(X_test)

# R² (R-squared) mengukur seberapa baik model menjelaskan variasi data
# R² = 1.0 → model sempurna, R² = 0.0 → model tidak lebih baik dari rata-rata
r2 = r2_score(y_test, y_pred)

print(f"\n  *** EVALUASI AKURASI MODEL ***")
print(f"  R² Score (Akurasi Model) : {r2:.4f} ({r2*100:.2f}%)")

# Interpretasi R² Score
print(f"\n  Interpretasi R² = {r2:.2f}:")
if r2 >= 0.7:
    print(f"  ✓ BAIK: Model mampu menjelaskan {r2*100:.1f}% variasi Total_Sales")
    print(f"    yang disebabkan oleh Ad_Budget.")
elif r2 >= 0.4:
    print(f"  ~ CUKUP: Model menjelaskan {r2*100:.1f}% variasi, namun ada faktor lain")
    print(f"    yang mempengaruhi penjualan (harga, kategori, musim, dll).")
else:
    print(f"  ✗ LEMAH: Ad_Budget saja hanya menjelaskan {r2*100:.1f}% variasi penjualan.")
    print(f"    Pertimbangkan untuk menambahkan fitur lain (regresi multipel).")

# -- Visualisasi Regresi Linear --
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# -- Plot 1: Scatter + Garis Regresi --
ax1 = axes[0]
ax1.scatter(X_test, y_test,
            color='steelblue', alpha=0.6, s=60, edgecolors='white', label='Data Aktual (Test)')

# Membuat garis prediksi: hitung prediksi untuk range nilai X yang mulus
import numpy as np
x_garis  = np.linspace(X['Ad_Budget'].min(), X['Ad_Budget'].max(), 200).reshape(-1, 1)
y_garis  = model.predict(x_garis)

ax1.plot(x_garis, y_garis,
         color='red', linewidth=2.5, label=f'Garis Regresi (R²={r2:.2f})')

ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'Rp {x/1e6:.1f}Jt'))
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'Rp {x/1e6:.1f}Jt'))
ax1.set_title('Regresi Linear: Ad_Budget → Total_Sales', fontsize=12, fontweight='bold')
ax1.set_xlabel('Anggaran Iklan (Ad_Budget)', fontsize=11)
ax1.set_ylabel('Total Penjualan (Total_Sales)', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# -- Plot 2: Prediksi vs Aktual --
ax2 = axes[1]
ax2.scatter(y_test, y_pred,
            color='#9b59b6', alpha=0.6, s=60, edgecolors='white', label='Prediksi vs Aktual')

# Garis diagonal sempurna (jika prediksi = aktual, semua titik ada di garis ini)
batas_min = min(y_test.min(), y_pred.min())
batas_max = max(y_test.max(), y_pred.max())
ax2.plot([batas_min, batas_max], [batas_min, batas_max],
         color='red', linewidth=2, linestyle='--', label='Prediksi Sempurna')

ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'Rp {x/1e6:.0f}Jt'))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'Rp {x/1e6:.0f}Jt'))
ax2.set_title(f'Nilai Aktual vs Nilai Prediksi\n(R² = {r2:.4f})', fontsize=12, fontweight='bold')
ax2.set_xlabel('Total Sales Aktual', fontsize=11)
ax2.set_ylabel('Total Sales Prediksi', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plot_regresi_linear.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OUTPUT] Grafik disimpan sebagai 'plot_regresi_linear.png'")


# =============================================================================
# RINGKASAN AKHIR SEMUA HASIL ANALISIS
# =============================================================================

print("\n" + "=" * 60)
print("  RINGKASAN HASIL ANALISIS LENGKAP")
print("=" * 60)
print(f"""
  [DATA]
  - Total transaksi valid    : {len(df)} baris
  - Periode data             : {df['Order_Date'].min().date()} s/d {df['Order_Date'].max().date()}
  - Kategori produk          : {', '.join(df['Product_Category'].unique())}

  [UNDERPERFORMER - Tahap 3]
  - Produk dg harga > rata² & qty < rata²: {len(df_under)} transaksi
  - Rata-rata harga          : Rp {mean_harga:,.0f}
  - Rata-rata kuantitas      : {mean_kuantitas:.2f} unit

  [RFM SEGMENTASI - Tahap 4]
  - Total pelanggan dianalisis: {len(rfm_df)} pelanggan
  - Segmen Champions         : {len(rfm_df[rfm_df['Segmen'] == 'Champions'])} pelanggan
  - Segmen Loyal Customers   : {len(rfm_df[rfm_df['Segmen'] == 'Loyal Customers'])} pelanggan
  - Segmen At Risk           : {len(rfm_df[rfm_df['Segmen'] == 'At Risk'])} pelanggan

  [KONTRIBUSI KATEGORI - Tahap 5]
  - Kategori paling efisien  : {kategori_df.iloc[-1]['Product_Category']} (rasio {kategori_df.iloc[-1]['Rasio_Efisiensi']:.2f}x)
  - Kategori kurang efisien  : {kategori_df.iloc[0]['Product_Category']} (rasio {kategori_df.iloc[0]['Rasio_Efisiensi']:.2f}x)

  [UJI HIPOTESIS IKLAN - Tahap 6]
  - Rata-rata sales iklan tinggi : Rp {rata_iklan_tinggi:,.0f}
  - Rata-rata sales iklan rendah : Rp {rata_iklan_rendah:,.0f}
  - Perbedaan                    : +{persen_lebih:.1f}%

  [MODEL REGRESI - Tahap 7]
  - Koefisien Iklan (β₁)     : {koefisien_iklan:.4f}
  - R² Score (Akurasi)       : {r2:.4f} ({r2*100:.2f}%)
""")

print("=" * 60)
print("  ANALISIS SELESAI! Semua grafik telah disimpan.")
print("=" * 60)
