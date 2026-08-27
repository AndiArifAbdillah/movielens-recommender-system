# %% [markdown]
# # Proyek Akhir: Membuat Model Sistem Rekomendasi
# ## Sistem Rekomendasi Film — *Content-Based Filtering* & *Collaborative Filtering*
#
# - **Nama:** Andi Arif Abdillah
# - **Kelas:** Machine Learning Terapan — Dicoding
# - **Dataset:** [MovieLens *Small*](https://grouplens.org/datasets/movielens/latest/) (100.836 rating) — GroupLens Research, University of Minnesota
#
# ---
#
# ### Ringkasan Alur Notebook
#
# | Tahap | Isi |
# |---|---|
# | 1 | Import *library* dan konfigurasi |
# | 2 | *Data Loading* — pengunduhan dan pemuatan empat berkas dataset |
# | 3 | *Data Understanding* — struktur, kondisi data, dan *exploratory data analysis* |
# | 4 | *Data Preparation* — delapan tahap persiapan data yang berurutan |
# | 5 | *Modeling* — tiga model rekomendasi dan satu *baseline* pembanding |
# | 6 | *Evaluation* — metrik prediksi rating dan metrik peringkat (*ranking*) |
# | 7 | Kesimpulan |
#
# Seluruh keluaran notebook ini dapat direproduksi karena setiap sumber keacakan
# dikunci dengan *random seed* yang tetap.

# %% [markdown]
# ---
# # 0. Project Overview dan Business Understanding
#
# ## Latar Belakang
#
# Katalog layanan hiburan digital hari ini berisi puluhan ribu judul, jauh
# melampaui kemampuan siapa pun untuk menelusurinya satu per satu. Ketika pilihan
# terlalu banyak, pengguna justru sulit memutuskan — gejala yang dikenal sebagai
# *choice overload*. Sistem rekomendasi hadir sebagai jawabannya: ia mempersempit
# katalog raksasa menjadi belasan judul yang benar-benar layak dipertimbangkan
# seorang pengguna.
#
# Nilai ekonominya nyata. Netflix melaporkan bahwa lebih dari 80% jam tontonan di
# platform mereka berasal dari rekomendasi, bukan dari pencarian mandiri, dan
# menaksir nilai sistem rekomendasi mereka lebih dari satu miliar dolar per tahun
# (Gomez-Uribe & Hunt, 2015). Bagi penyedia layanan, rekomendasi yang baik
# menurunkan tingkat berhenti berlangganan sekaligus menghidupkan katalog bagian
# ekor yang jarang tersentuh.
#
# ## Mengapa Proyek Ini Penting Diselesaikan
#
# 1. **Tanpa rekomendasi, katalog panjang menjadi beban, bukan aset.** Analisis
#    data pada notebook ini menunjukkan hanya sekitar 7% film menyerap separuh
#    seluruh rating, sementara 62% film menerima kurang dari lima rating. Ribuan
#    judul praktis tidak pernah ditemukan siapa pun.
# 2. **Waktu pengguna terbatas.** Layar rekomendasi biasanya hanya memuat
#    sekitar sepuluh judul. Setiap slot yang salah adalah kesempatan yang hilang.
# 3. **Pilihan pendekatan bukan hal sepele.** *Content-based filtering* dan
#    *collaborative filtering* memiliki kekuatan dan kelemahan yang berbeda.
#    Menentukan mana yang tepat memerlukan bukti kuantitatif, bukan asumsi.
#
# ## Problem Statements
#
# 1. Bagaimana merekomendasikan film yang relevan bagi seorang pengguna tanpa
#    mengharuskannya menjelajahi katalog berisi ribuan judul?
# 2. Bagaimana sistem tetap dapat memberi rekomendasi untuk film yang jarang atau
#    belum pernah dirating, yang jumlahnya mendominasi katalog?
# 3. Pendekatan mana yang lebih tepat untuk kasus ini, dan dengan bukti apa
#    keunggulannya dapat dipertanggungjawabkan?
#
# ## Goals
#
# 1. Menghasilkan **top-10 rekomendasi film** yang dipersonalisasi bagi setiap
#    pengguna.
# 2. Membangun sistem yang tetap mampu menjangkau film di ekor katalog, diukur
#    melalui metrik cakupan katalog (*coverage*).
# 3. Membuktikan secara terukur bahwa model yang dipilih **mengungguli baseline
#    populer** pada metrik Precision@10, Recall@10, dan NDCG@10.
#
# ## Solution Approach
#
# **Pendekatan 1 — *Content-Based Filtering*.** Genre dan tag setiap film diubah
# menjadi vektor TF-IDF, lalu kemiripan antarfilm dihitung dengan *cosine
# similarity*. Rekomendasi disusun dari film yang kontennya paling mirip dengan
# yang disukai pengguna.
#
# **Pendekatan 2 — *Collaborative Filtering*.** Pola interaksi antarpengguna
# dipelajari melalui faktorisasi matriks, dalam dua varian: *Matrix
# Factorization* berbias yang dilatih dengan SGD pada rating eksplisit, dan
# *Implicit ALS* yang dilatih pada sinyal biner "disukai".
#
# Sebagai pembanding kelayakan, disertakan pula **baseline popularitas** yang
# merekomendasikan film terpopuler kepada semua orang.
#
# > Uraian lengkap latar belakang, riset terkait, dan pembahasan hasil disajikan
# > pada berkas laporan `Laporan_Proyek_Sistem_Rekomendasi.md`.

# %% [markdown]
# ---
# # 1. Import Library
#
# Proyek ini sengaja dibangun **tanpa pustaka sistem rekomendasi siap pakai**
# (seperti `surprise` atau `implicit`). Kedua algoritma *collaborative filtering*
# diimplementasikan langsung dengan NumPy agar setiap komponen matematis yang
# dijelaskan pada laporan — fungsi objektif, aturan pembaruan, dan regularisasi —
# benar-benar tercermin pada kode yang dijalankan.

# %%
import os
import re
import time
import zipfile
import urllib.request
from io import BytesIO

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Saat berkas ini dijalankan sebagai skrip .py biasa (bukan di notebook),
# backend non-interaktif dipakai agar gambar tetap tersimpan ke folder images/
# tanpa membuka jendela yang menghentikan eksekusi.
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Konfigurasi tampilan
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 160)
sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.dpi'] = 130
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10

# Kunci seluruh sumber keacakan
SEED = 42
np.random.seed(SEED)

# Direktori keluaran gambar untuk laporan
IMG_DIR = 'images'
os.makedirs(IMG_DIR, exist_ok=True)


def simpan_gambar(nama):
    """Menyimpan figure yang aktif ke folder images/ agar dapat dirujuk laporan .md"""
    plt.savefig(os.path.join(IMG_DIR, nama))


print('NumPy        :', np.__version__)
print('pandas       :', pd.__version__)
print('SciPy        :', scipy.__version__)
print('scikit-learn :', sklearn.__version__)

# %% [markdown]
# ---
# # 2. Data Loading
#
# Dataset yang digunakan adalah **MovieLens *ml-latest-small*** yang dirilis oleh
# [GroupLens Research](https://grouplens.org/datasets/movielens/latest/), sebuah
# laboratorium riset di University of Minnesota. Berkas diunduh langsung dari
# tautan resmi berikut:
#
# > **Tautan unduh:** <https://files.grouplens.org/datasets/movielens/ml-latest-small.zip>
#
# Sel di bawah memakai salinan lokal apabila tersedia; jika tidak, berkas akan
# diunduh dan diekstrak secara otomatis. Dengan begitu notebook dapat dijalankan
# ulang baik di Google Colab maupun di komputer lokal.

# %%
URL_DATASET = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
DIR_DATA = os.path.join('data', 'ml-latest-small')

if not os.path.exists(os.path.join(DIR_DATA, 'ratings.csv')):
    print('Mengunduh dataset dari GroupLens ...')
    with urllib.request.urlopen(URL_DATASET, timeout=120) as resp:
        konten = resp.read()
    zipfile.ZipFile(BytesIO(konten)).extractall('data')
    print('Selesai. Ukuran arsip: {:.1f} KB'.format(len(konten) / 1024))
else:
    print('Menggunakan salinan dataset yang sudah tersedia secara lokal.')

print('\nIsi direktori dataset:')
for berkas in sorted(os.listdir(DIR_DATA)):
    ukuran = os.path.getsize(os.path.join(DIR_DATA, berkas)) / 1024
    print(f'  - {berkas:<12} {ukuran:8.1f} KB')

# %% [markdown]
# ### Memuat Keempat Berkas ke dalam DataFrame
#
# Keempat berkas CSV dibaca sekaligus agar strukturnya dapat dibandingkan
# berdampingan pada tahap pemahaman data.

# %%
ratings = pd.read_csv(os.path.join(DIR_DATA, 'ratings.csv'))
movies = pd.read_csv(os.path.join(DIR_DATA, 'movies.csv'))
tags = pd.read_csv(os.path.join(DIR_DATA, 'tags.csv'))
links = pd.read_csv(os.path.join(DIR_DATA, 'links.csv'))

print('Jumlah baris dan kolom tiap berkas')
for nama, df in [('ratings', ratings), ('movies', movies), ('tags', tags), ('links', links)]:
    print(f'  {nama:<8}: {df.shape[0]:>6,} baris x {df.shape[1]} kolom')

# %% [markdown]
# ---
# # 3. Data Understanding
#
# ## 3.1 Struktur Berkas dan Uraian Variabel
#
# Dataset terdiri atas empat berkas CSV yang saling terhubung melalui kunci
# `movieId` dan `userId`.

# %%
for nama, df in [('ratings', ratings), ('movies', movies), ('tags', tags), ('links', links)]:
    print('=' * 78)
    print(f'BERKAS: {nama}.csv  —  {df.shape[0]:,} baris, {df.shape[1]} kolom')
    print('=' * 78)
    df.info()
    print('\nCuplikan 5 baris pertama:')
    print(df.head().to_string(index=False))
    print()

# %% [markdown]
# ### Uraian Seluruh Variabel
#
# **`ratings.csv` — 100.836 baris (data interaksi utama)**
#
# | Variabel | Tipe | Keterangan |
# |---|---|---|
# | `userId` | int64 | Identitas anonim pengguna. Setiap pengguna dijamin memiliki minimal 20 rating. |
# | `movieId` | int64 | Identitas film; menjadi kunci penghubung ke `movies.csv`, `tags.csv`, dan `links.csv`. |
# | `rating` | float64 | Penilaian eksplisit pengguna pada skala 0,5–5,0 dengan kelipatan 0,5 (10 tingkat). |
# | `timestamp` | int64 | Waktu pemberian rating dalam detik sejak 1 Januari 1970 (UTC). Dipakai untuk pemisahan data secara temporal. |
#
# **`movies.csv` — 9.742 baris (metadata konten)**
#
# | Variabel | Tipe | Keterangan |
# |---|---|---|
# | `movieId` | int64 | Identitas film. |
# | `title` | object | Judul film beserta tahun rilis dalam tanda kurung, contoh `Toy Story (1995)`. |
# | `genres` | object | Daftar genre yang dipisahkan tanda pipa, contoh `Adventure\|Animation\|Children`. Terdapat 19 kategori genre serta nilai khusus `(no genres listed)`. |
#
# **`tags.csv` — 3.683 baris (metadata konten yang berasal dari pengguna)**
#
# | Variabel | Tipe | Keterangan |
# |---|---|---|
# | `userId` | int64 | Pengguna yang memberi tag. |
# | `movieId` | int64 | Film yang diberi tag. |
# | `tag` | object | Kata atau frasa bebas buatan pengguna, contoh `pixar`, `dark comedy`, `based on a book`. |
# | `timestamp` | int64 | Waktu pemberian tag. |
#
# **`links.csv` — 9.742 baris (pranala ke basis data film eksternal)**
#
# | Variabel | Tipe | Keterangan |
# |---|---|---|
# | `movieId` | int64 | Identitas film pada MovieLens. |
# | `imdbId` | int64 | Identitas film yang bersesuaian pada IMDb. |
# | `tmdbId` | float64 | Identitas film yang bersesuaian pada TMDB; terdapat 8 nilai kosong. |
#
# Berkas `links.csv` hanya berguna untuk pengayaan data dari sumber eksternal
# (misalnya poster atau sinopsis) sehingga **tidak dipakai** pada proyek ini.

# %% [markdown]
# ## 3.2 Kondisi Data
#
# Pemeriksaan kondisi data mencakup empat hal: nilai kosong (*missing value*),
# duplikasi baris, duplikasi pasangan kunci `(userId, movieId)`, dan konsistensi
# rentang nilai.

# %%
print('--- Nilai kosong (missing value) per berkas ---')
for nama, df in [('ratings', ratings), ('movies', movies), ('tags', tags), ('links', links)]:
    kosong = df.isna().sum()
    kosong = kosong[kosong > 0]
    if len(kosong) == 0:
        print(f'  {nama:<8}: tidak ada nilai kosong')
    else:
        print(f'  {nama:<8}: {kosong.to_dict()}')

print('\n--- Duplikasi baris utuh ---')
for nama, df in [('ratings', ratings), ('movies', movies), ('tags', tags), ('links', links)]:
    print(f'  {nama:<8}: {df.duplicated().sum()} baris duplikat')

print('\n--- Duplikasi pasangan kunci ---')
print('  ratings (userId, movieId):', ratings.duplicated(subset=['userId', 'movieId']).sum())
print('  movies  (movieId)        :', movies.duplicated(subset=['movieId']).sum())
print('  movies  (title)          :', movies.duplicated(subset=['title']).sum())

print('\n--- Konsistensi rentang nilai ---')
print('  Rentang rating           :', ratings.rating.min(), '-', ratings.rating.max())
print('  Nilai rating unik        :', sorted(ratings.rating.unique()))
print('  Rentang waktu pengamatan :',
      pd.to_datetime(ratings.timestamp, unit='s').min().date(), 's/d',
      pd.to_datetime(ratings.timestamp, unit='s').max().date())
print('  movieId pada ratings yang tidak ada di movies:',
      int((~ratings.movieId.isin(movies.movieId)).sum()))

# %% [markdown]
# ### Statistik Deskriptif dan Skala Masalah
#
# Sel berikut merangkum sebaran nilai rating sekaligus menghitung tingkat
# kerenggangan (*sparsity*) matriks interaksi — angka yang menentukan seberapa
# sulit persoalan yang dihadapi kedua pendekatan rekomendasi.

# %%
print('--- Statistik deskriptif kolom rating ---')
print(ratings.rating.describe().round(3).to_string())

n_user = ratings.userId.nunique()
n_movie_rated = ratings.movieId.nunique()
n_movie_total = movies.movieId.nunique()
sparsity = 1 - len(ratings) / (n_user * n_movie_rated)

print('\n--- Ringkasan skala masalah ---')
print(f'  Jumlah pengguna unik             : {n_user:,}')
print(f'  Jumlah film pada katalog         : {n_movie_total:,}')
print(f'  Jumlah film yang pernah dirating : {n_movie_rated:,}')
print(f'  Jumlah film tanpa satu pun rating: {n_movie_total - n_movie_rated:,}')
print(f'  Jumlah rating                    : {len(ratings):,}')
print(f'  Kepadatan matriks interaksi      : {(1 - sparsity) * 100:.2f}%')
print(f'  Tingkat kerenggangan (sparsity)  : {sparsity * 100:.2f}%')

# %% [markdown]
# **Kondisi data secara ringkas:**
#
# 1. **Tidak ada nilai kosong** pada `ratings`, `movies`, maupun `tags`. Delapan
#    nilai kosong hanya terdapat pada kolom `tmdbId` di `links.csv`, berkas yang
#    tidak dipakai pada proyek ini.
# 2. **Tidak ada baris duplikat** dan tidak ada pasangan `(userId, movieId)`
#    ganda, sehingga setiap pengguna hanya memberi satu rating untuk satu film.
# 3. **Seluruh nilai rating berada pada rentang yang sah** (0,5–5,0 dengan
#    kelipatan 0,5), dan setiap `movieId` pada `ratings` selalu ditemukan pada
#    `movies` sehingga tidak ada relasi yang menggantung.
# 4. Terdapat **18 film yang belum pernah dirating sama sekali** — kasus
#    *cold-start item* yang nyata pada data.
# 5. Matriks interaksi sangat renggang: hanya sekitar **1,70%** sel yang terisi.
#    Kerenggangan inilah tantangan utama sistem rekomendasi.
