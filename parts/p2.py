# %% [markdown]
# ## 3.3 Exploratory Data Analysis (EDA)
#
# Bagian ini menggali karakteristik data melalui visualisasi. Setiap gambar
# disertai *insight* yang kemudian menjadi dasar keputusan pada tahap
# *Data Preparation* dan *Modeling*.

# %% [markdown]
# ### 3.3.1 Distribusi Nilai Rating

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

urut = sorted(ratings.rating.unique())
frek = ratings.rating.value_counts().reindex(urut)
warna = ['#d1495b' if r < 3.5 else '#2e86ab' for r in urut]
ax[0].bar([str(r) for r in urut], frek.values, color=warna)
for i, v in enumerate(frek.values):
    ax[0].text(i, v + 700, f'{v/len(ratings)*100:.1f}%', ha='center', fontsize=8)
ax[0].set_title('Distribusi Nilai Rating')
ax[0].set_xlabel('Nilai rating')
ax[0].set_ylabel('Jumlah rating')
ax[0].axvline(x=5.5, color='gray', ls='--', lw=1)
ax[0].text(5.6, frek.max() * 0.9, 'ambang "disukai"\n(rating >= 4,0)', fontsize=8, color='gray')

rata_user = ratings.groupby('userId').rating.mean()
ax[1].hist(rata_user, bins=30, color='#2e86ab', edgecolor='white')
ax[1].axvline(ratings.rating.mean(), color='#d1495b', ls='--',
              label=f'rata-rata global = {ratings.rating.mean():.2f}')
ax[1].set_title('Distribusi Rata-rata Rating per Pengguna')
ax[1].set_xlabel('Rata-rata rating seorang pengguna')
ax[1].set_ylabel('Jumlah pengguna')
ax[1].legend(fontsize=8)

plt.tight_layout()
simpan_gambar('fig_01_distribusi_rating.png')
plt.show()

print('Proporsi rating >= 4,0 (kategori "disukai"):',
      f'{(ratings.rating >= 4).mean() * 100:.1f}%')
print('Rata-rata rating global :', round(ratings.rating.mean(), 3))
print('Median rating global    :', ratings.rating.median())
print('Sebaran rata-rata rating antar pengguna: min={:.2f}, maks={:.2f}, std={:.2f}'.format(
    rata_user.min(), rata_user.max(), rata_user.std()))

# %% [markdown]
# **Insight 1 — Rating condong ke nilai tinggi dan setiap pengguna punya "gaya menilai" sendiri.**
#
# Nilai 4,0 adalah rating yang paling sering diberikan dan sekitar 48% rating
# bernilai ≥ 4,0. Distribusi yang miring ke kanan ini wajar pada data rekomendasi
# karena pengguna cenderung hanya menonton film yang memang mereka minati.
# Konsekuensinya, **rating 3,0 bukan berarti "netral"** melainkan relatif rendah.
#
# Panel kanan memperlihatkan hal yang lebih penting: rata-rata rating antarpengguna
# tersebar luas dari sekitar 1,3 sampai 5,0. Ada pengguna yang murah hati dan ada
# yang pelit memberi nilai. Perbedaan sistematis ini yang nantinya ditangani
# secara eksplisit oleh **suku bias pengguna** ($b_u$) pada model *Matrix
# Factorization*, sehingga model tidak salah menafsirkan rating 3,5 dari pengguna
# pelit sebagai sinyal negatif.

# %% [markdown]
# ### 3.3.2 Pola Ekor Panjang (*Long Tail*)

# %%
rating_per_film = ratings.groupby('movieId').size().sort_values(ascending=False)
rating_per_user = ratings.groupby('userId').size().sort_values(ascending=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(np.arange(1, len(rating_per_film) + 1), rating_per_film.values, color='#2e86ab')
ax[0].set_yscale('log')
ax[0].set_title('Jumlah Rating per Film (terurut menurun)')
ax[0].set_xlabel('Peringkat popularitas film')
ax[0].set_ylabel('Jumlah rating (skala log)')
n_top = int((rating_per_film.cumsum() <= len(ratings) * 0.5).sum()) + 1
ax[0].axvline(n_top, color='#d1495b', ls='--')
ax[0].text(n_top * 1.15, 100,
           f'{n_top} film ({n_top/len(rating_per_film)*100:.1f}% katalog)\nmenyerap 50% seluruh rating',
           fontsize=8, color='#d1495b')

ax[1].hist(rating_per_user.values, bins=50, color='#2e86ab', edgecolor='white')
ax[1].set_yscale('log')
ax[1].set_title('Distribusi Jumlah Rating per Pengguna')
ax[1].set_xlabel('Jumlah rating yang diberikan seorang pengguna')
ax[1].set_ylabel('Jumlah pengguna (skala log)')
ax[1].axvline(rating_per_user.median(), color='#d1495b', ls='--',
              label=f'median = {int(rating_per_user.median())} rating')
ax[1].legend(fontsize=8)

plt.tight_layout()
simpan_gambar('fig_02_long_tail.png')
plt.show()

print('--- Statistik jumlah rating per FILM ---')
print(rating_per_film.describe().round(2).to_string())
print(f'\n  Film dengan < 5 rating : {(rating_per_film < 5).sum():,} '
      f'({(rating_per_film < 5).mean()*100:.1f}% dari film yang pernah dirating)')
print(f'  Film dengan 1 rating   : {(rating_per_film == 1).sum():,}')

print('\n--- Statistik jumlah rating per PENGGUNA ---')
print(rating_per_user.describe().round(2).to_string())

# %% [markdown]
# **Insight 2 — Data sangat timpang, baik di sisi film maupun di sisi pengguna.**
#
# Hanya sekitar 7% film (659 judul) yang menyerap separuh dari seluruh rating, sementara lebih dari
# separuh film memiliki kurang dari 5 rating dan 3.446 film bahkan hanya dirating
# satu kali. Pola *long tail* ini punya dua akibat langsung:
#
# 1. *Collaborative filtering* **tidak punya sinyal yang memadai** untuk film di
#    ekor distribusi. Karena itu, pada tahap evaluasi peringkat digunakan
#    *candidate pool* berisi film dengan minimal 5 rating (lihat tahap
#    persiapan data ke-8).
# 2. Model yang sekadar merekomendasikan film terpopuler bisa terlihat "cukup
#    baik" secara angka. Karena itu **baseline popularitas wajib disertakan**
#    sebagai pembanding — tanpa itu, kita tidak dapat membuktikan model benar-benar
#    mempelajari selera dan bukan sekadar mengikuti arus.
#
# Di sisi pengguna, distribusi juga sangat miring: median hanya sekitar 70 rating,
# sedangkan pengguna paling aktif memberi 2.698 rating. Model perlu bekerja baik
# untuk pengguna dengan riwayat pendek maupun panjang.

# %% [markdown]
# ### 3.3.3 Genre: Sebaran Katalog dan Kualitas Persepsi

# %%
genre_exploded = (movies.assign(genre=movies.genres.str.split('|'))
                        .explode('genre'))
jumlah_genre = genre_exploded.genre.value_counts()

rating_genre = (ratings.merge(genre_exploded[['movieId', 'genre']], on='movieId')
                       .groupby('genre')
                       .agg(rata_rating=('rating', 'mean'), jumlah_rating=('rating', 'size')))
rating_genre = rating_genre[rating_genre.jumlah_rating >= 500].sort_values('rata_rating')

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

ax[0].barh(jumlah_genre.index[::-1], jumlah_genre.values[::-1], color='#2e86ab')
ax[0].set_title('Jumlah Film per Genre (satu film bisa banyak genre)')
ax[0].set_xlabel('Jumlah film')
for i, v in enumerate(jumlah_genre.values[::-1]):
    ax[0].text(v + 40, i, str(v), va='center', fontsize=7)

ax[1].barh(rating_genre.index, rating_genre.rata_rating, color='#6a994e')
ax[1].axvline(ratings.rating.mean(), color='#d1495b', ls='--',
              label=f'rata-rata global {ratings.rating.mean():.2f}')
ax[1].set_xlim(3.0, 4.1)
ax[1].set_title('Rata-rata Rating per Genre (min. 500 rating)')
ax[1].set_xlabel('Rata-rata rating')
ax[1].legend(fontsize=8, loc='lower right')

plt.tight_layout()
simpan_gambar('fig_03_genre.png')
plt.show()

print('Jumlah kategori genre unik :', movies.genres.str.split('|').explode().nunique())
print('Film tanpa genre           :', int((movies.genres == '(no genres listed)').sum()))
print('Rata-rata genre per film   :', round(movies.genres.str.split('|').apply(len).mean(), 2))
print('\nLima genre terbanyak:')
print(jumlah_genre.head().to_string())
print('\nTiga genre dengan rata-rata rating tertinggi:')
print(rating_genre.tail(3).round(3).to_string())

# %% [markdown]
# **Insight 3 — Genre adalah fitur yang informatif tetapi kasar.**
#
# `Drama` dan `Comedy` mendominasi katalog dengan lebih dari 4.300 dan 3.700 film.
# Rata-rata satu film memiliki 2,27 genre, sehingga genre dapat dijadikan
# representasi konten yang bermakna. Perbedaan rata-rata rating antargenre juga
# nyata: `Film-Noir`, `War`, dan `Documentary` konsisten dinilai lebih tinggi,
# sedangkan `Horror` dan `Comedy` berada di bawah rata-rata global.
#
# Namun perlu dicatat sejak awal: hanya ada 19 kategori genre untuk 9.742 film.
# Artinya ribuan film berbagi kombinasi genre yang **persis sama** — misalnya
# ratusan film sekaligus berlabel `Comedy|Drama|Romance`. Keterbatasan resolusi
# inilah alasan proyek ini **memperkaya representasi konten dengan tag buatan
# pengguna** (lihat tahap persiapan data ke-4), sekaligus alasan teoretis mengapa
# *content-based filtering* diperkirakan kalah presisi dibanding *collaborative
# filtering* — dugaan yang nanti diuji secara kuantitatif pada bagian Evaluasi.

# %% [markdown]
# ### 3.3.4 Film Terpopuler dan Hubungan Popularitas–Kualitas

# %%
info_film = (ratings.groupby('movieId')
                    .agg(jumlah=('rating', 'size'), rata=('rating', 'mean'))
                    .merge(movies[['movieId', 'title']], on='movieId'))
teratas = info_film.sort_values('jumlah', ascending=False).head(15)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].barh(teratas.title.str.slice(0, 38)[::-1], teratas.jumlah[::-1], color='#2e86ab')
ax[0].set_title('15 Film dengan Rating Terbanyak')
ax[0].set_xlabel('Jumlah rating')
for i, (n, r) in enumerate(zip(teratas.jumlah[::-1], teratas.rata[::-1])):
    ax[0].text(n + 3, i, f'{n}  (rata {r:.2f})', va='center', fontsize=7)

ax[1].scatter(info_film.jumlah, info_film.rata, s=8, alpha=0.25, color='#2e86ab')
ax[1].set_xscale('log')
ax[1].set_xlabel('Jumlah rating yang diterima film (skala log)')
ax[1].set_ylabel('Rata-rata rating film')
ax[1].set_title('Popularitas vs Rata-rata Rating')
ax[1].axhline(ratings.rating.mean(), color='#d1495b', ls='--',
              label=f'rata-rata global {ratings.rating.mean():.2f}')
ax[1].legend(fontsize=8, loc='lower right')

plt.tight_layout()
simpan_gambar('fig_04_popularitas.png')
plt.show()

korelasi = np.corrcoef(np.log1p(info_film.jumlah), info_film.rata)[0, 1]
print(f'Korelasi Pearson antara log(jumlah rating) dan rata-rata rating: {korelasi:.3f}')
print('\nContoh film dengan rata-rata 5,0 tetapi hanya 1 rating:')
print(info_film[(info_film.rata == 5.0) & (info_film.jumlah == 1)]
      .head(5)[['title', 'jumlah', 'rata']].to_string(index=False))

# %% [markdown]
# **Insight 4 — Rata-rata rating mentah tidak layak dijadikan dasar peringkat.**
#
# Diagram sebar memperlihatkan bentuk corong yang khas: film dengan sedikit rating
# menyebar dari 0,5 sampai 5,0, sementara film dengan ratusan rating memusat pada
# kisaran 3,5–4,3. Terdapat banyak film beroleh rata-rata sempurna 5,0 hanya
# karena dinilai satu orang.
#
# Karena itu peringkat berbasis rata-rata mentah akan dipenuhi film antah-berantah.
# Model *Matrix Factorization* pada proyek ini menangani hal tersebut melalui
# **regularisasi pada suku bias item** ($b_i$), yang otomatis menarik estimasi
# film berdata sedikit mendekati rata-rata global — sebuah bentuk *shrinkage*.
# Korelasi positif yang lemah antara popularitas dan rata-rata rating juga
# menunjukkan bahwa popularitas dan kualitas bukan hal yang sama, sehingga
# rekomendasi tidak cukup hanya mengandalkan popularitas.

# %% [markdown]
# ### 3.3.5 Dimensi Waktu dan Tag Pengguna

# %%
ratings_waktu = ratings.assign(tahun=pd.to_datetime(ratings.timestamp, unit='s').dt.year)
per_tahun = ratings_waktu.groupby('tahun').size()

movies_tahun = movies.assign(
    tahun_rilis=pd.to_numeric(movies.title.str.extract(r'\((\d{4})\)\s*$')[0], errors='coerce'))
film_dekade = (movies_tahun.dropna(subset=['tahun_rilis'])
                           .assign(dekade=lambda d: (d.tahun_rilis // 10 * 10).astype(int))
                           .query('dekade >= 1930')
                           .groupby('dekade').size())

fig, ax = plt.subplots(1, 3, figsize=(16, 4))

ax[0].bar(per_tahun.index, per_tahun.values, color='#2e86ab')
ax[0].set_title('Jumlah Rating per Tahun Pemberian')
ax[0].set_xlabel('Tahun')
ax[0].set_ylabel('Jumlah rating')

ax[1].bar(film_dekade.index.astype(str), film_dekade.values, color='#6a994e')
ax[1].set_title('Jumlah Film per Dekade Rilis')
ax[1].set_xlabel('Dekade rilis')
ax[1].tick_params(axis='x', rotation=45)

tag_teratas = tags.tag.str.lower().value_counts().head(15)
ax[2].barh(tag_teratas.index[::-1], tag_teratas.values[::-1], color='#e07a5f')
ax[2].set_title('15 Tag Paling Sering Dipakai')
ax[2].set_xlabel('Frekuensi')

plt.tight_layout()
simpan_gambar('fig_05_waktu_dan_tag.png')
plt.show()

print('Rentang tahun pemberian rating :', per_tahun.index.min(), '-', per_tahun.index.max())
print('Jumlah tag                     :', f'{len(tags):,}')
print('Tag unik                       :', f'{tags.tag.nunique():,}')
print('Film yang memiliki tag         :', f'{tags.movieId.nunique():,}',
      f'({tags.movieId.nunique()/n_movie_total*100:.1f}% katalog)')
print('Pengguna yang pernah memberi tag:', tags.userId.nunique(), f'dari {n_user}')

# %% [markdown]
# **Insight 5 — Rentang waktu 22 tahun menuntut evaluasi yang jujur, dan tag adalah sinyal konten tambahan yang berharga meski jarang.**
#
# Rating terkumpul selama 22 tahun (1996–2018) dengan lonjakan pada beberapa
# periode. Selera penonton dan komposisi katalog jelas berubah sepanjang waktu.
# Karena itu proyek ini **tidak memakai pembagian data acak**, melainkan
# **pembagian temporal per pengguna**: model dilatih dengan riwayat awal setiap
# pengguna dan diuji pada tontonan mereka berikutnya. Skema ini meniru kondisi
# produksi yang sesungguhnya, ketika sistem harus menebak masa depan dan bukan
# menambal celah acak di masa lalu.
#
# Tag hanya mencakup 16% katalog dan berasal dari 58 pengguna saja, tetapi isinya
# jauh lebih spesifik daripada genre (`pixar`, `in netflix queue`, `atmospheric`,
# `based on a book`). Tag dipakai sebagai **pelengkap**, bukan pengganti genre,
# sehingga film tanpa tag tetap memperoleh representasi konten yang sah dari
# genrenya.
