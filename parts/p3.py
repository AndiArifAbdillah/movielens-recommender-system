# %% [markdown]
# ---
# # 4. Data Preparation
#
# Delapan tahap berikut dijalankan **berurutan**. Urutan ini penting karena
# keluaran satu tahap menjadi masukan tahap berikutnya. Setiap tahap disertai
# alasan mengapa ia diperlukan.
#
# | No | Tahap | Alasan |
# |---|---|---|
# | 1 | Verifikasi duplikat, nilai kosong, dan pemilihan berkas | Memastikan asumsi "satu pengguna–satu film–satu rating" benar-benar berlaku sebelum matriks interaksi dibentuk |
# | 2 | Ekstraksi tahun rilis dan pembersihan judul | Judul mentah menempelkan tahun sehingga menyulitkan pencarian judul dan analisis per dekade |
# | 3 | Parsing dan normalisasi genre | Genre tersimpan sebagai satu string dengan pemisah pipa dan harus diubah menjadi token yang dapat divektorkan |
# | 4 | Agregasi dan normalisasi tag per film | Tag tersebar sebagai banyak baris per film dan harus diringkas menjadi satu dokumen teks per film |
# | 5 | Pembentukan *content soup* dan vektorisasi TF-IDF | *Content-based filtering* memerlukan representasi numerik konten film |
# | 6 | *Encoding* `userId` dan `movieId` menjadi indeks kontinu | Nilai id asli tidak berurutan sehingga tidak dapat langsung dipakai sebagai indeks matriks |
# | 7 | Pembagian data secara temporal per pengguna | Menghindari kebocoran informasi masa depan dan meniru kondisi produksi |
# | 8 | Pembentukan matriks interaksi, *candidate pool*, dan himpunan item relevan | Menyiapkan struktur data yang dipakai bersama oleh ketiga model dan oleh protokol evaluasi |

# %% [markdown]
# ## Tahap 1 — Verifikasi Duplikat, Nilai Kosong, dan Pemilihan Berkas
#
# **Mengapa diperlukan.** Seluruh model pada proyek ini mengasumsikan setiap sel
# matriks interaksi berisi paling banyak satu rating. Jika terdapat pasangan
# `(userId, movieId)` ganda, satu interaksi akan terhitung dua kali dan bobot
# pembelajaran menjadi timpang. Verifikasi pada tahap 3.2 menunjukkan data sudah
# bersih, sehingga tahap ini bersifat **penegasan (defensif)**: kode tetap
# dijalankan agar jika suatu saat dataset diperbarui dan mengandung duplikat,
# masalahnya langsung terdeteksi. Berkas `links.csv` juga dikeluarkan dari alur
# karena tidak memuat informasi yang dipakai model.

# %%
jml_awal = len(ratings)
ratings = ratings.drop_duplicates(subset=['userId', 'movieId'], keep='last').reset_index(drop=True)
ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])
movies = movies.drop_duplicates(subset=['movieId']).reset_index(drop=True)

print(f'Rating sebelum verifikasi : {jml_awal:,}')
print(f'Rating setelah verifikasi : {len(ratings):,}  (dibuang: {jml_awal - len(ratings)})')
print(f'Film pada katalog         : {len(movies):,}')
print('Berkas yang dipakai       : ratings.csv, movies.csv, tags.csv')
print('Berkas yang tidak dipakai : links.csv (hanya berisi id eksternal IMDb/TMDB)')

# %% [markdown]
# ## Tahap 2 — Ekstraksi Tahun Rilis dan Pembersihan Judul
#
# **Mengapa diperlukan.** Kolom `title` menggabungkan dua informasi berbeda dalam
# satu string, misalnya `Toy Story (1995)`. Selama masih menyatu, tahun rilis
# tidak dapat dianalisis sebagai angka dan pencarian judul menjadi rapuh karena
# pengguna harus menuliskan tahunnya dengan tepat. Tahun rilis dipisahkan ke
# kolom `tahun_rilis`, sedangkan judul bersih disimpan pada `judul_bersih` dan
# dipakai sebagai antarmuka fungsi rekomendasi.

# %%
movies['tahun_rilis'] = pd.to_numeric(
    movies.title.str.extract(r'\((\d{4})\)\s*$')[0], errors='coerce')
movies['judul_bersih'] = movies.title.str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True).str.strip()

print('Film tanpa tahun rilis yang dapat diekstrak:', int(movies.tahun_rilis.isna().sum()))
print('Rentang tahun rilis:', int(movies.tahun_rilis.min()), '-', int(movies.tahun_rilis.max()))
print()
print(movies[['movieId', 'title', 'judul_bersih', 'tahun_rilis']].head(5).to_string(index=False))

# %% [markdown]
# ## Tahap 3 — Parsing dan Normalisasi Genre
#
# **Mengapa diperlukan.** TF-IDF bekerja pada dokumen teks yang tersusun atas
# token. Genre `Sci-Fi` dan `Film-Noir` mengandung tanda hubung, sedangkan
# `(no genres listed)` sebenarnya bukan genre melainkan penanda ketiadaan
# informasi. Tanpa normalisasi, `Sci-Fi` berisiko terpecah menjadi dua token
# (`sci` dan `fi`) dan bercampur dengan token lain, sementara `(no genres listed)`
# akan diperlakukan sebagai genre semu yang membuat 34 film saling mirip tanpa
# dasar. Karena itu setiap genre diubah menjadi satu token tunggal huruf kecil
# tanpa karakter khusus, dan penanda ketiadaan genre dihapus.

# %%
def normalkan_token(teks):
    """Mengubah frasa menjadi satu token tunggal: huruf kecil tanpa karakter non-alfanumerik."""
    return re.sub(r'[^a-z0-9]+', '', str(teks).lower())


movies['daftar_genre'] = (movies.genres
                          .str.replace('(no genres listed)', '', regex=False)
                          .str.split('|')
                          .apply(lambda g: [x for x in g if x]))
movies['token_genre'] = movies.daftar_genre.apply(
    lambda g: ' '.join(normalkan_token(x) for x in g))

print('Contoh hasil normalisasi genre:')
print(movies[['judul_bersih', 'genres', 'token_genre']].head(4).to_string(index=False))
print('\nContoh film tanpa genre (token menjadi string kosong):')
print(movies.loc[movies.token_genre == '', ['movieId', 'judul_bersih', 'genres']]
      .head(3).to_string(index=False))
print('\nJumlah film tanpa token genre:', int((movies.token_genre == '').sum()))

# %% [markdown]
# ## Tahap 4 — Agregasi dan Normalisasi Tag per Film
#
# **Mengapa diperlukan.** Pada `tags.csv`, satu film dapat muncul di banyak baris
# karena diberi tag oleh beberapa pengguna, sedangkan TF-IDF membutuhkan tepat
# **satu dokumen per film**. Seluruh tag milik satu film karena itu digabungkan
# menjadi satu string. Frasa seperti `based on a book` dinormalkan menjadi satu
# token utuh (`basedonabook`) supaya frasa tetap bermakna sebagai satu konsep dan
# tidak pecah menjadi kata umum `based`, `on`, `a`, `book` yang menghubungkan
# film-film yang sebenarnya tidak berkaitan. Tag sepanjang satu karakter dibuang
# karena tidak informatif.
#
# Frekuensi tag turut dipertahankan: bila lima pengguna menandai sebuah film
# dengan `pixar`, token tersebut muncul lima kali sehingga TF-IDF memberinya bobot
# lebih besar — mekanisme *term frequency* yang bekerja persis seperti pada teks
# biasa.

# %%
tags_bersih = tags.copy()
tags_bersih['token_tag'] = tags_bersih.tag.apply(normalkan_token)
tags_bersih = tags_bersih[tags_bersih.token_tag.str.len() > 1]

tag_per_film = (tags_bersih.groupby('movieId')
                           .token_tag.apply(lambda s: ' '.join(s))
                           .rename('token_tag'))

movies = movies.merge(tag_per_film, on='movieId', how='left')
movies['token_tag'] = movies.token_tag.fillna('')

print(f'Baris tag sebelum pembersihan : {len(tags):,}')
print(f'Baris tag setelah pembersihan : {len(tags_bersih):,}')
print(f'Film yang memperoleh tag      : {int((movies.token_tag != "").sum()):,} '
      f'dari {len(movies):,} film ({(movies.token_tag != "").mean()*100:.1f}%)')
print('\nContoh agregasi tag:')
print(movies.loc[movies.token_tag != '', ['judul_bersih', 'token_tag']]
      .head(3).to_string(index=False))

# %% [markdown]
# ## Tahap 5 — Pembentukan *Content Soup* dan Vektorisasi TF-IDF
#
# **Mengapa diperlukan.** *Content-based filtering* mengukur kemiripan antarfilm,
# dan kemiripan hanya dapat dihitung apabila setiap film sudah berbentuk vektor
# numerik. Token genre dan token tag digabung menjadi satu dokumen teks per film
# yang lazim disebut *content soup*, lalu diubah menjadi matriks TF-IDF.
#
# Dua keputusan penting pada tahap ini:
#
# 1. **Token genre digandakan dua kali.** Genre tersedia untuk hampir seluruh
#    katalog, sedangkan tag hanya untuk 16% film. Tanpa penggandaan, film yang
#    memiliki puluhan tag akan didominasi tag sehingga sinyal genrenya tenggelam,
#    dan kemiripannya dengan film tanpa tag menjadi tidak sebanding. Penggandaan
#    membuat genre tetap menjadi tulang punggung representasi, dengan tag sebagai
#    pelengkap yang mempertajam.
# 2. **Pembobotan TF-IDF, bukan sekadar hitungan.** Bobot IDF menekan token yang
#    muncul di mana-mana (`drama` ada pada 4.361 film sehingga hampir tidak
#    membedakan apa pun) dan mengangkat token langka yang justru khas
#    (`pixar`, `filmnoir`). Ini persis persoalan yang dirancang untuk diselesaikan
#    oleh IDF.
#
# Vektor tiap film kemudian dinormalisasi L2 agar hasil perkalian titik antardua
# vektor **setara dengan *cosine similarity***, sehingga kemiripan dapat dihitung
# lewat satu perkalian matriks yang efisien.

# %%
movies['content_soup'] = ((movies.token_genre + ' ') * 2 + movies.token_tag).str.strip()

print('Contoh content soup:')
for _, baris in movies.head(3).iterrows():
    print(f'  {baris.judul_bersih:<22} -> {baris.content_soup}')

vectorizer = TfidfVectorizer(token_pattern=r'\S+', min_df=1)
tfidf_semua = vectorizer.fit_transform(movies.content_soup)

print(f'\nDimensi matriks TF-IDF : {tfidf_semua.shape[0]:,} film x '
      f'{tfidf_semua.shape[1]:,} token')
print(f'Kepadatan matriks      : {tfidf_semua.nnz / np.prod(tfidf_semua.shape) * 100:.3f}%')
print(f'Contoh token kosakata  : {list(vectorizer.get_feature_names_out()[:12])}')

# Contoh bobot TF-IDF tertinggi pada satu film
contoh_idx = movies.index[movies.judul_bersih == 'Toy Story'][0]
baris = tfidf_semua[contoh_idx].toarray().ravel()
urut_bobot = np.argsort(-baris)[:6]
print('\nBobot TF-IDF tertinggi untuk "Toy Story":')
for j in urut_bobot:
    print(f'  {vectorizer.get_feature_names_out()[j]:<12} {baris[j]:.3f}')

# %% [markdown]
# ## Tahap 6 — *Encoding* `userId` dan `movieId` Menjadi Indeks Kontinu
#
# **Mengapa diperlukan.** `userId` berkisar 1–610 sedangkan `movieId` melompat-lompat
# hingga 193.609. Bila `movieId` dipakai langsung sebagai indeks matriks, akan
# terbentuk matriks berukuran 610 × 193.610 yang 95% kolomnya kosong dan memboroskan
# memori sekaligus komputasi. Setiap id karena itu dipetakan ke indeks berurutan
# 0..n−1, dan pemetaan baliknya disimpan agar hasil rekomendasi dapat diterjemahkan
# kembali menjadi judul film yang terbaca manusia.
#
# Katalog model dibatasi pada film yang **pernah muncul di data latih**. Film di
# luar itu tidak mungkin dipelajari oleh *collaborative filtering* dan justru akan
# menghasilkan vektor laten acak yang menyesatkan.

# %%
ratings = ratings.sort_values(['userId', 'timestamp']).reset_index(drop=True)

id_pengguna = np.sort(ratings.userId.unique())
pengguna_ke_idx = {uid: i for i, uid in enumerate(id_pengguna)}

print(f'Jumlah pengguna yang di-encode : {len(id_pengguna):,}')
print(f'Rentang userId asli            : {id_pengguna.min()} - {id_pengguna.max()}')
print(f'Rentang movieId asli           : {ratings.movieId.min()} - {ratings.movieId.max()}')
print('\nContoh pemetaan userId -> indeks:',
      {k: pengguna_ke_idx[k] for k in list(pengguna_ke_idx)[:5]})

# %% [markdown]
# ## Tahap 7 — Pembagian Data Secara Temporal per Pengguna
#
# **Mengapa diperlukan.** Pembagian acak akan membuat model belajar dari rating
# yang diberikan pengguna **pada tahun 2018** untuk memprediksi rating yang ia
# berikan **pada tahun 2000**. Kebocoran waktu semacam itu melambungkan skor
# evaluasi tetapi tidak pernah terjadi di dunia nyata, karena sistem produksi
# selalu memprediksi ke depan.
#
# Karena itu rating setiap pengguna diurutkan berdasarkan `timestamp`, lalu:
#
# - **80% pertama** riwayat setiap pengguna → data latih penuh (`train_penuh`)
# - **20% terakhir** riwayat setiap pengguna → **data uji** (`test`)
#
# Pembagian dilakukan **per pengguna**, bukan pada satu titik waktu global,
# supaya setiap pengguna tetap terwakili baik di data latih maupun data uji.
#
# `train_penuh` kemudian dibagi lagi dengan cara yang sama menjadi `train_fit`
# (80%) dan `validasi` (20%). **Seluruh penyetelan *hyperparameter* hanya
# menggunakan `validasi`**; data uji sama sekali tidak disentuh sampai model final
# terbentuk. Tanpa pemisahan ini, memilih *hyperparameter* berdasarkan skor uji
# sama saja dengan melatih model pada data uji dan angka evaluasi menjadi terlalu
# optimistis.

# %%
def bagi_temporal(df, porsi_latih=0.8):
    """Membagi rating tiap pengguna secara temporal: bagian awal untuk latih, akhir untuk uji."""
    df = df.sort_values(['userId', 'timestamp'])
    urutan = df.groupby('userId').cumcount()
    banyak = df.groupby('userId').movieId.transform('size')
    batas = np.ceil(banyak * porsi_latih)
    return df[urutan < batas].copy(), df[urutan >= batas].copy()


train_penuh, test = bagi_temporal(ratings, 0.8)
train_fit, validasi = bagi_temporal(train_penuh, 0.8)

ringkas = pd.DataFrame([
    {'subset': 'train_penuh', 'jumlah_rating': len(train_penuh), 'pengguna': train_penuh.userId.nunique(),
     'film': train_penuh.movieId.nunique(), 'porsi': len(train_penuh) / len(ratings)},
    {'subset': 'test', 'jumlah_rating': len(test), 'pengguna': test.userId.nunique(),
     'film': test.movieId.nunique(), 'porsi': len(test) / len(ratings)},
    {'subset': '  train_fit', 'jumlah_rating': len(train_fit), 'pengguna': train_fit.userId.nunique(),
     'film': train_fit.movieId.nunique(), 'porsi': len(train_fit) / len(ratings)},
    {'subset': '  validasi', 'jumlah_rating': len(validasi), 'pengguna': validasi.userId.nunique(),
     'film': validasi.movieId.nunique(), 'porsi': len(validasi) / len(ratings)},
])
ringkas['porsi'] = (ringkas.porsi * 100).round(1).astype(str) + '%'
print(ringkas.to_string(index=False))

print('\nVerifikasi tidak ada kebocoran waktu (waktu latih < waktu uji untuk tiap pengguna):')
maks_latih = train_penuh.groupby('userId').timestamp.max()
min_uji = test.groupby('userId').timestamp.min()
gabung = pd.concat([maks_latih.rename('maks_latih'), min_uji.rename('min_uji')], axis=1).dropna()
print(f'  Jumlah pengguna yang melanggar urutan waktu: {int((gabung.maks_latih > gabung.min_uji).sum())}')

# %% [markdown]
# ## Tahap 8 — Matriks Interaksi, *Candidate Pool*, dan Item Relevan
#
# **Mengapa diperlukan.** Tahap ini menyiapkan tiga struktur data yang dipakai
# bersama oleh seluruh model sehingga perbandingan antarmodel benar-benar setara.
#
# 1. **Matriks interaksi renggang (*sparse*).** Matriks 610 × 8.246 dalam bentuk
#    padat memerlukan sekitar 40 MB, sementara hanya 1,7% selnya terisi. Format
#    `csr_matrix` menyimpan yang terisi saja sehingga hemat memori dan mempercepat
#    perkalian matriks.
# 2. ***Candidate pool*.** Film yang hanya memiliki kurang dari 5 rating pada data
#    latih tidak memiliki bukti kolaboratif yang memadai; merekomendasikannya sama
#    saja dengan menebak. Kumpulan kandidat karena itu dibatasi pada film dengan
#    minimal 5 rating. Pembatasan ini **berlaku sama bagi ketiga model dan
#    baseline**, sehingga perbandingan tetap adil dan bukan sekadar menguntungkan
#    salah satu pendekatan.
# 3. **Himpunan item relevan.** Evaluasi peringkat membutuhkan definisi "benar"
#    yang tegas. Sebuah film dianggap **relevan** bagi seorang pengguna apabila
#    pada data uji ia memberi rating **≥ 4,0** — ambang yang sejalan dengan
#    Insight 1, yaitu bahwa 4,0 adalah nilai penanda "film ini saya sukai".

# %%
katalog = np.sort(train_penuh.movieId.unique())
film_ke_idx = {mid: i for i, mid in enumerate(katalog)}
idx_ke_film = {i: mid for mid, i in film_ke_idx.items()}

N_PENGGUNA = len(id_pengguna)
N_FILM = len(katalog)

# Metadata film diselaraskan dengan urutan katalog
meta = movies.set_index('movieId').loc[katalog].reset_index()
tfidf = normalize(vectorizer.transform(meta.content_soup))   # normalisasi L2 -> cosine = dot product

# Matriks interaksi data latih
baris = train_penuh.userId.map(pengguna_ke_idx).to_numpy()
kolom = train_penuh.movieId.map(film_ke_idx).to_numpy()
matriks_latih = sp.csr_matrix((train_penuh.rating.to_numpy(float), (baris, kolom)),
                              shape=(N_PENGGUNA, N_FILM))
sudah_ditonton = np.zeros((N_PENGGUNA, N_FILM), dtype=bool)
sudah_ditonton[baris, kolom] = True

# Candidate pool: film dengan minimal 5 rating pada data latih
MIN_RATING_KANDIDAT = 5
jumlah_rating_film = np.asarray((matriks_latih > 0).sum(axis=0)).ravel()
pool = np.where(jumlah_rating_film >= MIN_RATING_KANDIDAT)[0]
pool_set = set(pool.tolist())

# Himpunan item relevan pada data uji
AMBANG_SUKA = 4.0
test_relevan = test[test.movieId.isin(film_ke_idx) & (test.rating >= AMBANG_SUKA)].copy()
test_relevan['idx_film'] = test_relevan.movieId.map(film_ke_idx)
test_relevan = test_relevan[test_relevan.idx_film.isin(pool_set)]
relevan_uji = {pengguna_ke_idx[u]: set(g.idx_film) for u, g in test_relevan.groupby('userId')}

meta['jumlah_rating_latih'] = jumlah_rating_film
meta['dalam_pool'] = np.isin(np.arange(N_FILM), pool)

print(f'Ukuran matriks interaksi     : {N_PENGGUNA:,} pengguna x {N_FILM:,} film')
print(f'Elemen terisi                : {matriks_latih.nnz:,} '
      f'({matriks_latih.nnz / (N_PENGGUNA * N_FILM) * 100:.2f}%)')
print(f'Memori format sparse (CSR)   : {matriks_latih.data.nbytes / 1024**2:.2f} MB')
print(f'Memori jika format padat     : {N_PENGGUNA * N_FILM * 8 / 1024**2:.2f} MB')
print(f'\nUkuran candidate pool        : {len(pool):,} film '
      f'({len(pool)/N_FILM*100:.1f}% katalog latih)')
print(f'Cakupan rating pool          : '
      f'{jumlah_rating_film[pool].sum() / jumlah_rating_film.sum() * 100:.1f}% dari rating data latih')
print(f'\nPengguna yang dapat dievaluasi: {len(relevan_uji):,} dari {N_PENGGUNA:,}')
print(f'Rata-rata item relevan/pengguna: '
      f'{np.mean([len(v) for v in relevan_uji.values()]):.1f}')

# %% [markdown]
# ### Struktur Padanan untuk Data Validasi
#
# Seluruh struktur di atas dibentuk ulang menggunakan `train_fit` dan `validasi`.
# Padanan ini diperlukan agar penyetelan *hyperparameter* dinilai dengan protokol
# yang persis sama dengan pengujian akhir, hanya pada data yang berbeda.

# %%
# Padanan struktur yang sama untuk data validasi (dipakai saat penyetelan hyperparameter)
baris_fit = train_fit.userId.map(pengguna_ke_idx).to_numpy()
kolom_fit = train_fit.movieId.map(film_ke_idx).to_numpy()
sudah_ditonton_fit = np.zeros((N_PENGGUNA, N_FILM), dtype=bool)
sudah_ditonton_fit[baris_fit, kolom_fit] = True

jumlah_rating_fit = np.asarray((sp.csr_matrix(
    (np.ones(len(train_fit)), (baris_fit, kolom_fit)), shape=(N_PENGGUNA, N_FILM)) > 0).sum(axis=0)).ravel()
pool_fit = np.where(jumlah_rating_fit >= MIN_RATING_KANDIDAT)[0]
pool_fit_set = set(pool_fit.tolist())

val_relevan = validasi[validasi.movieId.isin(film_ke_idx) & (validasi.rating >= AMBANG_SUKA)].copy()
val_relevan['idx_film'] = val_relevan.movieId.map(film_ke_idx)
val_relevan = val_relevan[val_relevan.idx_film.isin(pool_fit_set)]
relevan_validasi = {pengguna_ke_idx[u]: set(g.idx_film) for u, g in val_relevan.groupby('userId')}

print(f'Candidate pool validasi       : {len(pool_fit):,} film')
print(f'Pengguna pada evaluasi validasi: {len(relevan_validasi):,}')

# %% [markdown]
# ### Ringkasan Hasil Data Preparation
#
# Setelah delapan tahap di atas, tersedia:
#
# - `meta` — metadata 8.246 film pada katalog latih, lengkap dengan judul bersih,
#   tahun rilis, token genre, token tag, dan jumlah rating.
# - `tfidf` — matriks TF-IDF ternormalisasi berukuran 8.246 × 1.461 sebagai
#   masukan *content-based filtering*.
# - `matriks_latih`, `sudah_ditonton` — matriks interaksi dan penanda film yang
#   sudah ditonton, sebagai masukan *collaborative filtering*.
# - `train_fit` / `validasi` / `train_penuh` / `test` — pembagian data temporal
#   berjenjang untuk penyetelan dan pengujian akhir.
# - `pool`, `relevan_uji` — protokol evaluasi peringkat yang identik bagi semua model.
