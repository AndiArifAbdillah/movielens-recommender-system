# %% [markdown]
# ---
# ## 5.3 Model 3 — Collaborative Filtering dengan *Implicit ALS*
#
# ### Mengapa model kedua ini diperlukan
#
# *Matrix factorization* pada 5.2 dilatih untuk **menebak angka rating**. Padahal
# yang benar-benar dibutuhkan sistem rekomendasi adalah **mengurutkan** film:
# yang penting sepuluh judul teratas tepat, bukan apakah taksirannya 4,3 atau 4,1.
# Kedua tujuan itu tidak identik. Model yang mahir menebak angka bisa saja
# menempatkan film asing berprediksi 4,6 di atas film yang benar-benar akan
# ditonton pengguna, karena galat pada film asing tidak pernah dihukum — film
# tersebut memang tidak ada pada data teramati.
#
# *Implicit ALS* (Hu, Koren & Volinsky, 2008) menyerang persoalan dari sisi yang
# berbeda. Rating diterjemahkan menjadi **sinyal biner**: $p_{ui} = 1$ bila
# pengguna memberi rating ≥ 4 (menyukai), dan $p_{ui} = 0$ untuk **seluruh
# pasangan lain** — termasuk film yang belum pernah dilihat. Setiap pengamatan
# diberi bobot keyakinan
#
# $$c_{ui} = 1 + \alpha \, p_{ui}$$
#
# sehingga interaksi yang teramati ditimbang $1 + \alpha$ sedangkan yang tidak
# teramati tetap ikut dihitung dengan bobot 1. Perbedaan inilah kuncinya: model
# **mempelajari pasangan negatif juga**, sehingga tahu bahwa film populer yang
# dilewatkan pengguna kemungkinan besar memang tidak diminati.
#
# Fungsi objektifnya mencakup seluruh $610 \times 8.246$ pasangan:
#
# $$\min_{x,y} \sum_{u,i} c_{ui}\left(p_{ui} - \mathbf{x}_u^{\top}\mathbf{y}_i\right)^2 + \lambda\left(\sum_u \lVert \mathbf{x}_u \rVert^2 + \sum_i \lVert \mathbf{y}_i \rVert^2\right)$$
#
# Menjumlahkan lima juta pasangan pada tiap langkah SGD jelas tidak praktis.
# Solusinya adalah *alternating least squares*: bila $Y$ dianggap tetap, fungsi
# objektif menjadi kuadratik terhadap $X$ dan memiliki solusi tertutup
#
# $$\mathbf{x}_u = \left(Y^{\top}Y + Y^{\top}(C^u - I)Y + \lambda I\right)^{-1} Y^{\top} C^u \mathbf{p}(u)$$
#
# lalu perannya ditukar untuk memperbarui $Y$. Suku $Y^{\top}Y$ dihitung **satu
# kali** untuk semua pengguna, sedangkan $Y^{\top}(C^u - I)Y$ hanya melibatkan
# film yang benar-benar berinteraksi dengan pengguna $u$. Dengan trik itu, satu
# iterasi ALS hanya berbiaya sepersekian detik untuk data ini.

# %%
def latih_ials(data_latih, n_faktor=32, reg=2.0, alpha=10.0, iterasi=15,
               ambang=AMBANG_SUKA, seed=SEED):
    """Melatih implicit ALS (Hu, Koren & Volinsky 2008) dari interaksi 'disukai'."""
    positif = data_latih[data_latih.rating >= ambang]
    C = sp.csr_matrix((np.ones(len(positif)),
                       (positif.userId.map(pengguna_ke_idx), positif.movieId.map(film_ke_idx))),
                      shape=(N_PENGGUNA, N_FILM))
    C_pengguna, C_film = C.tocsr(), C.tocsc()

    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.01, (N_PENGGUNA, n_faktor))
    Y = rng.normal(0, 0.01, (N_FILM, n_faktor))
    I_reg = reg * np.eye(n_faktor)

    for _ in range(iterasi):
        YtY = Y.T @ Y + I_reg                     # dihitung sekali untuk semua pengguna
        for u in range(N_PENGGUNA):
            item = C_pengguna.indices[C_pengguna.indptr[u]:C_pengguna.indptr[u + 1]]
            if len(item) == 0:
                X[u] = 0
                continue
            Yi = Y[item]
            A = YtY + alpha * (Yi.T @ Yi)         # Y^T Y + Y^T (C^u - I) Y + lambda I
            b = (1 + alpha) * Yi.sum(axis=0)      # Y^T C^u p(u)
            X[u] = np.linalg.solve(A, b)

        XtX = X.T @ X + I_reg                     # dihitung sekali untuk semua film
        for i in range(N_FILM):
            pengguna = C_film.indices[C_film.indptr[i]:C_film.indptr[i + 1]]
            if len(pengguna) == 0:
                Y[i] = 0
                continue
            Xu = X[pengguna]
            A = XtX + alpha * (Xu.T @ Xu)
            b = (1 + alpha) * Xu.sum(axis=0)
            Y[i] = np.linalg.solve(A, b)

    return {'X': X, 'Y': Y,
            'params': {'n_faktor': n_faktor, 'reg': reg, 'alpha': alpha, 'iterasi': iterasi}}


print('Fungsi pelatihan Implicit ALS siap.')

# %% [markdown]
# ### Penyetelan *hyperparameter* iALS pada data validasi
#
# Karena iALS dinilai berdasarkan mutu urutan dan bukan galat angka, kriteria
# pemilihan yang dipakai adalah **NDCG@10 pada data validasi**.

# %%
grid_ials = [
    {'n_faktor': 16, 'reg': 1.0, 'alpha': 10},
    {'n_faktor': 32, 'reg': 1.0, 'alpha': 10},
    {'n_faktor': 32, 'reg': 2.0, 'alpha': 10},
    {'n_faktor': 32, 'reg': 1.0, 'alpha': 40},
    {'n_faktor': 64, 'reg': 1.0, 'alpha': 10},
    {'n_faktor': 64, 'reg': 0.1, 'alpha': 40},
]

waktu_mulai = time.time()
hasil_grid_ials = []
for konf in grid_ials:
    model_uji = latih_ials(train_fit, iterasi=15, **konf)
    metrik = evaluasi_ranking(model_uji['X'] @ model_uji['Y'].T,
                              relevan_validasi, sudah_ditonton_fit, pool_fit, K=10)
    hasil_grid_ials.append({**konf, **metrik})
    print(f"k={konf['n_faktor']:>3} reg={konf['reg']:<4} alpha={konf['alpha']:<3} -> "
          f"NDCG@10 validasi {metrik['NDCG@10']:.4f}  ({time.time() - waktu_mulai:.0f} detik)")

tabel_grid_ials = pd.DataFrame(hasil_grid_ials).sort_values('NDCG@10', ascending=False).reset_index(drop=True)
print('\n--- Hasil penyetelan Implicit ALS ---')
print(tabel_grid_ials.round(4).to_string(index=False))

konfigurasi_ials = tabel_grid_ials.iloc[0]
print(f"\nKonfigurasi terpilih: k={int(konfigurasi_ials.n_faktor)}, "
      f"reg={konfigurasi_ials.reg}, alpha={int(konfigurasi_ials.alpha)}")

# %% [markdown]
# ### Pelatihan Model iALS Final
#
# Konfigurasi terpilih dilatih ulang pada `train_penuh` — gabungan `train_fit` dan
# `validasi` — agar model final memanfaatkan seluruh data yang tersedia sebelum
# diuji.

# %%
model_ials = latih_ials(train_penuh,
                        n_faktor=int(konfigurasi_ials.n_faktor),
                        reg=float(konfigurasi_ials.reg),
                        alpha=float(konfigurasi_ials.alpha),
                        iterasi=15)
skor_ials = model_ials['X'] @ model_ials['Y'].T

print('Model iALS final selesai dilatih pada train_penuh.')
print('Parameter:', model_ials['params'])
print('Dimensi faktor pengguna :', model_ials['X'].shape)
print('Dimensi faktor film     :', model_ials['Y'].shape)

# %% [markdown]
# ---
# ## 5.4 Baseline — Rekomendasi Berbasis Popularitas
#
# *Baseline* ini merekomendasikan film yang paling banyak dirating pada data
# latih, sama untuk semua orang. Ia tidak personal sama sekali, tetapi justru
# karena itulah ia menjadi pembanding yang tepat: sebuah model personal baru layak
# disebut berhasil apabila mampu **melampaui** daftar populer ini.

# %%
skor_pop = np.tile(jumlah_rating_film.astype(float), (N_PENGGUNA, 1))

print('Sepuluh film terpopuler pada data latih (isi rekomendasi baseline):')
urut_pop = np.argsort(-jumlah_rating_film)[:10]
print(meta.iloc[urut_pop][['judul_bersih', 'tahun_rilis', 'jumlah_rating_latih']]
      .to_string(index=False))

# %% [markdown]
# ---
# ## 5.5 Hasil: Top-N Recommendation
#
# Bagian ini menampilkan keluaran akhir sistem, yaitu **daftar 10 film teratas**
# untuk seorang pengguna nyata dari keempat pendekatan. Agar hasilnya dapat
# dinilai secara jujur, dipilih pengguna yang memiliki cukup riwayat pada data
# latih sekaligus cukup banyak film relevan pada data uji, sehingga ketepatan
# rekomendasi benar-benar dapat diperiksa.

# %%
def rekomendasi_topn(skor, idx_pengguna, top_n=10, kandidat=None, sudah=None, nama_kolom='skor'):
    """Menghasilkan daftar top-N film untuk seorang pengguna dari matriks skor."""
    kandidat = pool if kandidat is None else kandidat
    sudah = sudah_ditonton if sudah is None else sudah

    s = np.full(N_FILM, -np.inf)
    s[kandidat] = skor[idx_pengguna, kandidat].astype(float)
    s[sudah[idx_pengguna]] = -np.inf
    urutan = np.lexsort((-jumlah_rating_film, -s))[:top_n]

    hasil = meta.iloc[urutan][['judul_bersih', 'tahun_rilis', 'genres']].copy()
    hasil.insert(0, 'peringkat', np.arange(1, top_n + 1))
    hasil[nama_kolom] = s[urutan].round(4)
    hasil['relevan?'] = ['YA' if i in relevan_uji.get(idx_pengguna, set()) else '-' for i in urutan]
    return hasil.reset_index(drop=True)


# Memilih pengguna contoh secara deterministik
kandidat_pengguna = [(u, len(v)) for u, v in relevan_uji.items()
                     if 80 <= sudah_ditonton[u].sum() <= 200 and len(v) >= 12]
idx_contoh = sorted(kandidat_pengguna, key=lambda x: (-x[1], x[0]))[0][0]
user_contoh = int(id_pengguna[idx_contoh])

riwayat_contoh = (train_penuh[train_penuh.userId == user_contoh]
                  .merge(meta[['movieId', 'judul_bersih', 'tahun_rilis', 'genres']], on='movieId')
                  .sort_values('rating', ascending=False))

print(f'PENGGUNA CONTOH: userId = {user_contoh} (indeks internal {idx_contoh})')
print(f'  Jumlah film yang ditonton pada data latih : {int(sudah_ditonton[idx_contoh].sum())}')
print(f'  Rata-rata rating yang ia berikan          : {riwayat_contoh.rating.mean():.2f}')
print(f'  Jumlah film relevan pada data uji         : {len(relevan_uji[idx_contoh])}')
print('\nSepuluh film favoritnya pada data latih:')
print(riwayat_contoh.head(10)[['judul_bersih', 'tahun_rilis', 'rating', 'genres']].to_string(index=False))

# %% [markdown]
# ### Keluaran 1 — Baseline Popularitas
#
# Daftar ini identik untuk seluruh pengguna karena tidak dipersonalisasi sama
# sekali. Kolom `relevan?` menandai film yang benar-benar ditonton dan disukai
# pengguna pada data uji.

# %%
print('=' * 104)
print(f'TOP-10 REKOMENDASI UNTUK userId={user_contoh}  —  BASELINE POPULARITAS')
print('=' * 104)
print(rekomendasi_topn(skor_pop, idx_contoh, nama_kolom='jml_rating').to_string(index=False))

# %% [markdown]
# ### Keluaran 2 — Content-Based Filtering
#
# Film diperingkat berdasarkan kedekatan konten dengan profil selera pengguna.

# %%
print('=' * 104)
print(f'TOP-10 REKOMENDASI UNTUK userId={user_contoh}  —  MODEL 1: CONTENT-BASED FILTERING')
print('=' * 104)
print(rekomendasi_topn(skor_cbf, idx_contoh, nama_kolom='kemiripan').to_string(index=False))

# %% [markdown]
# ### Keluaran 3 — Matrix Factorization (SGD)
#
# Film diperingkat berdasarkan prediksi nilai rating $\hat{r}_{ui}$.

# %%
print('=' * 104)
print(f'TOP-10 REKOMENDASI UNTUK userId={user_contoh}  —  MODEL 2: MATRIX FACTORIZATION (SGD)')
print('=' * 104)
print(rekomendasi_topn(skor_mf, idx_contoh, nama_kolom='prediksi_rating').to_string(index=False))

# %% [markdown]
# ### Keluaran 4 — Implicit ALS
#
# Film diperingkat berdasarkan skor preferensi $\mathbf{x}_u^{	op}\mathbf{y}_i$.

# %%
print('=' * 104)
print(f'TOP-10 REKOMENDASI UNTUK userId={user_contoh}  —  MODEL 3: IMPLICIT ALS')
print('=' * 104)
print(rekomendasi_topn(skor_ials, idx_contoh, nama_kolom='skor_preferensi').to_string(index=False))

# %% [markdown]
# **Pembacaan keluaran top-N.** Empat daftar di atas memperlihatkan karakter
# masing-masing pendekatan dengan gamblang. Pengguna contoh adalah penggemar film
# laga, kriminal, dan perang klasik (*The Godfather*, *Braveheart*, *The Matrix*,
# *Saving Private Ryan*).
#
# - **Popularitas** memberi daftar yang sama untuk siapa pun. Isinya film-film
#   besar yang memang banyak ditonton, sehingga sesekali tepat secara kebetulan —
#   satu tepat sasaran dari sepuluh.
# - **CBF** menghasilkan daftar yang paling seragam: kesepuluh judulnya berbagi
#   kombinasi genre yang nyaris identik (`Action|Adventure|Sci-Fi|Thriller`) dengan
#   skor kemiripan yang seri persis di angka 0,879 dan 0,853. Tidak satu pun tepat
#   sasaran. Inilah gejala ***over-specialization***: sistem terus menawarkan
#   varian dari hal yang sama, sambil kehilangan kemampuan membedakan mana yang
#   benar-benar bermutu di antara ratusan film bergenre serupa.
# - **MF** menyodorkan sepuluh film klasik era 1940–1960-an dengan prediksi rating
#   di atas 5,0, dan tidak satu pun tepat sasaran. Hasil ini adalah kelanjutan
#   langsung dari temuan pada bias film: model yang dioptimasi untuk menebak angka
#   mengangkat film berdata sedikit yang kebetulan dinilai sangat tinggi oleh
#   segelintir orang. Prediksi rating melampaui 5,0 karena skor sengaja tidak
#   dipangkas pada tahap pemeringkatan — pemangkasan tidak mengubah urutan.
# - **iALS** menghasilkan daftar yang paling seimbang: judul-judul yang cukup
#   dikenal untuk masuk akal, tetap sesuai selera laga dan misteri pengguna, dan
#   menghasilkan **empat penanda `YA`** — film-film yang benar-benar ia tonton dan
#   sukai pada periode berikutnya.
#
# Kesan kualitatif ini akan diuji secara kuantitatif pada bab berikutnya.
