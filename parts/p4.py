# %% [markdown]
# ---
# # 5. Modeling and Result
#
# Empat sistem dibangun dan dibandingkan:
#
# | Kode | Model | Pendekatan | Sinyal yang dipakai |
# |---|---|---|---|
# | **POP** | Popularitas | *Baseline* non-personal | Jumlah rating pada data latih |
# | **CBF** | *Content-Based Filtering* | TF-IDF + *cosine similarity* | Genre dan tag film |
# | **MF** | *Matrix Factorization* (SGD) | *Collaborative filtering* eksplisit | Nilai rating 0,5–5,0 |
# | **iALS** | *Implicit Alternating Least Squares* | *Collaborative filtering* implisit | Interaksi "disukai" (rating ≥ 4) |
#
# POP disertakan bukan sebagai kandidat solusi, melainkan sebagai **garis batas
# kelayakan**: model personal apa pun yang tidak mampu mengungguli daftar film
# terpopuler belum membuktikan dirinya mempelajari selera pengguna.

# %% [markdown]
# ## 5.0 Fungsi Bantu Evaluasi
#
# Fungsi berikut dipakai lebih dahulu pada tahap penyetelan *hyperparameter*
# (menggunakan data validasi) dan dipakai ulang pada Bab 6 untuk pengujian akhir.
# Formula lengkap tiap metrik beserta cara kerjanya dijelaskan pada Bab 6.

# %%
def evaluasi_ranking(skor, relevan, sudah, kandidat, K=10):
    """Menghitung Precision@K, Recall@K, NDCG@K, dan Coverage@K dari matriks skor.

    skor     : matriks (n_pengguna x n_film) berisi skor preferensi
    relevan  : dict {indeks_pengguna: himpunan indeks film relevan}
    sudah    : matriks boolean film yang sudah ditonton pada data latih
    kandidat : array indeks film yang boleh direkomendasikan (candidate pool)
    """
    presisi, recall, ndcg = [], [], []
    terekomendasi = set()
    diskon = 1 / np.log2(np.arange(2, K + 2))
    idcg_kumulatif = np.cumsum(diskon)

    for u, item_relevan in relevan.items():
        s = np.full(skor.shape[1], -np.inf)
        s[kandidat] = skor[u, kandidat].astype(float)
        s[sudah[u]] = -np.inf                       # jangan rekomendasikan yang sudah ditonton
        teratas = np.argpartition(-s, K)[:K]
        teratas = teratas[np.argsort(-s[teratas])]  # urutkan K teratas
        terekomendasi.update(teratas.tolist())

        hit = np.array([1.0 if i in item_relevan else 0.0 for i in teratas])
        presisi.append(hit.sum() / K)
        recall.append(hit.sum() / len(item_relevan))
        dcg = float((hit * diskon).sum())
        idcg = idcg_kumulatif[min(len(item_relevan), K) - 1]
        ndcg.append(dcg / idcg)

    return {
        f'Precision@{K}': float(np.mean(presisi)),
        f'Recall@{K}': float(np.mean(recall)),
        f'NDCG@{K}': float(np.mean(ndcg)),
        f'Coverage@{K}': len(terekomendasi) / len(kandidat),
    }


def evaluasi_rating(y_benar, y_prediksi):
    """Menghitung RMSE dan MAE untuk tugas prediksi nilai rating."""
    galat = np.asarray(y_benar, dtype=float) - np.asarray(y_prediksi, dtype=float)
    return {'RMSE': float(np.sqrt((galat ** 2).mean())), 'MAE': float(np.abs(galat).mean())}


print('Fungsi bantu evaluasi siap digunakan.')

# %% [markdown]
# ---
# ## 5.1 Model 1 — Content-Based Filtering
#
# ### Cara kerja
#
# Setiap film sudah direpresentasikan sebagai vektor TF-IDF ternormalisasi
# $\mathbf{v}_i \in \mathbb{R}^{1461}$ (Tahap 5 persiapan data). Kemiripan dua
# film diukur dengan ***cosine similarity***:
#
# $$\text{sim}(i, j) = \cos(\theta) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\lVert \mathbf{v}_i \rVert \, \lVert \mathbf{v}_j \rVert}$$
#
# Karena seluruh vektor sudah dinormalisasi L2 ($\lVert \mathbf{v} \rVert = 1$),
# penyebutnya bernilai 1 sehingga kemiripan cukup dihitung sebagai perkalian titik.
# Nilainya berkisar 0 (tidak berbagi token sama sekali) sampai 1 (representasi
# konten identik).
#
# Model ini dipakai dalam dua mode.
#
# **Mode A — rekomendasi item-ke-item.** Diberikan satu film, sistem menampilkan
# N film paling mirip. Inilah mekanisme di balik kolom "karena Anda menonton …".
#
# **Mode B — rekomendasi personal.** Profil pengguna $\mathbf{p}_u$ dibangun
# sebagai rata-rata berbobot vektor konten seluruh film yang ia sukai (rating ≥ 4)
# pada data latih, dengan rating sebagai bobot:
#
# $$\mathbf{p}_u = \frac{\sum_{i \in \mathcal{L}_u} r_{ui} \, \mathbf{v}_i}{\left\lVert \sum_{i \in \mathcal{L}_u} r_{ui} \, \mathbf{v}_i \right\rVert}$$
#
# dengan $\mathcal{L}_u$ himpunan film yang disukai pengguna $u$. Skor sebuah film
# kandidat adalah $\text{skor}(u, i) = \mathbf{p}_u \cdot \mathbf{v}_i$.

# %%
judul_ke_idx = {judul: i for i, judul in enumerate(meta.judul_bersih)}


def rekomendasi_serupa(judul, top_n=10, min_rating=5):
    """Mode A: mencari film yang kontennya paling mirip dengan judul yang diberikan."""
    if judul not in judul_ke_idx:
        kandidat = [j for j in judul_ke_idx if judul.lower() in j.lower()][:5]
        raise KeyError(f'Judul "{judul}" tidak ditemukan. Saran: {kandidat}')

    i = judul_ke_idx[judul]
    kemiripan = np.asarray((tfidf[i] @ tfidf.T).todense()).ravel()
    kemiripan[i] = -1                                                  # jangan rekomendasikan dirinya
    kemiripan[meta.jumlah_rating_latih.to_numpy() < min_rating] = -1   # buang film tanpa bukti minat

    # Skor kemiripan banyak yang seri; popularitas dipakai sebagai pemecah seri
    urutan = np.lexsort((-meta.jumlah_rating_latih.to_numpy(), -kemiripan))[:top_n]
    hasil = meta.iloc[urutan][['judul_bersih', 'tahun_rilis', 'genres', 'jumlah_rating_latih']].copy()
    hasil.insert(0, 'peringkat', np.arange(1, len(urutan) + 1))
    hasil['kemiripan'] = kemiripan[urutan].round(3)
    return hasil.reset_index(drop=True)


for judul_uji in ['Toy Story', 'Godfather, The', 'Silence of the Lambs, The']:
    asal = meta.loc[judul_ke_idx[judul_uji]]
    print('=' * 100)
    print(f'TOP-10 FILM MIRIP DENGAN: {judul_uji} ({int(asal.tahun_rilis)})  |  genre: {asal.genres}')
    print('=' * 100)
    print(rekomendasi_serupa(judul_uji).to_string(index=False))
    print()

# %% [markdown]
# **Pembacaan hasil Mode A.** Sistem menempatkan *A Bug's Life*, *Toy Story 2*,
# dan *Monsters, Inc.* sebagai film termirip dengan *Toy Story* — ketiganya
# animasi keluarga produksi Pixar, dan dua di antaranya terhubung lewat tag
# `pixar`, bukan sekadar lewat genre. Untuk *The Godfather*, sistem
# mengembalikan *Goodfellas*, *Casino*, dan *Donnie Brasco* dengan kemiripan
# 1,000: keempatnya berbagi kombinasi genre `Crime|Drama` yang persis sama.
# Nilai kemiripan sempurna ini sekaligus memperlihatkan **keterbatasan mendasar**
# pendekatan konten pada dataset ini — metadata tidak cukup halus untuk
# membedakan film-film dalam kombinasi genre yang sama, sehingga banyak skor
# yang seri dan urutan akhirnya ditentukan pemecah seri.

# %%
def matriks_konten(kolom):
    """Membangun matriks TF-IDF ternormalisasi dari sebuah kolom teks pada meta."""
    vec = TfidfVectorizer(token_pattern=r'\S+')
    return normalize(vec.fit_transform(meta[kolom].fillna('')))


def bangun_skor_cbf(data_latih, matriks_isi=None, gunakan_deviasi=False, ambang=AMBANG_SUKA):
    """Menghasilkan matriks skor CBF (n_pengguna x n_film) dari profil konten pengguna.

    gunakan_deviasi=False : profil dibangun hanya dari film yang disukai (rating >= ambang),
                            dengan rating sebagai bobot.
    gunakan_deviasi=True  : seluruh rating dipakai dengan bobot (rating - rata-rata pengguna),
                            sehingga film yang tidak disukai memberi kontribusi negatif.
    """
    X = tfidf if matriks_isi is None else matriks_isi
    if gunakan_deviasi:
        dipakai = data_latih
        bobot = data_latih.rating - data_latih.groupby('userId').rating.transform('mean')
    else:
        dipakai = data_latih[data_latih.rating >= ambang]
        bobot = dipakai.rating
    W = sp.csr_matrix((bobot.to_numpy(float),
                       (dipakai.userId.map(pengguna_ke_idx), dipakai.movieId.map(film_ke_idx))),
                      shape=(N_PENGGUNA, N_FILM))
    profil = normalize(W @ X)                 # rata-rata berbobot, lalu dinormalisasi L2
    return np.asarray((profil @ X.T).todense())


print('Fungsi pembentuk profil dan skor CBF siap.')

# %% [markdown]
# ### Pemilihan varian representasi konten berdasarkan data validasi
#
# Sebelum model CBF final ditetapkan, tiga varian dibandingkan pada **data
# validasi** agar keputusan rancangan tidak mencemari angka pengujian akhir:
#
# 1. **Genre saja** — hanya token genre yang dipakai.
# 2. **Genre + tag** — representasi gabungan hasil Tahap 5 persiapan data.
# 3. **Genre + tag dengan profil deviasi** — profil dibangun dari seluruh rating
#    dengan bobot $r_{ui} - \bar{r}_u$, sehingga film yang dinilai di bawah
#    kebiasaan pengguna memberi kontribusi negatif dan aktif menjauhkan
#    rekomendasi dari selera yang tidak disukai.

# %%
meta['konten_genre_saja'] = meta.token_genre
tfidf_genre_saja = matriks_konten('konten_genre_saja')

varian_cbf = {
    'Genre saja': dict(matriks_isi=tfidf_genre_saja, gunakan_deviasi=False),
    'Genre + tag': dict(matriks_isi=tfidf, gunakan_deviasi=False),
    'Genre + tag, profil deviasi': dict(matriks_isi=tfidf, gunakan_deviasi=True),
}

hasil_varian = []
for nama, argumen in varian_cbf.items():
    skor_v = bangun_skor_cbf(train_fit, **argumen)
    metrik = evaluasi_ranking(skor_v, relevan_validasi, sudah_ditonton_fit, pool_fit, K=10)
    hasil_varian.append({'Varian': nama, **metrik})

tabel_varian_cbf = pd.DataFrame(hasil_varian).sort_values('NDCG@10', ascending=False).reset_index(drop=True)
print('--- Perbandingan varian CBF pada data validasi ---')
print(tabel_varian_cbf.round(4).to_string(index=False))

varian_terpilih = tabel_varian_cbf.iloc[0].Varian
print(f'\nVarian terpilih: {varian_terpilih}')

# %% [markdown]
# **Pembacaan hasil ablasi.** Varian **genre saja** justru unggul tipis untuk
# rekomendasi personal. Penjelasannya masuk akal: tag hanya tersedia pada 16%
# katalog dan berasal dari 58 pengguna. Ketika profil seseorang kebetulan
# terbentuk dari beberapa film bertag banyak, profil itu ikut condong ke kosakata
# tag yang sempit dan tidak dimiliki mayoritas kandidat, sehingga pencocokan
# menjadi kurang stabil. Varian profil deviasi memberi cakupan katalog terluas
# tetapi tidak lebih tepat.
#
# Yang lebih penting: **ketiga varian sama-sama berada jauh di bawah baseline
# populer**. Ini petunjuk awal bahwa keterbatasan CBF di sini bukan soal cara
# menyusun profil pengguna, melainkan soal resolusi metadata yang tersedia —
# dugaan yang akan dikonfirmasi pada Bab 6.
#
# Karena itu ditetapkan pembagian peran berikut:
#
# - **Mode A (item-ke-item)** tetap memakai representasi **genre + tag**, karena
#   tag menghasilkan kemiripan yang lebih kaya dan penjelasan yang lebih meyakinkan
#   ("mirip karena sama-sama bertag *pixar*").
# - **Mode B (rekomendasi personal)** memakai varian yang menang pada data
#   validasi, yaitu representasi yang tercetak di atas.

# %%
skor_cbf = bangun_skor_cbf(train_penuh, **varian_cbf[varian_terpilih])

profil_kosong = int((np.abs(skor_cbf).sum(axis=1) == 0).sum())
print('Dimensi matriks skor CBF :', skor_cbf.shape)
print(f'Rentang skor             : {skor_cbf.min():.3f} sampai {skor_cbf.max():.3f}')
print('Pengguna tanpa profil    :', profil_kosong)

# %% [markdown]
# ---
# ## 5.2 Model 2 — Collaborative Filtering dengan *Matrix Factorization* (SGD)
#
# ### Cara kerja
#
# *Matrix factorization* mengasumsikan matriks rating $R$ berukuran
# $610 \times 8.246$ dapat dihampiri oleh perkalian dua matriks berdimensi jauh
# lebih kecil: matriks faktor pengguna $P \in \mathbb{R}^{610 \times k}$ dan
# matriks faktor film $Q \in \mathbb{R}^{8.246 \times k}$. Setiap pengguna dan
# setiap film diwakili satu vektor laten berdimensi $k$ yang dipelajari langsung
# dari data — tidak ada yang memberi tahu model bahwa suatu dimensi berarti
# "kadar aksi" atau "kadar drama"; makna itu muncul sendiri dari pola rating.
#
# Prediksi rating memakai bentuk **berbias** (Koren et al., 2009):
#
# $$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^{\top} \mathbf{q}_i$$
#
# - $\mu$ — rata-rata rating global, titik acuan
# - $b_u$ — bias pengguna, menangkap kecenderungan menilai murah hati atau pelit (Insight 1)
# - $b_i$ — bias film, menangkap kualitas rata-rata film (Insight 4)
# - $\mathbf{p}_u^{\top}\mathbf{q}_i$ — kecocokan selera personal
#
# Parameter dicari dengan meminimumkan galat kuadrat berregularisasi **hanya pada
# rating yang benar-benar teramati**:
#
# $$\min_{p,q,b} \sum_{(u,i) \in \mathcal{K}} \left( r_{ui} - \hat{r}_{ui} \right)^2 + \lambda \left( \lVert \mathbf{p}_u \rVert^2 + \lVert \mathbf{q}_i \rVert^2 + b_u^2 + b_i^2 \right)$$
#
# Optimasi dilakukan dengan *stochastic gradient descent*. Untuk setiap rating
# teramati, galat $e_{ui} = r_{ui} - \hat{r}_{ui}$ dihitung lalu seluruh parameter
# terkait diperbarui berlawanan arah gradien:
#
# $$b_u \leftarrow b_u + \eta\,(e_{ui} - \lambda b_u), \qquad b_i \leftarrow b_i + \eta\,(e_{ui} - \lambda b_i)$$
# $$\mathbf{p}_u \leftarrow \mathbf{p}_u + \eta\,(e_{ui}\,\mathbf{q}_i - \lambda \mathbf{p}_u), \qquad \mathbf{q}_i \leftarrow \mathbf{q}_i + \eta\,(e_{ui}\,\mathbf{p}_u - \lambda \mathbf{q}_i)$$
#
# dengan $\eta$ laju pembelajaran dan $\lambda$ kekuatan regularisasi.

# %%
def latih_mf(data_latih, n_faktor=50, reg=0.05, lr=0.01, epochs=25,
             data_pantau=None, seed=SEED, verbose=False):
    """Melatih biased matrix factorization dengan stochastic gradient descent."""
    u_idx = data_latih.userId.map(pengguna_ke_idx).to_numpy()
    i_idx = data_latih.movieId.map(film_ke_idx).to_numpy()
    r = data_latih.rating.to_numpy(float)

    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.05, (N_PENGGUNA, n_faktor))
    Q = rng.normal(0, 0.05, (N_FILM, n_faktor))
    bu = np.zeros(N_PENGGUNA)
    bi = np.zeros(N_FILM)
    mu = r.mean()

    if data_pantau is not None:
        pantau = data_pantau[data_pantau.movieId.isin(film_ke_idx)]
        pu_idx = pantau.userId.map(pengguna_ke_idx).to_numpy()
        pi_idx = pantau.movieId.map(film_ke_idx).to_numpy()
        pr = pantau.rating.to_numpy(float)

    urutan = np.arange(len(r))
    riwayat = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(urutan)
        for t in urutan:
            u, i, nilai = u_idx[t], i_idx[t], r[t]
            p_u = P[u].copy()
            q_i = Q[i]
            galat = nilai - (mu + bu[u] + bi[i] + p_u @ q_i)
            bu[u] += lr * (galat - reg * bu[u])
            bi[i] += lr * (galat - reg * bi[i])
            P[u] += lr * (galat * q_i - reg * p_u)
            Q[i] += lr * (galat * p_u - reg * q_i)

        pred_latih = np.clip(mu + bu[u_idx] + bi[i_idx] + np.sum(P[u_idx] * Q[i_idx], axis=1), 0.5, 5.0)
        catatan = {'epoch': epoch, 'rmse_latih': evaluasi_rating(r, pred_latih)['RMSE']}
        if data_pantau is not None:
            pred_pantau = np.clip(mu + bu[pu_idx] + bi[pi_idx] + np.sum(P[pu_idx] * Q[pi_idx], axis=1), 0.5, 5.0)
            catatan['rmse_pantau'] = evaluasi_rating(pr, pred_pantau)['RMSE']
        riwayat.append(catatan)
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print('   ', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in catatan.items()})

    return {'P': P, 'Q': Q, 'bu': bu, 'bi': bi, 'mu': mu, 'riwayat': pd.DataFrame(riwayat),
            'params': {'n_faktor': n_faktor, 'reg': reg, 'lr': lr, 'epochs': epochs}}


def skor_matriks_mf(model):
    """Menghasilkan matriks skor prediksi rating untuk seluruh pasangan pengguna-film."""
    return model['mu'] + model['bu'][:, None] + model['bi'][None, :] + model['P'] @ model['Q'].T


def prediksi_mf(model, data):
    """Memprediksi rating untuk sekumpulan pasangan (userId, movieId).

    Film yang tidak ada pada katalog latih (cold-start) diprediksi dengan
    cadangan mu + b_u, yaitu tebakan terbaik ketika film belum punya riwayat.
    """
    u = data.userId.map(pengguna_ke_idx).to_numpy()
    ada = data.movieId.isin(film_ke_idx).to_numpy()
    i = data.movieId.map(film_ke_idx).fillna(0).astype(int).to_numpy()
    penuh = model['mu'] + model['bu'][u] + model['bi'][i] + np.sum(model['P'][u] * model['Q'][i], axis=1)
    cadangan = model['mu'] + model['bu'][u]
    return np.clip(np.where(ada, penuh, cadangan), 0.5, 5.0)


print('Fungsi pelatihan Matrix Factorization siap.')

# %% [markdown]
# ### Penyetelan *hyperparameter* MF pada data validasi
#
# Empat kombinasi diuji untuk melihat pengaruh jumlah faktor laten $k$ dan
# kekuatan regularisasi $\lambda$. Model dilatih pada `train_fit` dan dinilai pada
# `validasi` — **data uji sama sekali tidak dilibatkan**.

# %%
grid_mf = [
    {'n_faktor': 20, 'reg': 0.05, 'lr': 0.01},
    {'n_faktor': 50, 'reg': 0.05, 'lr': 0.01},
    {'n_faktor': 50, 'reg': 0.10, 'lr': 0.01},
    {'n_faktor': 100, 'reg': 0.05, 'lr': 0.01},
]

waktu_mulai = time.time()
hasil_grid_mf = []
riwayat_terbaik = None
for konf in grid_mf:
    model_uji = latih_mf(train_fit, epochs=25, data_pantau=validasi, **konf)
    rmse_val = model_uji['riwayat'].rmse_pantau.to_numpy()
    epoch_terbaik = int(np.argmin(rmse_val)) + 1
    hasil_grid_mf.append({**konf, 'rmse_latih': model_uji['riwayat'].rmse_latih.iloc[-1],
                          'rmse_validasi': float(rmse_val.min()), 'epoch_terbaik': epoch_terbaik})
    print(f"k={konf['n_faktor']:>3} reg={konf['reg']:.2f} -> "
          f"RMSE validasi terbaik {rmse_val.min():.4f} pada epoch {epoch_terbaik} "
          f"({time.time() - waktu_mulai:.0f} detik)")
    if riwayat_terbaik is None or rmse_val.min() < riwayat_terbaik[1]:
        riwayat_terbaik = (model_uji['riwayat'], float(rmse_val.min()), konf)

tabel_grid_mf = pd.DataFrame(hasil_grid_mf).sort_values('rmse_validasi').reset_index(drop=True)
print('\n--- Hasil penyetelan Matrix Factorization ---')
print(tabel_grid_mf.round(4).to_string(index=False))

konfigurasi_mf = tabel_grid_mf.iloc[0]
print(f"\nKonfigurasi terpilih: k={int(konfigurasi_mf.n_faktor)}, "
      f"reg={konfigurasi_mf.reg}, lr={konfigurasi_mf.lr}, "
      f"epochs={int(konfigurasi_mf.epoch_terbaik)}")

# %% [markdown]
# ### Kurva Pembelajaran Konfigurasi Terbaik
#
# Riwayat RMSE per epoch dari konfigurasi dengan skor validasi terbaik
# divisualisasikan untuk memeriksa gejala *overfitting* sekaligus menentukan
# jumlah epoch model final.

# %%
riwayat, _, konf_kurva = riwayat_terbaik
plt.figure(figsize=(7, 4.2))
plt.plot(riwayat.epoch, riwayat.rmse_latih, marker='o', ms=3, label='RMSE data latih (train_fit)')
plt.plot(riwayat.epoch, riwayat.rmse_pantau, marker='s', ms=3, label='RMSE data validasi')
epoch_opt = int(riwayat.rmse_pantau.idxmin()) + 1
plt.axvline(epoch_opt, color='#d1495b', ls='--',
            label=f'epoch optimal = {epoch_opt}')
plt.title(f"Kurva Pembelajaran Matrix Factorization "
          f"(k={konf_kurva['n_faktor']}, reg={konf_kurva['reg']})")
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend(fontsize=8)
plt.tight_layout()
simpan_gambar('fig_06_kurva_pembelajaran_mf.png')
plt.show()

print('RMSE latih pada epoch terakhir   :', round(riwayat.rmse_latih.iloc[-1], 4))
print('RMSE validasi pada epoch terakhir:', round(riwayat.rmse_pantau.iloc[-1], 4))
print('RMSE validasi terbaik            :', round(riwayat.rmse_pantau.min(), 4))

# %% [markdown]
# **Pembacaan kurva pembelajaran.** RMSE data latih terus menurun tajam sepanjang
# pelatihan hingga menyentuh sekitar 0,58, sementara RMSE validasi turun cepat pada
# epoch-epoch awal lalu mendatar di kisaran 0,87. Jarak yang terus melebar di antara
# keduanya adalah tanda *overfitting* klasik: model mulai menghafal rating individual
# alih-alih mempelajari pola selera yang dapat digeneralisasi.
#
# Jumlah epoch model final ditetapkan pada titik RMSE validasi terendah, bukan pada
# epoch terakhir. Untuk konfigurasi terpilih kedua titik itu kebetulan berimpit,
# sehingga seluruh 25 epoch tetap dipakai; sedangkan pada konfigurasi
# k=50 reg=0,10 titik terendah muncul lebih awal, yaitu pada epoch 20.

# %%
model_mf = latih_mf(train_penuh,
                    n_faktor=int(konfigurasi_mf.n_faktor),
                    reg=float(konfigurasi_mf.reg),
                    lr=float(konfigurasi_mf.lr),
                    epochs=int(konfigurasi_mf.epoch_terbaik),
                    data_pantau=None, verbose=False)
skor_mf = skor_matriks_mf(model_mf)

print('Model MF final selesai dilatih pada train_penuh.')
print('Parameter:', model_mf['params'])
print(f"Rata-rata global (mu)      : {model_mf['mu']:.4f}")
print(f"Rentang bias pengguna (b_u): {model_mf['bu'].min():.3f} sampai {model_mf['bu'].max():.3f}")
print(f"Rentang bias film (b_i)    : {model_mf['bi'].min():.3f} sampai {model_mf['bi'].max():.3f}")

# Bias film tertinggi dan terendah - pemeriksaan kewajaran hasil belajar model
urut_bias = np.argsort(-model_mf['bi'])
print('\nLima film dengan bias tertinggi (disukai lintas selera):')
print(meta.iloc[urut_bias[:5]][['judul_bersih', 'jumlah_rating_latih']]
      .assign(bias=model_mf['bi'][urut_bias[:5]].round(3)).to_string(index=False))

# %% [markdown]
# **Catatan penting atas bias film.** Lima film dengan bias tertinggi seluruhnya
# adalah film klasik yang hanya memiliki 8–39 rating. Regularisasi $\lambda = 0{,}05$
# — nilai yang dipilih karena memberikan RMSE validasi terbaik — ternyata terlalu
# lemah untuk menarik estimasi film berdata sedikit kembali ke rata-rata global.
# Akibatnya film-film tersebut memperoleh prediksi rating yang sangat tinggi dan
# akan membanjiri daftar rekomendasi.
#
# Inilah pertama kalinya terlihat bahwa **kriteria pemilihan berbasis RMSE dapat
# merugikan kualitas peringkat**: RMSE dihitung hanya pada rating yang teramati,
# sehingga tidak pernah menghukum model karena menempatkan film asing di puncak
# daftar. Konsekuensi konkretnya akan terlihat pada keluaran top-N di bagian 5.5
# dan pada angka Precision@10 di Bab 6.
