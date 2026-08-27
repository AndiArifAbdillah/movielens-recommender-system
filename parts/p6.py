# %% [markdown]
# ---
# # 6. Evaluation
#
# ## 6.1 Metrik Evaluasi yang Digunakan
#
# Sistem rekomendasi pada proyek ini menjalankan dua tugas yang berbeda, sehingga
# dibutuhkan dua kelompok metrik.
#
# ### Kelompok A — Metrik prediksi rating (khusus model MF)
#
# **RMSE (*Root Mean Squared Error*)**
#
# $$\text{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|} \sum_{(u,i) \in \mathcal{T}} \left(r_{ui} - \hat{r}_{ui}\right)^2}$$
#
# *Cara kerja:* selisih antara rating sebenarnya dan rating prediksi dikuadratkan,
# dirata-ratakan, lalu diakarkan. Pengkuadratan membuat galat besar dihukum jauh
# lebih berat daripada galat kecil — meleset 2 bintang dihitung empat kali lebih
# buruk daripada meleset 1 bintang. Pengakaran mengembalikan satuan ke satuan
# bintang sehingga hasilnya mudah ditafsirkan. RMSE = 0,87 berarti tebakan model
# rata-rata meleset sekitar 0,87 bintang. Semakin kecil semakin baik.
#
# **MAE (*Mean Absolute Error*)**
#
# $$\text{MAE} = \frac{1}{|\mathcal{T}|} \sum_{(u,i) \in \mathcal{T}} \left| r_{ui} - \hat{r}_{ui} \right|$$
#
# *Cara kerja:* rata-rata nilai mutlak galat. Berbeda dengan RMSE, MAE
# memperlakukan semua galat secara proporsional sehingga tidak sensitif terhadap
# pencilan. Menyajikan keduanya berguna: bila RMSE jauh lebih besar daripada MAE,
# berarti model sesekali meleset sangat jauh meskipun rata-rata galatnya kecil.
#
# ### Kelompok B — Metrik kualitas peringkat (seluruh model)
#
# Sebuah film dinyatakan **relevan** bagi pengguna $u$ apabila pada data uji ia
# memberi rating ≥ 4,0. Notasi $\mathcal{R}_u$ adalah himpunan film relevan dan
# $\text{Top}_K(u)$ adalah $K$ film teratas yang direkomendasikan.
#
# **Precision@K**
#
# $$\text{Precision@}K = \frac{1}{|\mathcal{U}|}\sum_{u \in \mathcal{U}} \frac{\left| \text{Top}_K(u) \cap \mathcal{R}_u \right|}{K}$$
#
# *Cara kerja:* dari $K$ film yang disodorkan, berapa proporsi yang ternyata benar
# disukai pengguna. Metrik ini menjawab pertanyaan **"seberapa bersih layar
# rekomendasi dari salah tebak?"** — sudut pandang pengguna yang waktunya terbatas.
#
# **Recall@K**
#
# $$\text{Recall@}K = \frac{1}{|\mathcal{U}|}\sum_{u \in \mathcal{U}} \frac{\left| \text{Top}_K(u) \cap \mathcal{R}_u \right|}{\left| \mathcal{R}_u \right|}$$
#
# *Cara kerja:* dari seluruh film yang sebenarnya disukai pengguna, berapa bagian
# yang berhasil tertangkap dalam $K$ rekomendasi. Metrik ini menjawab
# **"seberapa banyak minat pengguna yang tidak terlewat?"** — sudut pandang
# katalog. Nilainya secara alami rendah ketika seorang pengguna menyukai puluhan
# film sementara hanya 10 slot yang tersedia.
#
# **NDCG@K (*Normalized Discounted Cumulative Gain*)**
#
# $$\text{DCG@}K = \sum_{j=1}^{K} \frac{rel_j}{\log_2(j+1)}, \qquad
#   \text{IDCG@}K = \sum_{j=1}^{\min(|\mathcal{R}_u|, K)} \frac{1}{\log_2(j+1)}, \qquad
#   \text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$
#
# *Cara kerja:* $rel_j$ bernilai 1 bila item pada posisi ke-$j$ relevan dan 0 bila
# tidak. Setiap hit dibagi $\log_2(j+1)$ sehingga hit di posisi 1 bernilai penuh
# (1,00), di posisi 3 bernilai 0,50, dan di posisi 10 hanya 0,29. Pembagian dengan
# IDCG — nilai DCG maksimum yang mungkin dicapai bila semua hit menumpuk di atas —
# membuat skor ternormalisasi pada rentang 0–1. **Inilah satu-satunya metrik di
# sini yang peduli pada urutan**, dan itu penting karena pengguna membaca daftar
# dari atas ke bawah.
#
# **Coverage@K**
#
# $$\text{Coverage@}K = \frac{\left| \bigcup_{u \in \mathcal{U}} \text{Top}_K(u) \right|}{\left| \mathcal{C} \right|}$$
#
# *Cara kerja:* proporsi katalog kandidat $\mathcal{C}$ yang pernah muncul pada
# rekomendasi siapa pun. Metrik ini bukan metrik ketepatan melainkan metrik
# **keberagaman**: sistem yang menyodorkan sepuluh judul yang sama kepada seluruh
# pengguna hanya mencakup sepersekian persen katalog dan membuat ribuan film lain
# tidak pernah punya kesempatan ditemukan.
#
# ### Mengapa kombinasi metrik ini sesuai dengan konteks proyek
#
# 1. **Sesuai dengan bentuk data.** Rating eksplisit 0,5–5,0 memungkinkan
#    pengukuran galat numerik (RMSE/MAE), sementara sifat data yang sangat renggang
#    (98,3% sel kosong) menuntut metrik berbasis peringkat karena mustahil menilai
#    prediksi pada film yang tidak pernah dirating.
# 2. **Sesuai dengan *problem statement*.** Yang dijanjikan sistem kepada pengguna
#    adalah **sepuluh judul teratas**, bukan taksiran angka. Precision@10,
#    Recall@10, dan NDCG@10 mengukur persis apa yang dilihat pengguna di layar.
# 3. **Sesuai dengan tujuan bisnis.** Coverage@10 menjaga agar keberhasilan tidak
#    dicapai dengan cara yang merugikan katalog, yaitu dengan mengulang-ulang film
#    populer yang sebenarnya sudah pasti ditemukan pengguna tanpa bantuan sistem.
# 4. **Akurasi saja bisa menyesatkan.** Bab ini justru akan memperlihatkan bahwa
#    model dengan RMSE terbaik bukanlah model dengan peringkat terbaik — bukti
#    nyata bahwa pemilihan metrik menentukan kesimpulan.

# %% [markdown]
# ## 6.2 Hasil Evaluasi Prediksi Rating
#
# Model MF dibandingkan dengan empat *baseline* statistik sederhana pada seluruh
# 19.940 rating data uji. Film yang belum pernah muncul di data latih diprediksi
# dengan nilai cadangan yang sesuai untuk masing-masing metode.

# %%
mu_latih = train_penuh.rating.mean()
rata_pengguna = train_penuh.groupby('userId').rating.mean()
rata_film = train_penuh.groupby('movieId').rating.mean()

y_uji = test.rating.to_numpy(float)
u_uji = test.userId.map(pengguna_ke_idx).to_numpy()
ada_di_katalog = test.movieId.isin(film_ke_idx).to_numpy()
i_uji = test.movieId.map(film_ke_idx).fillna(0).astype(int).to_numpy()

prediksi = {
    'Rata-rata global': np.full(len(test), mu_latih),
    'Rata-rata per pengguna': test.userId.map(rata_pengguna).to_numpy(float),
    'Rata-rata per film': test.movieId.map(rata_film).fillna(mu_latih).to_numpy(float),
    'Model bias saja (mu + b_u + b_i)': np.clip(
        np.where(ada_di_katalog,
                 model_mf['mu'] + model_mf['bu'][u_uji] + model_mf['bi'][i_uji],
                 model_mf['mu'] + model_mf['bu'][u_uji]), 0.5, 5.0),
    'Matrix Factorization (MF)': prediksi_mf(model_mf, test),
}

tabel_rating = pd.DataFrame([{'Model': nama, **evaluasi_rating(y_uji, p)}
                             for nama, p in prediksi.items()])
tabel_rating['Perbaikan RMSE vs rata-rata global'] = (
    (tabel_rating.RMSE.iloc[0] - tabel_rating.RMSE) / tabel_rating.RMSE.iloc[0] * 100).round(2)

print('--- Evaluasi prediksi rating pada seluruh data uji ({:,} rating) ---'.format(len(test)))
print(tabel_rating.round(4).to_string(index=False))

hanya_dikenal = ada_di_katalog
print(f'\nCatatan: {int((~ada_di_katalog).sum()):,} rating uji '
      f'({(~ada_di_katalog).mean()*100:.1f}%) menyangkut film yang tidak pernah muncul '
      'di data latih (cold-start item).')
print('RMSE MF bila cold-start item dikecualikan: '
      f"{evaluasi_rating(y_uji[hanya_dikenal], prediksi['Matrix Factorization (MF)'][hanya_dikenal])['RMSE']:.4f}")

# %% [markdown]
# **Pembacaan hasil.** Model MF mencapai RMSE **0,880** pada seluruh data uji
# (0,865 bila film *cold-start* dikecualikan), membaik **17,7%** dibanding menebak
# dengan rata-rata global. Yang lebih menarik adalah perbandingan dengan baris
# "model bias saja": suku bias $\mu + b_u + b_i$ — yang sama sekali tidak
# mengandung interaksi personal — sudah menyumbang 16,4 dari 17,7 poin persentase
# perbaikan itu. Seluruh 100 faktor laten hanya menambah sisanya, yakni sekitar
# 1,3 poin persentase.
#
# Temuan ini konsisten dengan literatur *Netflix Prize*: sebagian besar variasi
# rating dapat dijelaskan oleh "siapa yang menilai" dan "film apa yang dinilai",
# sedangkan selera personal yang sesungguhnya adalah lapisan sinyal yang lebih
# halus di atasnya. Perlu dicatat pula bahwa 8,4% rating uji menyangkut film yang
# belum pernah muncul di data latih — bukti bahwa masalah *cold-start* pada
# *collaborative filtering* bukan sekadar wacana teoretis.

# %% [markdown]
# ## 6.3 Hasil Evaluasi Kualitas Peringkat (Top-N Recommendation)
#
# Seluruh model dievaluasi dengan protokol yang sama persis: kumpulan kandidat
# yang sama, film yang sudah ditonton disingkirkan, dan definisi relevansi yang
# sama.

# %%
model_untuk_ranking = {
    'Popularitas (baseline)': skor_pop,
    'Content-Based Filtering': skor_cbf,
    'Matrix Factorization (MF)': skor_mf,
    'Implicit ALS (iALS)': skor_ials,
}

baris_hasil = []
for nama, skor in model_untuk_ranking.items():
    metrik = evaluasi_ranking(skor, relevan_uji, sudah_ditonton, pool, K=10)
    baris_hasil.append({'Model': nama, **metrik})

tabel_ranking = pd.DataFrame(baris_hasil)
print('--- Evaluasi peringkat pada data uji (K=10, {} pengguna dievaluasi) ---'
      .format(len(relevan_uji)))
print(tabel_ranking.round(4).to_string(index=False))

terbaik = tabel_ranking.loc[tabel_ranking['NDCG@10'].idxmax()]
dasar = tabel_ranking.loc[tabel_ranking.Model.str.startswith('Popularitas')].iloc[0]
print(f"\nModel terbaik menurut NDCG@10 : {terbaik.Model} ({terbaik['NDCG@10']:.4f})")
print(f"Peningkatan atas baseline populer: "
      f"{(terbaik['NDCG@10'] / dasar['NDCG@10'] - 1) * 100:+.1f}% NDCG@10, "
      f"{(terbaik['Precision@10'] / dasar['Precision@10'] - 1) * 100:+.1f}% Precision@10, "
      f"{(terbaik['Recall@10'] / dasar['Recall@10'] - 1) * 100:+.1f}% Recall@10")

# %% [markdown]
# ### Uji Kestabilan pada Beberapa Nilai K
#
# Kesimpulan yang hanya berlaku pada satu nilai K patut dicurigai. Karena itu
# seluruh metrik dihitung ulang pada K = 5, 10, dan 20.

# %%
# Evaluasi pada beberapa nilai K untuk melihat kestabilan peringkat
baris_k = []
for K in [5, 10, 20]:
    for nama, skor in model_untuk_ranking.items():
        metrik = evaluasi_ranking(skor, relevan_uji, sudah_ditonton, pool, K=K)
        baris_k.append({'K': K, 'Model': nama,
                        'Precision': metrik[f'Precision@{K}'],
                        'Recall': metrik[f'Recall@{K}'],
                        'NDCG': metrik[f'NDCG@{K}'],
                        'Coverage': metrik[f'Coverage@{K}']})
tabel_k = pd.DataFrame(baris_k)
print('--- Metrik pada K = 5, 10, dan 20 ---')
print(tabel_k.pivot(index='Model', columns='K', values=['Precision', 'NDCG'])
      .round(4).to_string())

# %% [markdown]
# ### Visualisasi Perbandingan Antarmodel
#
# Tiga panel: perbandingan metrik ketepatan pada K=10, pergerakan NDCG@K terhadap
# nilai K, dan cakupan katalog masing-masing model.

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
warna_model = {'Popularitas (baseline)': '#9e9e9e', 'Content-Based Filtering': '#e07a5f',
               'Matrix Factorization (MF)': '#3d5a80', 'Implicit ALS (iALS)': '#2a9d8f'}

metrik_bar = ['Precision@10', 'Recall@10', 'NDCG@10']
x = np.arange(len(metrik_bar))
lebar = 0.2
for j, (nama, _) in enumerate(model_untuk_ranking.items()):
    nilai = tabel_ranking.loc[tabel_ranking.Model == nama, metrik_bar].to_numpy().ravel()
    posisi = x + (j - 1.5) * lebar
    ax[0].bar(posisi, nilai, lebar, label=nama, color=warna_model[nama])
    for xp, v in zip(posisi, nilai):
        ax[0].text(xp, v + 0.002, f'{v:.3f}', ha='center', fontsize=7, rotation=90)
ax[0].set_xticks(x)
ax[0].set_xticklabels(metrik_bar)
ax[0].set_title('Perbandingan Metrik Peringkat (K=10)')
ax[0].set_ylabel('Nilai metrik')
ax[0].legend(fontsize=7)
ax[0].set_ylim(0, max(tabel_ranking[metrik_bar].to_numpy().max() * 1.35, 0.12))

for nama in model_untuk_ranking:
    sub = tabel_k[tabel_k.Model == nama].sort_values('K')
    ax[1].plot(sub.K, sub.NDCG, marker='o', label=nama, color=warna_model[nama])
ax[1].set_title('NDCG@K terhadap Nilai K')
ax[1].set_xlabel('K (jumlah rekomendasi)')
ax[1].set_ylabel('NDCG@K')
ax[1].set_xticks([5, 10, 20])
ax[1].legend(fontsize=7)

nilai_cov = tabel_ranking['Coverage@10'].to_numpy() * 100
ax[2].barh([n for n in model_untuk_ranking], nilai_cov,
           color=[warna_model[n] for n in model_untuk_ranking])
for i, v in enumerate(nilai_cov):
    ax[2].text(v + 0.6, i, f'{v:.1f}%', va='center', fontsize=8)
ax[2].set_title('Coverage@10 — Cakupan Katalog')
ax[2].set_xlabel('Persen katalog kandidat yang direkomendasikan')
ax[2].set_xlim(0, max(nilai_cov) * 1.25)

plt.tight_layout()
simpan_gambar('fig_07_perbandingan_model.png')
plt.show()

# %% [markdown]
# **Pembacaan hasil peringkat.** Tabel dan grafik di atas memuat empat temuan
# utama.
#
# 1. **iALS adalah satu-satunya model yang benar-benar mengalahkan baseline
#    populer**, dan selisihnya besar: Precision@10 naik 47%, Recall@10 naik 75%,
#    dan NDCG@10 naik 43%. Sekitar satu dari dua belas judul yang
#    direkomendasikannya benar-benar ditonton dan disukai pengguna pada periode
#    berikutnya. Sekaligus, cakupan katalognya tujuh kali lebih luas daripada
#    baseline populer — jadi keunggulan itu tidak diperoleh dengan cara
#    mengulang-ulang film terkenal.
# 2. **MF justru kalah dari baseline populer** pada seluruh metrik peringkat,
#    padahal ia adalah model dengan RMSE terbaik. Sebabnya sudah terlihat pada
#    bagian 5.2: regularisasi ringan yang optimal bagi RMSE membiarkan film
#    berdata sedikit memperoleh bias tinggi, dan film-film itulah yang membanjiri
#    sepuluh besar. RMSE tidak pernah menghukum perilaku ini karena ia hanya
#    dihitung pada rating yang teramati.
# 3. **CBF berada paling bawah pada ketepatan tetapi paling atas pada cakupan.**
#    Ia menyentuh 40,9% katalog kandidat, sementara baseline populer hanya 3,2%.
#    Pertukaran ini nyata dan perlu disadari: CBF memberi kesempatan hidup bagi
#    film di ekor katalog, dengan harga ketepatan yang jauh lebih rendah.
# 4. **Urutan peringkat model stabil di semua nilai K.** Pada K = 5, 10, maupun
#    20, susunannya tetap iALS > POP > MF > CBF, sehingga kesimpulan ini bukan
#    kebetulan akibat pemilihan satu nilai K tertentu.

# %% [markdown]
# ## 6.4 Analisis Tambahan — Performa Menurut Panjang Riwayat Pengguna
#
# Angka rata-rata dapat menyembunyikan perbedaan perilaku antarsegmen pengguna.
# Bagian ini memecah hasil menurut banyaknya film yang pernah ditonton seorang
# pengguna pada data latih, untuk menguji dugaan umum bahwa *collaborative
# filtering* melemah ketika riwayat pengguna pendek.

# %%
jumlah_tonton = sudah_ditonton.sum(axis=1)
batas_segmen = np.quantile([jumlah_tonton[u] for u in relevan_uji], [0.33, 0.66])


def segmen_pengguna(n):
    if n <= batas_segmen[0]:
        return f'Riwayat pendek (<= {int(batas_segmen[0])} film)'
    if n <= batas_segmen[1]:
        return f'Riwayat sedang ({int(batas_segmen[0]) + 1}-{int(batas_segmen[1])} film)'
    return f'Riwayat panjang (> {int(batas_segmen[1])} film)'


baris_segmen = []
for nama_segmen in sorted({segmen_pengguna(jumlah_tonton[u]) for u in relevan_uji},
                          key=lambda s: batas_segmen[0] if 'pendek' in s else (
                              batas_segmen[1] if 'sedang' in s else 1e9)):
    subset = {u: r for u, r in relevan_uji.items()
              if segmen_pengguna(jumlah_tonton[u]) == nama_segmen}
    catatan = {'Segmen': nama_segmen, 'Jumlah pengguna': len(subset)}
    for nama, skor in model_untuk_ranking.items():
        catatan[nama] = evaluasi_ranking(skor, subset, sudah_ditonton, pool, K=10)['NDCG@10']
    baris_segmen.append(catatan)

tabel_segmen = pd.DataFrame(baris_segmen)
print('--- NDCG@10 menurut panjang riwayat pengguna ---')
print(tabel_segmen.round(4).to_string(index=False))

# %% [markdown]
# ### Visualisasi Performa Antarsegmen
#
# Perbandingan NDCG@10 keempat model pada tiga segmen panjang riwayat pengguna.

# %%
plt.figure(figsize=(8, 4.2))
x = np.arange(len(tabel_segmen))
lebar = 0.2
for j, nama in enumerate(model_untuk_ranking):
    plt.bar(x + (j - 1.5) * lebar, tabel_segmen[nama], lebar, label=nama, color=warna_model[nama])
plt.xticks(x, [s.replace(' (', '\n(') for s in tabel_segmen.Segmen], fontsize=8)
plt.ylabel('NDCG@10')
plt.title('Performa Model Menurut Panjang Riwayat Pengguna')
plt.legend(fontsize=7)
plt.tight_layout()
simpan_gambar('fig_08_segmen_pengguna.png')
plt.show()

# %% [markdown]
# **Pembacaan analisis segmen.** Dugaan awal ternyata hanya sebagian benar.
# iALS memang bekerja paling baik pada pengguna berriwayat panjang (NDCG@10
# 0,127), tetapi pada pengguna berriwayat pendek pun ia tetap unggul (0,104) dan
# masih jauh di atas baseline populer (0,061). Dengan kata lain, 20–37 film sudah
# cukup bagi iALS untuk menangkap selera seseorang.
#
# Sebaliknya, MF melemah drastis pada pengguna berriwayat pendek (0,008) — jauh di
# bawah baseline populer. Pengguna dengan sedikit rating memiliki vektor laten yang
# nyaris tak terlatih, sehingga peringkatnya hampir seluruhnya ditentukan bias film
# yang, seperti sudah ditunjukkan, condong pada film klasik berdata sedikit.
#
# CBF konsisten rendah di seluruh segmen. Ini menegaskan sekali lagi bahwa
# kendalanya bukan pada banyak-sedikitnya data pengguna, melainkan pada resolusi
# metadata yang tersedia.
