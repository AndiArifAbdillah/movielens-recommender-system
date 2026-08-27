# %% [markdown]
# ## 6.5 Kelebihan dan Kekurangan Tiap Pendekatan
#
# Tabel berikut merangkum temuan kuantitatif di atas menjadi penilaian
# menyeluruh terhadap kedua pendekatan yang diajukan.
#
# ### Content-Based Filtering
#
# **Kelebihan**
#
# 1. **Bebas dari masalah *cold-start item*.** Film baru dapat langsung
#    direkomendasikan begitu genre-nya tercatat, tanpa perlu menunggu satu rating
#    pun. Pada data uji ini, 8,4% rating menyangkut film yang tak dikenal
#    *collaborative filtering* — seluruhnya dapat ditangani CBF.
# 2. **Cakupan katalog jauh lebih luas.** Coverage@10 CBF mencapai **40,9%**,
#    tertinggi di antara seluruh model dan hampir tiga belas kali lipat baseline
#    populer yang hanya menyentuh 3,2% katalog. CBF memberi kesempatan hidup bagi
#    film di ekor distribusi.
# 3. **Dapat dijelaskan.** Alasan rekomendasi selalu tersedia dalam bahasa manusia:
#    "karena Anda menyukai *Toy Story* yang beranimasi, bergenre keluarga, dan
#    bertag *pixar*". Transparansi ini bernilai tinggi untuk membangun kepercayaan
#    pengguna.
# 4. **Tidak bergantung pada pengguna lain.** Sistem tetap berfungsi bagi pengguna
#    dengan selera tidak lazim yang tak punya "tetangga selera".
#
# **Kekurangan**
#
# 1. **Ketepatannya paling rendah** di antara seluruh pendekatan: Precision@10
#    hanya **0,010**, kurang dari seperlima baseline populer (0,056). Sebabnya
#    jelas — metadata yang tersedia terlalu kasar untuk membedakan selera.
# 2. **Terkunci pada resolusi metadata.** Ratusan film berbagi kombinasi genre
#    yang sama persis sehingga kemiripannya bernilai 1,000 dan tak dapat
#    diperingkat secara bermakna — persoalan yang terlihat gamblang pada contoh
#    *The Godfather*.
# 3. ***Over-specialization*.** Sistem hanya menawarkan hal serupa dengan yang
#    sudah ditonton dan tidak pernah memperkenalkan kejutan menyenangkan, padahal
#    justru itulah nilai utama sebuah sistem rekomendasi.
# 4. **Buta terhadap kualitas.** Kemiripan konten tidak tahu-menahu soal bagus
#    atau buruknya sebuah film; film jelek bergenre sama tetap dinilai sangat mirip.
#
# ### Collaborative Filtering
#
# **Kelebihan**
#
# 1. **Ketepatan tertinggi.** iALS mencapai Precision@10 **0,083**, Recall@10
#    **0,096**, dan NDCG@10 **0,107** — masing-masing 47%, 75%, dan 43% di atas
#    baseline populer, dan merupakan hasil terbaik di antara seluruh pendekatan.
# 2. **Menemukan pola yang tak terlihat pada metadata.** Model dapat mengaitkan
#    film lintas genre semata-mata karena orang yang sama menyukai keduanya —
#    sesuatu yang mustahil ditangkap CBF.
# 3. **Memperhitungkan kualitas dan bias secara eksplisit.** Suku $b_u$ dan $b_i$
#    menormalkan perbedaan gaya menilai antarpengguna dan perbedaan mutu
#    antarfilm, sehingga peringkat lebih adil.
# 4. **Meningkat seiring bertambahnya data.** NDCG@10 iALS naik dari 0,104 pada
#    pengguna berriwayat pendek menjadi 0,127 pada pengguna berriwayat panjang.
#    Yang menggembirakan, bahkan pada segmen terpendek pun ia sudah mengungguli
#    baseline populer.
#
# **Kekurangan**
#
# 1. ***Cold-start* pada dua sisi.** Film tanpa rating tidak memiliki vektor laten
#    yang bermakna, dan pengguna baru tidak memiliki profil sama sekali.
# 2. **Cakupan katalog lebih sempit daripada CBF** (22,0% berbanding 40,9%),
#    karena model condong pada film yang punya cukup bukti interaksi.
# 3. **Sulit dijelaskan.** Faktor laten tidak memiliki makna yang dapat
#    diverbalkan, sehingga alasan rekomendasi sukar dikomunikasikan kepada pengguna.
# 4. **Metrik optimasi harus dipilih dengan hati-hati.** Ini temuan terpenting
#    proyek ini: MF memenangi RMSE (0,880) tetapi NDCG@10-nya hanya **0,028**,
#    seperempat dari iALS (**0,107**) yang bahkan tidak dapat memprediksi rating
#    sama sekali — dan MF pun kalah dari baseline populer (0,075). Melatih model
#    untuk menebak angka ternyata bukan cara terbaik untuk menyusun peringkat.

# %% [markdown]
# ## 6.6 Ringkasan Seluruh Hasil Evaluasi

# %%
ringkasan_akhir = tabel_ranking.copy()
ringkasan_akhir['RMSE (uji)'] = [
    np.nan,
    np.nan,
    float(tabel_rating.loc[tabel_rating.Model == 'Matrix Factorization (MF)', 'RMSE'].iloc[0]),
    np.nan,
]
ringkasan_akhir['Pendekatan'] = ['Baseline non-personal', 'Content-based filtering',
                                 'Collaborative filtering (eksplisit)',
                                 'Collaborative filtering (implisit)']
ringkasan_akhir = ringkasan_akhir[['Model', 'Pendekatan', 'Precision@10', 'Recall@10',
                                   'NDCG@10', 'Coverage@10', 'RMSE (uji)']]

print('=' * 110)
print('RINGKASAN AKHIR — SELURUH MODEL PADA DATA UJI')
print('=' * 110)
print(ringkasan_akhir.round(4).to_string(index=False))
print('\nCatatan: RMSE hanya berlaku bagi model yang memprediksi nilai rating.')
print('         POP, CBF, dan iALS menghasilkan skor preferensi, bukan taksiran rating.')

# %% [markdown]
# ---
# # 7. Kesimpulan
#
# Proyek ini membangun dan membandingkan dua pendekatan sistem rekomendasi film
# di atas dataset MovieLens berisi 100.836 rating dari 610 pengguna terhadap 9.742
# film, dengan protokol evaluasi temporal yang meniru kondisi produksi.
#
# **Menjawab pernyataan masalah**
#
# 1. **Bagaimana merekomendasikan film yang relevan tanpa harus menunggu pengguna
#    menjelajah katalog sendiri?** Keduanya berhasil menghasilkan top-N
#    rekomendasi yang berfungsi, tetapi dengan mutu yang berbeda jauh. iALS
#    memberikan hasil terbaik: kurang lebih satu dari dua belas film yang
#    disodorkannya benar-benar ditonton dan disukai pengguna pada periode
#    berikutnya, tanpa pengguna perlu mencari apa pun.
#
# 2. **Bagaimana menangani film yang jarang atau belum pernah dirating?** CBF
#    menjawab persoalan ini secara langsung — ia mampu menilai film mana pun yang
#    memiliki metadata, termasuk 8,4% rating uji yang menyangkut film tak dikenal
#    *collaborative filtering*, dan mencapai cakupan katalog tertinggi. Namun
#    harga yang dibayar adalah ketepatan yang jauh lebih rendah.
#
# 3. **Pendekatan mana yang lebih tepat dan bagaimana membuktikannya?**
#    *Collaborative filtering* dengan iALS terbukti unggul secara kuantitatif pada
#    seluruh metrik ketepatan peringkat, dan pembuktiannya sahih karena
#    menggunakan baseline populer sebagai pembanding, penyetelan *hyperparameter*
#    yang terpisah dari data uji, serta protokol evaluasi yang identik bagi semua
#    model.
#
# **Temuan metodologis yang paling penting**
#
# Model dengan RMSE terbaik **bukanlah** model dengan rekomendasi terbaik. MF
# unggul dalam menebak angka rating, sementara iALS — yang bahkan tidak
# memprediksi rating — jauh mengungguli MF dalam menyusun peringkat sepuluh besar.
# Pelajarannya: metrik evaluasi harus dipilih berdasarkan **apa yang benar-benar
# dilihat pengguna**, bukan berdasarkan apa yang paling mudah dihitung.
#
# **Keterbatasan dan arah pengembangan**
#
# 1. **Hibridisasi.** CBF dan CF memiliki kelemahan yang saling melengkapi.
#    Menggabungkan skor keduanya — misalnya menyerahkan pengguna dan film baru
#    kepada CBF, lalu beralih ke iALS setelah interaksi cukup terkumpul — adalah
#    langkah lanjutan yang paling menjanjikan.
# 2. **Metadata yang lebih kaya.** Sutradara, pemeran, sinopsis, dan *tag genome*
#    akan menaikkan resolusi representasi konten jauh di atas 19 genre yang
#    tersedia sekarang.
# 3. **Skala data.** Dengan 610 pengguna, model kolaboratif bekerja pada data yang
#    sangat terbatas. Versi MovieLens 25M akan memberi gambaran performa yang lebih
#    representatif.
# 4. **Evaluasi daring.** Seluruh angka pada laporan ini berasal dari evaluasi
#    luring pada data historis. Ukuran keberhasilan yang sesungguhnya — apakah
#    pengguna benar-benar menonton yang direkomendasikan — hanya dapat diperoleh
#    melalui pengujian A/B pada sistem nyata.

# %% [markdown]
# ---
# ## Referensi
#
# 1. Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and
#    Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), 1–19.
#    <https://doi.org/10.1145/2827872>
# 2. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques
#    for Recommender Systems. *Computer*, 42(8), 30–37.
#    <https://doi.org/10.1109/MC.2009.263>
# 3. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative Filtering for Implicit
#    Feedback Datasets. *IEEE International Conference on Data Mining (ICDM)*,
#    263–272. <https://doi.org/10.1109/ICDM.2008.22>
# 4. Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based Recommender
#    Systems: State of the Art and Trends. Dalam *Recommender Systems Handbook*
#    (hlm. 73–105). Springer. <https://doi.org/10.1007/978-0-387-85820-3_3>
# 5. Cremonesi, P., Koren, Y., & Turrin, R. (2010). Performance of Recommender
#    Algorithms on Top-N Recommendation Tasks. *Proceedings of the Fourth ACM
#    Conference on Recommender Systems*, 39–46.
#    <https://doi.org/10.1145/1864708.1864721>
# 6. Järvelin, K., & Kekäläinen, J. (2002). Cumulated Gain-based Evaluation of IR
#    Techniques. *ACM Transactions on Information Systems*, 20(4), 422–446.
#    <https://doi.org/10.1145/582415.582418>
