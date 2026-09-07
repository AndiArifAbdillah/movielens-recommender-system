# Bedah Kode — Penjelasan Baris demi Baris

Dokumen ini membedah kode proyek satu per satu: apa yang dikerjakan tiap baris, kenapa ditulis begitu, dan apa jadinya kalau ditulis dengan cara lain.

**Bedanya dengan dokumen lain di repo ini:**

| Dokumen | Isi |
|---|---|
| [`README.md`](README.md) | Ringkasan proyek dan hasil |
| [`Laporan_Proyek_Sistem_Rekomendasi.md`](Laporan_Proyek_Sistem_Rekomendasi.md) | Laporan formal: metodologi, hasil, pembahasan |
| [`BELAJAR.md`](BELAJAR.md) | Kurikulum berjenjang dengan latihan dan ujian mandiri |
| **`BEDAH_KODE.md`** (dokumen ini) | **Rujukan baris demi baris — dibuka saat kamu berhadapan dengan kodenya** |

Gunakan dokumen ini sambil membuka kodenya berdampingan. Kalau kamu belum paham konsep dasarnya (TF-IDF, cosine similarity), baca [`BELAJAR.md`](BELAJAR.md) lebih dulu — dokumen ini mengasumsikan konsepnya sudah kamu kenal dan langsung masuk ke implementasi.

## Daftar Isi

- [Bagian 1 — Isi Keempat Berkas Dataset](#bagian-1--isi-keempat-berkas-dataset)
- [Bagian 2 — Data Preparation, Delapan Tahap](#bagian-2--data-preparation-delapan-tahap)
- [Bagian 3 — Modeling: Content-Based Filtering](#bagian-3--modeling-content-based-filtering)
- [Bagian 4 — Modeling: Matrix Factorization (SGD)](#bagian-4--modeling-matrix-factorization-sgd)
- [Bagian 5 — Modeling: Implicit ALS](#bagian-5--modeling-implicit-als)
- [Bagian 6 — Baseline dan Penyajian Top-N](#bagian-6--baseline-dan-penyajian-top-n)

---

# Bagian 1 — Isi Keempat Berkas Dataset

## `ratings.csv` — siapa menilai apa, berapa, kapan

100.836 baris, 4 kolom.

```
userId,movieId,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247
1,6,4.0,964982224
```

| Kolom | Isi | Contoh |
|---|---|---|
| `userId` | Nomor anonim pengguna (1–610) | `1` |
| `movieId` | Nomor film — **kunci penghubung** ke berkas lain | `1` |
| `rating` | Penilaian, skala 0,5–5,0 kelipatan 0,5 | `4.0` |
| `timestamp` | Detik sejak 1 Januari 1970 | `964982703` = 30 Juli 2000, 18:45 |

**Satu baris = satu orang menilai satu film.** `userId 1` muncul 232 kali karena menilai 232 film. Tidak ada pasangan `(userId, movieId)` kembar.

Sebaran nilainya:

```
0.5  █                                       1.370  ( 1,4%)
1.0  ████                                    2.811  ( 2,8%)
1.5  ██                                      1.791  ( 1,8%)
2.0  ██████████                              7.551  ( 7,5%)
2.5  ███████                                 5.550  ( 5,5%)
3.0  ████████████████████████████           20.047  (19,9%)
3.5  ██████████████████                     13.136  (13,0%)
4.0  ██████████████████████████████████████ 26.818  (26,6%)   ← terbanyak
4.5  ████████████                            8.551  ( 8,5%)
5.0  ██████████████████                     13.211  (13,1%)
```

Miring ke kanan karena orang cenderung hanya menonton film yang memang diminati. **Akibatnya `3.0` bukan berarti "biasa saja" — itu relatif rendah.** Inilah dasar ambang "disukai" ditetapkan di `4.0`.

Kolom `timestamp` gampang diremehkan, padahal justru dia yang memungkinkan pembagian data secara temporal.

## `movies.csv` — kamus judul dan genre

9.742 baris, 3 kolom.

```
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
```

| Kolom | Isi | Catatan |
|---|---|---|
| `movieId` | Nomor film | Sama dengan di `ratings.csv` |
| `title` | Judul **beserta tahun** dalam kurung | Dua informasi menempel — dipisahkan di Tahap 2 |
| `genres` | Daftar genre dipisah tanda pipa | Rata-rata 2,27 genre per film |

Genre yang tersedia — **hanya 19 kategori** untuk 9.742 film:

```
Drama 4.361  ·  Comedy 3.756  ·  Thriller 1.894  ·  Action 1.828
Romance 1.596  ·  Adventure 1.263  ·  Crime 1.199  ·  Sci-Fi 980
Horror 978  ·  Fantasy 779  ·  Children 664  ·  Animation 611
Mystery 573  ·  Documentary 440  ·  War 382  ·  Musical 334
Western 167  ·  IMAX 158  ·  Film-Noir 87
```

Plus 34 film berlabel `(no genres listed)` — itu bukan genre, melainkan penanda data kosong.

**Ingat angka ini: 19 genre untuk 9.742 film.** Artinya ribuan film pasti berbagi kombinasi genre yang persis sama. Konsekuensinya muncul di Bagian 3.

## `tags.csv` — label bebas buatan pengguna

3.683 baris, 4 kolom.

```
userId,movieId,tag,timestamp
2,60756,funny,1445714994
2,60756,Highly quotable,1445714996
2,89774,Boxing story,1445715207
```

Bedanya dengan genre sangat mendasar. Genre adalah **kategori tertutup** (19, ditetapkan pengelola); tag adalah **teks bebas** yang diketik pengguna sendiri. Isinya jauh lebih spesifik:

```
in netflix queue 131x  ·  atmospheric 41x  ·  superhero 24x
thought-provoking 24x  ·  surreal 24x  ·  dark comedy 21x
twist ending 20x  ·  visually appealing 20x
```

**Tapi datanya sangat timpang.** Cuma 58 dari 610 pengguna pernah memberi tag, dan cuma 16,1% film punya tag:

```
userId 474   1.507 tag  (40,9% dari seluruh tag)
userId 567     432 tag  (11,7%)
userId 62      370 tag  (10,0%)
userId 599     323 tag  ( 8,8%)
userId 477     280 tag  ( 7,6%)
                         ───────
       lima orang menyumbang 79,1% dari SELURUH tag
```

Bandingkan: **Toy Story** punya 3 tag (`pixar`, `pixar`, `fun`), sementara **Fight Club** punya 54 tag. Datanya juga berantakan karena bebas diketik: `Pixar` vs `pixar`, `Sci-Fi` / `Sci-fi` / `sci-fi`.

Ketimpangan inilah yang menjelaskan dua keputusan di Tahap 4 dan 5.

## `links.csv` — jembatan ke basis data film luar

9.742 baris, 3 kolom. Satu baris untuk setiap film.

```
movieId,imdbId,tmdbId
1,0114709,862
2,0113497,8844
```

Gunanya cuma menerjemahkan id MovieLens ke id situs lain:

```
movieId 1  →  https://www.imdb.com/title/tt0114709/
           →  https://www.themoviedb.org/movie/862
```

`imdbId` disimpan tanpa awalan `tt` dan tanpa nol di depan; untuk jadi tautan harus dikembalikan ke 7 digit lalu diberi awalan.

**Berkas ini tidak dipakai proyek.** Isinya cuma nomor — tak ada informasi tentang filmnya. Baru berguna kalau menarik data dari IMDb/TMDB, dan itu butuh API key serta koneksi internet setiap kali dijalankan, sementara proyek ini sengaja dibuat bisa jalan hanya dengan `python`.

Tapi di sinilah potensinya: lewat `links.csv` kamu bisa menarik sutradara, pemeran, dan sinopsis dari TMDB — melompat dari 19 genre ke ribuan fitur. Itu jalan keluar paling menjanjikan dari keterbatasan CBF yang dibahas di Bagian 3.

## Ringkasan peran keempat berkas

| Berkas | Isi | Dipakai untuk |
|---|---|---|
| `ratings.csv` | Perilaku: siapa menilai apa | **Collaborative filtering** (MF, iALS) + evaluasi |
| `movies.csv` | Identitas: judul, tahun, genre | **Content-based filtering** |
| `tags.csv` | Label bebas dari pengguna | Memperkaya CBF (mode "film mirip") |
| `links.csv` | Nomor rujukan ke IMDb/TMDB | — tidak dipakai |

---

# Bagian 2 — Data Preparation, Delapan Tahap

Sumber: [`parts/p3.py`](parts/p3.py) · Notebook: Bab 4

## Tahap 1 — Verifikasi duplikat dan nilai kosong

```python
jml_awal = len(ratings)
ratings = ratings.drop_duplicates(subset=['userId', 'movieId'], keep='last').reset_index(drop=True)
ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])
movies = movies.drop_duplicates(subset=['movieId']).reset_index(drop=True)
```

- `drop_duplicates(subset=['userId','movieId'])` — buang baris yang pasangan pengguna–filmnya kembar. `subset` penting: yang dibandingkan cuma dua kolom itu, bukan seluruh baris. `keep='last'` mempertahankan yang terakhir — masuk akal karena itu penilaian terbaru.
- `reset_index(drop=True)` — setelah baris dibuang, indeksnya bolong (0, 1, 3, 4…). Ini merapikannya. `drop=True` supaya indeks lama tidak jadi kolom baru.
- `dropna(subset=[...])` — buang baris yang kolom kuncinya kosong.

**Kenapa perlu.** Semua model mengasumsikan **satu sel matriks = paling banyak satu rating**. Kalau ada kembar, satu interaksi terhitung dua kali dan bobot pembelajarannya timpang.

Hasilnya: **0 baris dibuang** — data MovieLens memang sudah bersih. Jadi tahap ini murni **defensif**: kalau datasetnya diperbarui dan ternyata kotor, masalahnya tertangani otomatis alih-alih diam-diam merusak model.

## Tahap 2 — Pisahkan tahun rilis dari judul

```python
movies['tahun_rilis'] = pd.to_numeric(
    movies.title.str.extract(r'\((\d{4})\)\s*$')[0], errors='coerce')
movies['judul_bersih'] = movies.title.str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True).str.strip()
```

Bedah regex `\((\d{4})\)\s*$`:

| Potongan | Artinya |
|---|---|
| `\(` | kurung buka harfiah (di-*escape* karena `(` punya arti khusus) |
| `(\d{4})` | **tangkap** persis 4 digit — kurung di sini berarti "simpan bagian ini" |
| `\)` | kurung tutup harfiah |
| `\s*` | boleh ada spasi setelahnya, boleh tidak |
| `$` | **harus di ujung teks** |

`$` itu krusial. Tanpa dia, judul seperti `Apollo 13 (1995)` bisa salah tangkap.

- `.str.extract(...)` mengembalikan DataFrame (satu kolom per grup tangkapan), jadi `[0]` mengambil kolom pertama.
- `errors='coerce'` — judul tanpa tahun menghasilkan `NaN`, **bukan error**. Inilah yang menangani 13 film yang judulnya memang tak memuat tahun.
- Baris kedua memakai pola sama untuk **menghapus**, lalu `.str.strip()` merapikan spasi sisa.

**Kenapa perlu.** Selama tahun menyatu dengan judul, ia tak bisa dihitung sebagai angka, dan mencari film harus mengetik tahunnya dengan tepat. Setelah dipisah, `rekomendasi_serupa('Toy Story')` cukup pakai judul.

## Tahap 3 — Normalisasi genre jadi token

```python
def normalkan_token(teks):
    return re.sub(r'[^a-z0-9]+', '', str(teks).lower())

movies['daftar_genre'] = (movies.genres
                          .str.replace('(no genres listed)', '', regex=False)
                          .str.split('|')
                          .apply(lambda g: [x for x in g if x]))
movies['token_genre'] = movies.daftar_genre.apply(
    lambda g: ' '.join(normalkan_token(x) for x in g))
```

**Fungsi `normalkan_token`:** `[^a-z0-9]+` berarti "satu atau lebih karakter yang **bukan** huruf kecil atau angka" — tanda `^` di dalam kurung siku artinya *negasi*. Urutannya penting: `.lower()` jalan **dulu**, baru penyaringan, supaya huruf besar tidak ikut terhapus.

```
'Sci-Fi'          → 'scifi'
'Film-Noir'       → 'filmnoir'
'based on a book' → 'basedonabook'
```

**Rantai tiga langkah pada `daftar_genre`:**

1. `.str.replace('(no genres listed)', '', regex=False)` — hapus penanda kosong. `regex=False` **wajib**, sebab tanda kurung dalam teks itu akan diartikan sebagai grup regex kalau tidak dimatikan.
2. `.str.split('|')` — pecah jadi list: `'Comedy|Romance'` → `['Comedy', 'Romance']`
3. `.apply(lambda g: [x for x in g if x])` — buang string kosong. Ini membersihkan sisa langkah 1: film tanpa genre jadi `''`, lalu `split` menghasilkan `['']`, dan baris ini mengubahnya jadi `[]`.

**Kenapa perlu.** TF-IDF bekerja pada token. Tanpa normalisasi, `Sci-Fi` berisiko pecah jadi `sci` dan `fi`. Dan `(no genres listed)` bukan genre — kalau dibiarkan, 34 film dianggap saling mirip padahal yang mereka bagi cuma *ketiadaan informasi*.

## Tahap 4 — Kumpulkan tag jadi satu dokumen per film

```python
tags_bersih = tags.copy()
tags_bersih['token_tag'] = tags_bersih.tag.apply(normalkan_token)
tags_bersih = tags_bersih[tags_bersih.token_tag.str.len() > 1]

tag_per_film = (tags_bersih.groupby('movieId')
                           .token_tag.apply(lambda s: ' '.join(s))
                           .rename('token_tag'))

movies = movies.merge(tag_per_film, on='movieId', how='left')
movies['token_tag'] = movies.token_tag.fillna('')
```

- `.copy()` — bekerja di salinan supaya `tags` asli tidak berubah.
- Fungsi `normalkan_token` yang sama dipakai ulang, jadi `Pixar` dan `pixar` menyatu.
- `.str.len() > 1` — buang tag satu karakter. Di dataset ini tak ada yang terbuang, tapi jaring pengamannya tetap dipasang.
- `groupby('movieId').token_tag.apply(' '.join)` — **inti tahapnya.** Satu film punya banyak baris tag; ini menggabungkannya jadi satu string:

```
Fight Club:  'darkcomedy' + 'psychology' + 'twistending' ...
          →  'darkcomedy psychology twistending ...'
```

- `merge(..., how='left')` — `left` berarti **semua film dipertahankan** meski tak punya tag. Kalau pakai `inner`, 84% katalog lenyap.
- `.fillna('')` — film tanpa tag dapat `NaN` dari merge; ini menggantinya dengan string kosong supaya penggabungan teks di Tahap 5 tidak error.

**Kenapa perlu.** TF-IDF butuh **tepat satu dokumen per film**. Frekuensi sengaja dipertahankan: kalau dua orang memberi tag `pixar`, token itu muncul dua kali dan TF-IDF memberinya bobot lebih besar.

## Tahap 5 — Content soup dan TF-IDF

```python
movies['content_soup'] = ((movies.token_genre + ' ') * 2 + movies.token_tag).str.strip()

vectorizer = TfidfVectorizer(token_pattern=r'\S+', min_df=1)
tfidf_semua = vectorizer.fit_transform(movies.content_soup)
```

**`(movies.token_genre + ' ') * 2`** — pada Series bertipe teks, `*` **mengulang string**, bukan mengalikan angka. Hasilnya:

```
Toy Story → 'adventure animation children comedy fantasy
             adventure animation children comedy fantasy pixar pixar fun'
```

**`token_pattern=r'\S+'`** — krusial dan gampang terlewat. Pola bawaan `TfidfVectorizer` adalah `(?u)\b\w\w+\b`, yang **membuang token satu huruf dan memotong di tanda baca**. Karena token kita sudah dinormalkan sendiri, kita mau scikit-learn **tidak ikut campur** — cukup pisahkan di spasi. `\S+` berarti "satu atau lebih karakter bukan-spasi".

**`min_df=1`** — token dipakai walau cuma muncul di satu film. Untuk data sekecil ini, membuang token langka justru merugikan.

**`fit_transform`** melakukan dua hal: *fit* (pelajari kosakata + hitung IDF tiap token) lalu *transform* (ubah tiap film jadi vektor). Hasilnya matriks **9.742 × 1.461**.

Bukti IDF bekerja — bobot tertinggi untuk *Toy Story*:

| Token | Bobot |
|---|---|
| pixar | 0,714 |
| fun | 0,349 |
| animation | 0,314 |
| children | 0,307 |

`pixar` menang telak atas `animation` padahal keduanya muncul dua kali. Yang membedakan murni faktor IDF: `animation` ada di 611 film, `pixar` cuma di segelintir.

**Kenapa genre digandakan.** Genre ada di hampir seluruh katalog, tag cuma di 16%. Tanpa penggandaan, *Fight Club* dengan 54 tag akan didominasi tag sampai sinyal genrenya tenggelam — dan kemiripannya dengan film tanpa tag jadi tak sebanding.

## Tahap 6 — Ubah id jadi indeks berurutan

```python
ratings = ratings.sort_values(['userId', 'timestamp']).reset_index(drop=True)
id_pengguna = np.sort(ratings.userId.unique())
pengguna_ke_idx = {uid: i for i, uid in enumerate(id_pengguna)}
```

`enumerate` memberi pasangan (posisi, nilai), jadi kamusnya: `{1:0, 2:1, 3:2, ...}`.

**Kenapa perlu.** `userId` rapi 1–610, tapi `movieId` melompat sampai **193.609**. Kalau `movieId` dipakai langsung sebagai nomor kolom, terbentuk matriks 610 × 193.610 yang 95% kolomnya kosong permanen — boros memori dan komputasi tanpa manfaat.

`sort_values(['userId','timestamp'])` menyiapkan Tahap 7: urutan waktu per pengguna harus benar sebelum dibagi.

## Tahap 7 — Bagi data secara temporal

```python
def bagi_temporal(df, porsi_latih=0.8):
    df = df.sort_values(['userId', 'timestamp'])
    urutan = df.groupby('userId').cumcount()
    banyak = df.groupby('userId').movieId.transform('size')
    batas = np.ceil(banyak * porsi_latih)
    return df[urutan < batas].copy(), df[urutan >= batas].copy()

train_penuh, test = bagi_temporal(ratings, 0.8)
train_fit, validasi = bagi_temporal(train_penuh, 0.8)
```

Fungsi paling padat di seluruh tahap persiapan. Dua baris tengahnya bekerja berpasangan:

- **`cumcount()`** — nomor urut **di dalam** tiap kelompok, mulai 0. Karena sudah diurutkan waktu, angka ini berarti "*ini rating ke-berapa yang pernah diberikan orang itu*".
- **`transform('size')`** — total anggota kelompok, **disiarkan ke setiap baris**. Bedanya dengan `agg('size')`: `agg` meringkas jadi 610 baris, `transform` mempertahankan 100.836 baris tetapi tiap baris kini tahu total rating pemiliknya.

Ilustrasi untuk pengguna dengan 10 rating:

```
urutan :  0  1  2  3  4  5  6  7  8  9
banyak : 10 10 10 10 10 10 10 10 10 10
batas  :  8  8  8  8  8  8  8  8  8  8     ← ceil(10 × 0,8)
           └──────── latih ────────┘ └uji┘
```

`np.ceil` (bulat ke atas) dipakai supaya pengguna dengan sedikit rating tetap kebagian data latih memadai.

Fungsi ini **dipanggil dua kali**. Panggilan kedua membelah `train_penuh` jadi `train_fit` dan `validasi` — inilah yang memungkinkan penyetelan *hyperparameter* **tanpa menyentuh data uji sama sekali**.

Verifikasinya membuktikan hasilnya: **0 pengguna melanggar urutan waktu**.

| Subset | Rating | Porsi |
|---|---|---|
| `train_penuh` | 80.896 | 80,2% |
| `test` | 19.940 | 19,8% |
| ├─ `train_fit` | 64.960 | 64,4% |
| └─ `validasi` | 15.936 | 15,8% |

## Tahap 8 — Matriks interaksi, candidate pool, item relevan

Bagian terpanjang, dipecah jadi empat.

### 8a — Selaraskan metadata dengan katalog

```python
katalog = np.sort(train_penuh.movieId.unique())
film_ke_idx = {mid: i for i, mid in enumerate(katalog)}

meta = movies.set_index('movieId').loc[katalog].reset_index()
tfidf = normalize(vectorizer.transform(meta.content_soup))
```

`.loc[katalog]` bukan sekadar menyaring — ia juga **mengurutkan ulang** `meta` supaya persis mengikuti urutan `katalog`. Ini wajib: baris ke-*i* di `meta` harus film yang sama dengan kolom ke-*i* di matriks interaksi. Kalau tidak selaras, seluruh rekomendasi menunjuk film keliru — bug senyap yang sangat sulit dilacak.

Perhatikan di sini `vectorizer.transform`, **bukan** `fit_transform`. Kosakata dan bobot IDF sudah dipelajari di Tahap 5 dari seluruh 9.742 film; di sini kita hanya menerapkannya ke 8.246 film katalog.

`normalize()` menjadikan panjang tiap vektor persis 1, sehingga **cosine similarity cukup dihitung sebagai perkalian titik**.

### 8b — Bangun matriks interaksi

```python
baris = train_penuh.userId.map(pengguna_ke_idx).to_numpy()
kolom = train_penuh.movieId.map(film_ke_idx).to_numpy()
matriks_latih = sp.csr_matrix((train_penuh.rating.to_numpy(float), (baris, kolom)),
                              shape=(N_PENGGUNA, N_FILM))

sudah_ditonton = np.zeros((N_PENGGUNA, N_FILM), dtype=bool)
sudah_ditonton[baris, kolom] = True
```

Bentuk `csr_matrix((data, (baris, kolom)), shape=...)` dibaca: **"taruh nilai `data[k]` di posisi `(baris[k], kolom[k])`, sisanya nol"**. Tiga array itu sejajar posisinya.

`sudah_ditonton` adalah matriks boolean yang nanti dipakai untuk **mencoret film yang sudah ditonton** dari daftar rekomendasi. Baris `sudah_ditonton[baris, kolom] = True` memakai *fancy indexing* NumPy — menandai puluhan ribu posisi sekaligus tanpa satu pun perulangan Python.

### 8c — Candidate pool

```python
MIN_RATING_KANDIDAT = 5
jumlah_rating_film = np.asarray((matriks_latih > 0).sum(axis=0)).ravel()
pool = np.where(jumlah_rating_film >= MIN_RATING_KANDIDAT)[0]
```

- `(matriks_latih > 0)` mengubah nilai rating jadi True/False. Yang dihitung adalah **berapa orang menilai**, bukan **jumlah nilainya**. Tanpa `> 0`, satu rating 5,0 terhitung sama dengan sepuluh rating 0,5.
- `.sum(axis=0)` menjumlahkan **ke bawah** (per kolom = per film).
- `np.asarray(...).ravel()` — SciPy mengembalikan matriks 1×N; ini memipihkannya jadi array biasa.
- `np.where(kondisi)[0]` mengembalikan **posisi** yang memenuhi syarat. `[0]` diperlukan karena `np.where` mengembalikan tuple.

Hasilnya 3.039 film dari 8.246 — tapi mencakup **88,7% seluruh rating**. Yang tersisih hanya ekor yang memang tak punya bukti kolaboratif.

### 8d — Himpunan item relevan

```python
AMBANG_SUKA = 4.0
test_relevan = test[test.movieId.isin(film_ke_idx) & (test.rating >= AMBANG_SUKA)].copy()
test_relevan['idx_film'] = test_relevan.movieId.map(film_ke_idx)
test_relevan = test_relevan[test_relevan.idx_film.isin(pool_set)]
relevan_uji = {pengguna_ke_idx[u]: set(g.idx_film) for u, g in test_relevan.groupby('userId')}
```

Tiga saringan berlapis:

1. `isin(film_ke_idx)` — film harus dikenal model (buang *cold-start item*)
2. `rating >= 4.0` — hanya yang benar-benar disukai dihitung "benar"
3. `isin(pool_set)` — hanya film yang boleh direkomendasikan

Baris terakhir menghasilkan kamus `{indeks_pengguna: {kumpulan indeks film}}`. Dipakai `set` bukan list karena pengecekan `i in item_relevan` di fungsi evaluasi berjalan seketika pada set, sedangkan pada list harus menelusuri satu per satu.

Hasil akhir: **586 dari 610 pengguna** bisa dievaluasi, rata-rata 12,6 film relevan per orang.

## Kenapa urutannya harus begitu

Kedelapan tahap tidak bisa ditukar. Tiap tahap memakan keluaran tahap sebelumnya:

```
1 bersihkan  →  2 pisah judul/tahun  →  3 token genre  ┐
                                        4 token tag   ├→ 5 TF-IDF
6 encode id  →  7 bagi temporal  →  8 matriks + pool + relevan
```

Tahap 5 tak bisa jalan sebelum 3 dan 4 selesai. Tahap 8 butuh katalog dari Tahap 7 dan kamus id dari Tahap 6. Inilah alasan laporan menekankan urutan — bukan soal kerapian, tapi memang begitu ketergantungannya.

---

# Bagian 3 — Modeling: Content-Based Filtering

Sumber: [`parts/p4.py`](parts/p4.py) · Notebook: Bab 5.1

CBF bekerja dalam **dua mode** yang keperluannya berbeda:

| Mode | Pertanyaan yang dijawab | Fungsi |
|---|---|---|
| **A** — item ke item | "Film apa yang mirip dengan *Toy Story*?" | `rekomendasi_serupa()` |
| **B** — personal | "Film apa yang cocok untuk pengguna 452?" | `bangun_skor_cbf()` |

Mode A adalah mekanisme di balik kolom "karena Anda menonton…". Mode B yang dinilai pada evaluasi.

---

## Mode A — Rekomendasi item ke item

### Kamus judul

```python
judul_ke_idx = {judul: i for i, judul in enumerate(meta.judul_bersih)}
```

Memetakan judul bersih ke **nomor baris di `meta`**. Ingat dari Tahap 8a: baris ke-*i* di `meta` bersesuaian dengan baris ke-*i* di matriks `tfidf`. Jadi kamus ini sekaligus jalan pintas dari judul ke vektor kontennya.

`enumerate` dipakai lagi di sini dengan pola yang sama seperti `pengguna_ke_idx` dan `film_ke_idx` — satu pola yang berulang di seluruh proyek: **teks/id yang ramah manusia → nomor baris yang ramah matriks**.

### Penjaga masukan

```python
def rekomendasi_serupa(judul, top_n=10, min_rating=5):
    if judul not in judul_ke_idx:
        kandidat = [j for j in judul_ke_idx if judul.lower() in j.lower()][:5]
        raise KeyError(f'Judul "{judul}" tidak ditemukan. Saran: {kandidat}')
```

Ini *guard clause* — pemeriksaan di awal fungsi yang langsung berhenti kalau masukannya tidak sah.

Yang membedakannya dari penjagaan biasa: ia **tidak sekadar menolak**, tapi mencari judul yang mengandung teks yang kamu ketik dan menawarkannya sebagai saran. `judul.lower() in j.lower()` adalah pencarian substring tanpa peduli besar-kecil huruf, `[:5]` membatasi saran jadi lima.

Jadi kalau kamu mengetik `Godfather`, pesan errornya menyebutkan `Godfather, The` — bukan sekadar "tidak ditemukan". Ini penting karena judul MovieLens punya format tak lazim (`Godfather, The`, bukan `The Godfather`), dan tanpa saran itu orang akan mengira filmnya memang tidak ada.

### Menghitung kemiripan ke seluruh katalog

```python
    i = judul_ke_idx[judul]
    kemiripan = np.asarray((tfidf[i] @ tfidf.T).todense()).ravel()
```

**Ini baris terpenting Mode A.** Dibaca dari dalam:

| Potongan | Bentuk | Artinya |
|---|---|---|
| `tfidf[i]` | 1 × 1.461 | vektor konten film yang ditanyakan |
| `tfidf.T` | 1.461 × 8.246 | seluruh katalog, ditranspos |
| `tfidf[i] @ tfidf.T` | 1 × 8.246 | kemiripan film `i` terhadap **semua** film |
| `.todense()` | matriks 1 × 8.246 | ubah dari sparse ke padat |
| `.ravel()` | array 8.246 | pipihkan jadi satu dimensi |

Satu perkalian matriks menghasilkan kemiripan terhadap **8.246 film sekaligus** — tanpa satu pun perulangan Python.

Dan ingat: `tfidf` sudah dinormalisasi L2 di Tahap 8a, sehingga panjang tiap vektor persis 1. Karena itu perkalian titik ini **sudah setara dengan cosine similarity** — tidak perlu membagi dengan panjang vektor lagi.

### Dua penyaringan

```python
    kemiripan[i] = -1                                                  # jangan rekomendasikan dirinya
    kemiripan[meta.jumlah_rating_latih.to_numpy() < min_rating] = -1   # buang film tanpa bukti minat
```

Baris pertama jelas: film paling mirip dengan dirinya sendiri (kemiripan 1,0), jadi harus dicoret.

Baris kedua memakai **boolean mask indexing**. `meta.jumlah_rating_latih.to_numpy() < min_rating` menghasilkan array True/False sepanjang 8.246; menaruhnya di dalam `[ ]` berarti "ubah semua posisi yang bernilai True". Satu baris ini mencoret ribuan film sekaligus.

Kenapa `-1` dan bukan `0`? Karena cosine similarity paling kecil bernilai 0 (tidak berbagi token apa pun). Kalau dicoret dengan 0, film tercoret masih bisa seri dengan film sah yang kebetulan kemiripannya 0. Nilai `-1` menjamin mereka selalu kalah.

**Kenapa film berdata sedikit dibuang.** Tanpa saringan ini, rekomendasi *Toy Story* akan dipenuhi animasi antah-berantah yang genrenya kebetulan sama persis. Mirip secara metadata, tapi tak berguna sebagai rekomendasi.

### Mengurutkan dengan pemecah seri

```python
    urutan = np.lexsort((-meta.jumlah_rating_latih.to_numpy(), -kemiripan))[:top_n]
```

**Baris yang paling sering disalahpahami di seluruh proyek.**

`np.lexsort` mengurutkan berdasarkan banyak kunci, tapi dengan aturan yang berlawanan dengan dugaan kebanyakan orang: **kunci terakhir adalah kunci utama**, yang sebelumnya jadi pemecah seri.

```python
np.lexsort((pemecah_seri, kunci_utama))
                          └──────────┘ ← yang ini diproses lebih dulu
```

Jadi di sini: urutkan menurut `-kemiripan` (utama), lalu untuk yang nilainya seri, urutkan menurut `-jumlah_rating_latih`.

Tanda minus dipakai karena `lexsort` selalu mengurutkan **menaik**. Menegatifkan nilai membalik arahnya jadi menurun — trik lazim di NumPy.

**Kenapa butuh pemecah seri?** Karena kemiripan sangat sering seri. Untuk *The Godfather* (`Crime|Drama`), tiga film teratas semuanya berkemiripan **1,000**:

| # | Judul | Kemiripan | Jml rating |
|---|---|---|---|
| 1 | Goodfellas | 1,000 | 107 |
| 2 | Casino | 1,000 | 69 |
| 3 | Donnie Brasco | 1,000 | 41 |
| 4 | Godfather: Part II, The | 0,851 | 113 |

Ketiganya punya himpunan token yang **persis sama**. Tanpa pemecah seri, urutannya ditentukan kebetulan posisi di katalog. Dengan pemecah seri popularitas, setidaknya yang lebih dikenal muncul lebih dulu.

Perhatikan baik-baik hasilnya: *Godfather Part II* — sekuel langsungnya — justru **kalah** dari tiga film itu. Ini bukan bug, melainkan **batas atas informasi** yang tersedia. Dengan 19 genre untuk 9.742 film, tabrakan seperti ini tak terhindarkan, dan inilah alasan pokok CBF kalah telak pada metrik ketepatan.

Bandingkan dengan *Toy Story*, yang punya tag pembeda:

| # | Judul | Kemiripan |
|---|---|---|
| 1 | Bug's Life, A | 0,839 |
| 2 | Toy Story 2 | 0,744 |
| 3 | Monsters, Inc. | 0,607 |
| 4 | Antz | 0,607 |

Kemiripannya **tidak seri di puncak** karena *A Bug's Life* dan *Toy Story 2* sama-sama bertag `pixar`. Tag berhasil membedakan di sini — sayangnya cuma 16% katalog yang punya tag.

### Merakit hasil

```python
    hasil = meta.iloc[urutan][['judul_bersih', 'tahun_rilis', 'genres', 'jumlah_rating_latih']].copy()
    hasil.insert(0, 'peringkat', np.arange(1, len(urutan) + 1))
    hasil['kemiripan'] = kemiripan[urutan].round(3)
    return hasil.reset_index(drop=True)
```

- `meta.iloc[urutan]` — ambil baris berdasarkan **posisi** (bukan label). `iloc` yang benar di sini, bukan `loc`, karena `urutan` berisi nomor posisi.
- `.copy()` — mencegah peringatan `SettingWithCopyWarning` saat kolom ditambahkan sesudahnya.
- `.insert(0, ...)` — sisipkan kolom di posisi **paling kiri**. Kalau memakai `hasil['peringkat'] = ...` biasa, kolomnya nempel di kanan dan tabelnya kurang enak dibaca.
- `kemiripan[urutan]` — ambil nilai kemiripan mengikuti urutan yang sama, supaya sejajar dengan barisnya.

---

## Mode B — Rekomendasi personal

### Pembangun matriks konten

```python
def matriks_konten(kolom):
    vec = TfidfVectorizer(token_pattern=r'\S+')
    return normalize(vec.fit_transform(meta[kolom].fillna('')))
```

Fungsi kecil untuk keperluan ablasi: membangun matriks TF-IDF dari kolom teks mana pun di `meta`.

Perhatikan `vec` dibuat **baru di dalam fungsi**, bukan memakai `vectorizer` global. Ini disengaja — kosakata dan bobot IDF harus dihitung ulang untuk setiap representasi. Kalau memakai vectorizer lama, varian "genre saja" akan memakai IDF yang dihitung dari genre **dan** tag, sehingga perbandingannya tidak jujur.

`.fillna('')` adalah jaring pengaman: `TfidfVectorizer` melempar error kalau menemui `NaN`.

### Membangun skor CBF

```python
def bangun_skor_cbf(data_latih, matriks_isi=None, gunakan_deviasi=False, ambang=AMBANG_SUKA):
    X = tfidf if matriks_isi is None else matriks_isi
```

Pola `None` sebagai nilai bawaan. Kenapa tidak langsung `matriks_isi=tfidf`? Karena nilai bawaan di Python dievaluasi **sekali saat fungsi didefinisikan**, bukan setiap kali dipanggil. Kalau `tfidf` berubah setelahnya, fungsi akan tetap memakai versi lama. Pola `None` membuat penentuannya terjadi saat pemanggilan.

### Dua cara membangun profil

```python
    if gunakan_deviasi:
        dipakai = data_latih
        bobot = data_latih.rating - data_latih.groupby('userId').rating.transform('mean')
    else:
        dipakai = data_latih[data_latih.rating >= ambang]
        bobot = dipakai.rating
```

**Cara 1 — hanya film yang disukai** (`gunakan_deviasi=False`). Saring rating ≥ 4, pakai nilai ratingnya sebagai bobot. Film rating 5 menyumbang lebih besar daripada film rating 4. Film yang tidak disukai diabaikan sama sekali.

**Cara 2 — deviasi dari kebiasaan** (`gunakan_deviasi=True`). Pakai **seluruh** rating, tapi bobotnya `rating − rata-rata pengguna itu`. Perhatikan `transform('mean')` — pola yang sama seperti Tahap 7: hitung rata-rata per pengguna, lalu siarkan kembali ke setiap baris.

Akibatnya bobot bisa **negatif**. Kalau seseorang biasanya memberi 4,0 lalu menilai sebuah film 2,0, bobotnya −2,0 dan vektor konten film itu **mengurangi** profilnya. Secara teori ini lebih kaya: profil tahu apa yang **tidak** disukai, bukan cuma apa yang disukai.

### Baris paling penting di Mode B

```python
    W = sp.csr_matrix((bobot.to_numpy(float),
                       (dipakai.userId.map(pengguna_ke_idx), dipakai.movieId.map(film_ke_idx))),
                      shape=(N_PENGGUNA, N_FILM))
    profil = normalize(W @ X)
    return np.asarray((profil @ X.T).todense())
```

`W` adalah matriks bobot berukuran 610 × 8.246: sel `(u, i)` berisi seberapa besar film `i` menyumbang ke profil pengguna `u`, dan nol kalau tidak menyumbang.

**Lalu `W @ X` — di sinilah keajaibannya.** Rumus profil di laporan berbunyi:

$$\mathbf{p}_u = \sum_{i \in \mathcal{L}_u} r_{ui} \, \mathbf{v}_i$$

yaitu "jumlahkan vektor konten semua film yang disukai, masing-masing dikali ratingnya". Kalau ditulis sebagai perulangan:

```python
for u in range(N_PENGGUNA):
    for i in film_yang_disukai(u):
        profil[u] += bobot[u, i] * X[i]
```

Perkalian matriks `W @ X` mengerjakan **persis itu** — untuk seluruh 610 pengguna sekaligus, dalam satu operasi. Karena begitulah definisi perkalian matriks: baris ke-*u* hasilnya adalah jumlah berbobot dari baris-baris `X`, dengan bobot diambil dari baris ke-*u* milik `W`.

Ukurannya: (610 × 8.246) @ (8.246 × 1.461) = **610 × 1.461**. Satu baris = satu profil pengguna.

**`normalize(...)`** menjadikan panjang tiap profil persis 1, sehingga langkah berikutnya cukup perkalian titik.

**`profil @ X.T`** menghasilkan (610 × 1.461) @ (1.461 × 8.246) = **610 × 8.246** — skor setiap pengguna terhadap setiap film. Ini matriks yang nanti dipakai fungsi evaluasi dan `rekomendasi_topn`.

Perhatikan polanya: kedua langkah adalah perkalian matriks, dan keduanya menghasilkan cosine similarity karena semua vektornya sudah dinormalisasi. Seluruh model CBF pada dasarnya **dua perkalian matriks** — tidak lebih.

---

## Memilih varian lewat data validasi

```python
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
```

Kamus `varian_cbf` menyimpan **argumen** tiap varian, bukan hasilnya. `**argumen` membongkar kamus jadi argumen bernama, sehingga `dict(matriks_isi=..., gunakan_deviasi=False)` berubah menjadi `bangun_skor_cbf(train_fit, matriks_isi=..., gunakan_deviasi=False)`.

`{'Varian': nama, **metrik}` memakai trik yang sama untuk menggabungkan kamus — hasilnya satu baris berisi nama varian plus keempat metriknya.

**Yang penting diperhatikan: `train_fit`, `relevan_validasi`, `sudah_ditonton_fit`, `pool_fit`.** Semuanya versi validasi, bukan versi uji. Keputusan rancangan diambil **tanpa pernah melihat data uji** — inilah yang membuat angka akhir di laporan bisa dipercaya.

Hasilnya:

| Varian | Precision@10 | Recall@10 | NDCG@10 | Coverage@10 |
|---|---|---|---|---|
| **Genre saja** | **0,0101** | **0,0104** | **0,0143** | 0,4400 |
| Genre + tag, profil deviasi | 0,0098 | 0,0104 | 0,0109 | 0,5858 |
| Genre + tag | 0,0073 | 0,0073 | 0,0107 | 0,3615 |

**Menambahkan tag justru memperburuk rekomendasi personal.** Penjelasannya: tag hanya ada di 16% katalog dan berasal dari 58 orang. Kalau profil seseorang kebetulan terbentuk dari beberapa film bertag banyak, profil itu ikut condong ke kosakata sempit yang tidak dimiliki mayoritas kandidat, sehingga pencocokannya jadi kurang stabil.

Profil deviasi memberi **cakupan katalog terluas** (0,5858) tapi tidak lebih tepat — sinyal negatifnya rupanya lebih banyak menyebarkan rekomendasi ketimbang menajamkannya.

Dan yang paling penting: **ketiganya jauh di bawah baseline populer (Precision@10 = 0,0563).** Ini petunjuk bahwa keterbatasan CBF **bukan soal cara menyusun profil**, melainkan soal resolusi metadata yang tersedia.

### Pembagian peran antar-mode

Karena hasil di atas, proyek menetapkan pembagian yang berbeda untuk dua mode:

- **Mode A tetap memakai genre + tag** — di sini tag terbukti membantu (lihat perbedaan *Toy Story* vs *The Godfather* di atas), dan penjelasannya ke pengguna jadi lebih meyakinkan.
- **Mode B memakai varian pemenang validasi**, yaitu genre saja.

## Model final

```python
skor_cbf = bangun_skor_cbf(train_penuh, **varian_cbf[varian_terpilih])

profil_kosong = int((np.abs(skor_cbf).sum(axis=1) == 0).sum())
```

Varian pemenang dilatih ulang pada `train_penuh` — gabungan `train_fit` dan `validasi` — supaya model final memakai seluruh data yang tersedia sebelum diuji.

Baris `profil_kosong` menghitung pengguna yang seluruh skornya nol. Kenapa `np.abs` dulu? Karena pada varian deviasi, skor bisa positif dan negatif lalu **saling meniadakan** saat dijumlahkan — nilai mutlak memastikan yang terdeteksi benar-benar baris nol, bukan baris yang kebetulan berimbang.

Hasilnya **3 pengguna tanpa profil**: mereka tidak pernah memberi rating ≥ 4 pada data latih, sehingga tidak ada film yang bisa dipakai menyusun profilnya. Bentuk *cold start* yang halus — mereka punya riwayat, tapi tak satu pun berupa sinyal positif.

Keluaran akhirnya matriks **610 × 8.246** dengan rentang skor 0,000 sampai 1,000.

---

## Ringkasan alur CBF

```
meta.content_soup  ──TF-IDF──►  X (8.246 × 1.461, ternormalisasi)
                                 │
        ┌────────────────────────┴────────────────────────┐
        │                                                 │
   MODE A                                             MODE B
   tfidf[i] @ tfidf.T                        W (bobot rating) @ X
        │                                                 │
   kemiripan 1 × 8.246                         profil 610 × 1.461
        │                                                 │
   coret diri + film sepi                        normalisasi L2
        │                                                 │
   lexsort + pemecah seri                        profil @ X.T
        │                                                 │
   top-10 film mirip                        skor 610 × 8.246
```

Seluruh model CBF terdiri dari **perkalian matriks dan pengurutan** — tidak ada pelatihan, tidak ada iterasi, tidak ada keacakan. Itulah sebabnya hasilnya sama persis setiap kali dijalankan, tak peduli berapa pun `SEED`-nya.

---

---

# Bagian 4 — Modeling: Matrix Factorization (SGD)

Sumber: [`parts/p4.py`](parts/p4.py) · Notebook: Bab 5.2

Kalau CBF cuma dua perkalian matriks tanpa pelatihan, MF sebaliknya: ada parameter yang **dipelajari** lewat ribuan langkah kecil. Di sinilah kode proyek paling padat.

## Persiapan sebelum perulangan

```python
def latih_mf(data_latih, n_faktor=50, reg=0.05, lr=0.01, epochs=25,
             data_pantau=None, seed=SEED, verbose=False):
    u_idx = data_latih.userId.map(pengguna_ke_idx).to_numpy()
    i_idx = data_latih.movieId.map(film_ke_idx).to_numpy()
    r = data_latih.rating.to_numpy(float)
```

Tiga baris ini menerjemahkan DataFrame jadi **tiga array NumPy sejajar**: `u_idx[t]`, `i_idx[t]`, dan `r[t]` bersama-sama menggambarkan satu rating.

`.to_numpy()` di sini bukan kerapian, tapi **keputusan performa**. Perulangan di bawah berjalan 80.896 × 25 = **lebih dari dua juta kali**. Mengakses `df.iloc[t]` sebanyak dua juta kali akan memakan waktu berjam-jam; mengakses `u_idx[t]` pada array NumPy hitungan menit.

## Inisialisasi — dan satu keputusan yang gampang dianggap sepele

```python
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.05, (N_PENGGUNA, n_faktor))
    Q = rng.normal(0, 0.05, (N_FILM, n_faktor))
    bu = np.zeros(N_PENGGUNA)
    bi = np.zeros(N_FILM)
    mu = r.mean()
```

Perhatikan ketidaksimetrisannya: **`P` dan `Q` diisi angka acak, tapi `bu` dan `bi` diisi nol.** Ini bukan kelalaian.

**Kenapa `P` dan `Q` tidak boleh nol.** Kalau keduanya nol, coba telusuri satu langkah pembaruan:

```
p_u @ q_i = 0            → prediksi cuma dari bias
P[u] += lr * (galat * q_i - reg * p_u)
       = lr * (galat * 0  - reg * 0) = 0     ← tidak bergerak
Q[i] += lr * (galat * p_u - reg * q_i)
       = lr * (galat * 0  - reg * 0) = 0     ← tidak bergerak
```

Faktor latennya **terkunci nol selamanya**. Modelnya jalan tanpa error, hasilnya cuma sedikit lebih baik daripada tebakan rata-rata, dan kamu tidak akan pernah tahu kenapa. Ini disebut *symmetry breaking*: keacakan awal diperlukan supaya tiap dimensi punya titik berangkat berbeda dan bisa berkembang ke arah masing-masing.

**Kenapa `bu` dan `bi` boleh nol.** Karena gradiennya tidak bergantung pada parameter lain:

```
bu[u] += lr * (galat - reg * bu[u])
```

Bahkan saat `bu[u]` nol, sukunya masih `lr * galat` yang bukan nol. Ia langsung bergerak sejak langkah pertama.

Skala `0,05` dipilih kecil dengan sengaja: cukup besar untuk memecah simetri, cukup kecil supaya prediksi awal tidak melenceng jauh dan pelatihan tidak dimulai dari lubang yang dalam.

**`mu` dihitung sekali dan tidak pernah diperbarui.** Ia bukan parameter yang dipelajari, melainkan titik acuan tetap. Semua yang dipelajari model adalah *simpangan* dari `mu`.

## Menyiapkan data pemantau

```python
    if data_pantau is not None:
        pantau = data_pantau[data_pantau.movieId.isin(film_ke_idx)]
        pu_idx = pantau.userId.map(pengguna_ke_idx).to_numpy()
        pi_idx = pantau.movieId.map(film_ke_idx).to_numpy()
        pr = pantau.rating.to_numpy(float)
```

`data_pantau` adalah data validasi yang dievaluasi tiap epoch untuk menggambar kurva pembelajaran. Dua hal penting:

- `isin(film_ke_idx)` menyaring film yang tak dikenal model. Tanpa ini, `map` menghasilkan `NaN` dan pengindeksan array akan gagal.
- Penyiapan indeksnya **di luar perulangan epoch**. Kalau ditaruh di dalam, pekerjaan yang sama diulang 25 kali percuma.

## Perulangan pelatihan

```python
    urutan = np.arange(len(r))
    for epoch in range(1, epochs + 1):
        rng.shuffle(urutan)
        for t in urutan:
```

**Kenapa perlu diacak tiap epoch.** Ingat Tahap 6: `ratings` diurutkan `['userId','timestamp']`. Tanpa pengacakan, SGD akan memproses seluruh 232 rating milik pengguna 1, lalu seluruh rating pengguna 2, dan seterusnya. Akibatnya `P[0]` disetel berkali-kali berturut-turut lalu ditinggalkan sepanjang sisa epoch — pembaruannya jadi tersentak-sentak alih-alih halus.

**Yang diacak adalah `urutan`, bukan datanya.** Mengacak array indeks 80.896 elemen jauh lebih murah daripada memindahkan tiga array data. Trik lazim yang layak ditiru.

## Inti pembaruan — dan baris paling halus di seluruh proyek

```python
            u, i, nilai = u_idx[t], i_idx[t], r[t]
            p_u = P[u].copy()
            q_i = Q[i]
            galat = nilai - (mu + bu[u] + bi[i] + p_u @ q_i)
            bu[u] += lr * (galat - reg * bu[u])
            bi[i] += lr * (galat - reg * bi[i])
            P[u] += lr * (galat * q_i - reg * p_u)
            Q[i] += lr * (galat * p_u - reg * q_i)
```

**Kenapa `P[u].copy()` tapi `Q[i]` tidak?**

Ini bukan inkonsistensi — ini **wajib**, dan salah satu kekeliruan paling umum saat orang menulis MF sendiri.

Di NumPy, `P[u]` mengembalikan **view**, bukan salinan: ia menunjuk ke memori yang sama dengan baris `P[u]`. Sekarang telusuri dua baris terakhir:

```
P[u] += ...     ← baris P berubah di tempat (in place)
Q[i] += lr * (galat * p_u - reg * q_i)
                       └──┘ butuh p_u yang LAMA
```

Matematikanya menuntut **pembaruan serentak**: `P` dan `Q` sama-sama dihitung dari nilai sebelum langkah ini. Kalau `p_u` cuma view, ia sudah ikut berubah saat `P[u]` diperbarui, sehingga baris `Q[i]` memakai nilai yang **salah** — bukan nilai lama, tapi campuran.

Bug ini tidak menimbulkan error. Modelnya tetap jalan, RMSE-nya cuma sedikit lebih buruk, dan penyebabnya nyaris mustahil dilacak.

Lalu kenapa `q_i` aman tanpa salinan? Karena `Q[i]` baru diubah di baris **terakhir**, dan Python menghitung ruas kanan lebih dulu sebelum menugaskan. Saat `q_i` dibaca di baris `P[u] += ...`, `Q` belum tersentuh sama sekali.

Bandingkan kode itu dengan rumus di laporan:

$$\mathbf{p}_u \leftarrow \mathbf{p}_u + \eta\left(e_{ui}\mathbf{q}_i - \lambda \mathbf{p}_u\right)$$

Sama persis, huruf demi huruf. `lr` adalah $\eta$, `reg` adalah $\lambda$, `galat` adalah $e_{ui}$.

## Evaluasi tiap akhir epoch

```python
        pred_latih = np.clip(mu + bu[u_idx] + bi[i_idx] + np.sum(P[u_idx] * Q[i_idx], axis=1), 0.5, 5.0)
```

Satu baris ini memprediksi **seluruh 80.896 rating sekaligus**, tanpa perulangan.

`P[u_idx]` adalah *fancy indexing*: `u_idx` berisi 80.896 nomor pengguna, jadi hasilnya matriks 80.896 × 100 — baris ke-*t* adalah vektor laten pemilik rating ke-*t*. Begitu pula `Q[i_idx]`.

**Kenapa `np.sum(A * B, axis=1)` dan bukan `A @ B.T`?** Karena kita cuma butuh perkalian titik **baris ke-t dengan baris ke-t**, bukan semua pasangan. `A @ B.T` akan menghasilkan matriks 80.896 × 80.896 — sekitar 51 GB. Perkalian elemen-per-elemen lalu dijumlahkan per baris memberi hasil yang benar dengan memori sepersekian juta kalinya.

`np.clip(..., 0.5, 5.0)` memangkas prediksi ke rentang rating yang sah. Model kadang menebak 5,3 atau 0,2 — nilai yang mustahil, dan membiarkannya akan memperbesar RMSE tanpa alasan.

## Prediksi untuk peringkat versus untuk RMSE

Proyek ini punya **dua** fungsi keluaran yang berbeda perlakuan, dan bedanya penting.

```python
def skor_matriks_mf(model):
    return model['mu'] + model['bu'][:, None] + model['bi'][None, :] + model['P'] @ model['Q'].T
```

`bu[:, None]` mengubah array 610 elemen jadi matriks **610 × 1**; `bi[None, :]` mengubah 8.246 elemen jadi **1 × 8.246**. Saat dijumlahkan, NumPy melakukan *broadcasting*: yang berukuran 610×1 direntangkan ke samping, yang 1×8.246 direntangkan ke bawah, hasilnya 610 × 8.246. Setiap sel `(u,i)` otomatis berisi `mu + bu[u] + bi[i]`.

**Perhatikan: tidak ada `np.clip` di sini.** Disengaja. Kalau dipangkas di 5,0, semua prediksi di atas 5,0 akan **seri** — dan justru film-film itulah yang mengisi puncak daftar. Pemangkasan akan menghancurkan urutannya. Inilah sebabnya top-10 MF di laporan menampilkan angka seperti 5,2921 dan 5,2897: nilainya mustahil sebagai rating, tapi **urutannya** yang kita perlukan.

```python
def prediksi_mf(model, data):
    u = data.userId.map(pengguna_ke_idx).to_numpy()
    ada = data.movieId.isin(film_ke_idx).to_numpy()
    i = data.movieId.map(film_ke_idx).fillna(0).astype(int).to_numpy()
    penuh = model['mu'] + model['bu'][u] + model['bi'][i] + np.sum(model['P'][u] * model['Q'][i], axis=1)
    cadangan = model['mu'] + model['bu'][u]
    return np.clip(np.where(ada, penuh, cadangan), 0.5, 5.0)
```

Fungsi ini untuk RMSE, jadi **di sini justru dipangkas**.

Yang menarik adalah penanganan *cold start*. `.fillna(0)` mengganti film tak dikenal dengan indeks 0 — bukan karena film nomor 0 relevan, tapi supaya pengindeksan array tidak gagal. Nilai `penuh` untuk baris itu memang omong kosong, tetapi `np.where(ada, penuh, cadangan)` langsung membuangnya dan memakai `cadangan` sebagai gantinya.

`cadangan = mu + bu[u]` adalah tebakan terbaik ketika film tak punya riwayat: rata-rata global, disesuaikan dengan kemurahan hati pengguna itu. Inilah yang menangani 8,4% rating uji yang menyangkut film asing.

## Penyetelan dan pemilihan epoch

```python
for konf in grid_mf:
    model_uji = latih_mf(train_fit, epochs=25, data_pantau=validasi, **konf)
    rmse_val = model_uji['riwayat'].rmse_pantau.to_numpy()
    epoch_terbaik = int(np.argmin(rmse_val)) + 1
```

`np.argmin` memberi **posisi** nilai terkecil (mulai 0), jadi `+ 1` mengubahnya jadi nomor epoch (mulai 1).

Pola ini adalah *early stopping* yang dikerjakan belakangan: model dilatih penuh 25 epoch, riwayatnya dicatat, lalu titik terbaiknya dipilih setelah semuanya selesai. Lebih boros komputasi daripada berhenti di tengah jalan, tetapi memberi kurva pembelajaran utuh yang bisa dilihat — dan untuk data sekecil ini biayanya tidak berarti.

Hasil penyetelannya:

| $k$ | $\lambda$ | RMSE latih | **RMSE validasi** | Epoch terbaik |
|---|---|---|---|---|
| **100** | **0,05** | 0,5753 | **0,8689** | 25 |
| 50 | 0,05 | 0,6378 | 0,8755 | 24 |
| 20 | 0,05 | 0,7134 | 0,8777 | 25 |
| 50 | 0,10 | 0,7645 | 0,8803 | 20 |

Perhatikan kolom RMSE latih: makin banyak faktor, makin kecil galat pada data latih (0,7134 → 0,5753) — model makin pandai **menghafal**. Tapi RMSE validasi cuma membaik tipis (0,8777 → 0,8689). Jarak antara 0,5753 dan 0,8689 itulah wujud *overfitting* dalam angka.

## Model final dan bias yang menjelaskan segalanya

```python
model_mf = latih_mf(train_penuh, n_faktor=100, reg=0.05, lr=0.01, epochs=25)

urut_bias = np.argsort(-model_mf['bi'])
```

`np.argsort(-x)` adalah cara lazim mengurutkan menurun di NumPy: `argsort` selalu menaik, jadi nilainya dinegatifkan lebih dulu. Hasilnya:

| Film | Jumlah rating latih | $b_i$ |
|---|---|---|
| Yojimbo | 11 | 0,931 |
| Paths of Glory | 8 | 0,905 |
| Guess Who's Coming to Dinner | 9 | 0,880 |
| His Girl Friday | 12 | 0,878 |
| Lawrence of Arabia | 39 | 0,871 |

**Inilah bukti langsung dari kelemahan MF.** Semua film berbias tertinggi cuma punya 8–39 rating. Regularisasi $\lambda = 0{,}05$ terpilih karena memberi RMSE validasi terbaik — tapi terlalu lemah untuk menarik film berdata sedikit kembali ke rata-rata.

Telusuri akibatnya sampai ke keluaran akhir: bias tinggi → prediksi rating tinggi → mengisi puncak daftar → Precision@10 ambruk ke 0,0232, kalah dari baseline populer (0,0563).

Rantai sebabnya lengkap dan bisa ditunjuk baris kodenya:

```
fungsi objektif menjumlahkan pada rating TERAMATI saja
        │
        ▼
lambda kecil terasa optimal bagi RMSE — tak ada yang menghukumnya
        │
        ▼
b_i film berdata sedikit melambung  ← tabel di atas
        │
        ▼
top-10 dipenuhi klasik lawas        ← Sunset Blvd., High Noon, dst.
        │
        ▼
NDCG@10 = 0,0278 — kalah dari daftar film populer
```

---

# Bagian 5 — Modeling: Implicit ALS

Sumber: [`parts/p5.py`](parts/p5.py) · Notebook: Bab 5.3

## Membangun matriks interaksi biner

```python
def latih_ials(data_latih, n_faktor=32, reg=2.0, alpha=10.0, iterasi=15,
               ambang=AMBANG_SUKA, seed=SEED):
    positif = data_latih[data_latih.rating >= ambang]
    C = sp.csr_matrix((np.ones(len(positif)),
                       (positif.userId.map(pengguna_ke_idx), positif.movieId.map(film_ke_idx))),
                      shape=(N_PENGGUNA, N_FILM))
    C_pengguna, C_film = C.tocsr(), C.tocsc()
```

Perhatikan `np.ones(len(positif))` — **nilai ratingnya dibuang**. Rating 4,0 dan 5,0 sama-sama jadi 1. Inilah perbedaan mendasar dengan MF: iALS tidak peduli seberapa suka, cuma peduli **suka atau tidak**.

**Baris `tocsr()` dan `tocsc()` layak diperhatikan.** Matriks yang sama disimpan dalam dua orientasi:

| Format | Menyimpan | Cepat untuk |
|---|---|---|
| **CSR** (*Compressed Sparse Row*) | per baris | "film apa saja yang disukai pengguna u" |
| **CSC** (*Compressed Sparse Column*) | per kolom | "siapa saja yang menyukai film i" |

ALS bergantian menyapu pengguna lalu film, jadi ia butuh **kedua** arah. Menyimpan dua salinan memakan memori dua kali lipat (tetap cuma 0,6 MB) tapi menghemat waktu berlipat-lipat — kalau cuma punya CSR, mencari "siapa menyukai film i" harus menyisir seluruh matriks.

## Inisialisasi

```python
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.01, (N_PENGGUNA, n_faktor))
    Y = rng.normal(0, 0.01, (N_FILM, n_faktor))
    I_reg = reg * np.eye(n_faktor)
```

Sama seperti MF, keacakan awal memecah simetri. Tapi **tidak ada `bu` dan `bi`** di sini — iALS tidak memakai suku bias sama sekali, karena yang diprediksi bukan nilai rating melainkan skor preferensi tak berskala.

`I_reg` adalah matriks identitas dikali $\lambda$, dihitung **sekali di luar semua perulangan**. Ia dipakai ribuan kali di dalam; menghitungnya ulang tiap kali adalah pemborosan murni.

## Jantung ALS

```python
    for _ in range(iterasi):
        YtY = Y.T @ Y + I_reg
        for u in range(N_PENGGUNA):
            item = C_pengguna.indices[C_pengguna.indptr[u]:C_pengguna.indptr[u + 1]]
            if len(item) == 0:
                X[u] = 0
                continue
            Yi = Y[item]
            A = YtY + alpha * (Yi.T @ Yi)
            b = (1 + alpha) * Yi.sum(axis=0)
            X[u] = np.linalg.solve(A, b)
```

**Baris `item = ...`** adalah pembacaan langsung struktur CSR yang dibahas di Bagian 2: `indptr[u]` sampai `indptr[u+1]` menandai potongan milik baris `u`, dan `indices` di rentang itu berisi nomor kolomnya. Diterjemahkan: **"daftar film yang disukai pengguna u"**.

**Kenapa `YtY` di luar perulangan pengguna.** Rumusnya:

$$\mathbf{x}_u = \left(Y^{\top}C^u Y + \lambda I\right)^{-1} Y^{\top} C^u \mathbf{p}(u)$$

Karena $c_{ui} = 1 + \alpha$ untuk yang teramati dan $1$ untuk sisanya, matriks $C^u$ bisa dipecah jadi "identitas untuk semua, ditambah $\alpha$ untuk yang teramati saja":

$$Y^{\top}C^u Y = \underbrace{Y^{\top}Y}_{\text{sama untuk semua pengguna}} + \alpha \underbrace{Y_u^{\top}Y_u}_{\text{hanya film yang disukai}}$$

Suku pertama tidak bergantung pada `u` sama sekali — makanya dihitung **satu kali** untuk seluruh 610 pengguna. Suku kedua cuma melibatkan puluhan film, bukan 8.246.

Tanpa pemecahan ini, tiap pengguna butuh perkalian matriks 8.246 × 32 — 610 kali per iterasi, 15 iterasi. Dengan pemecahan ini, satu iterasi selesai dalam sepersekian detik.

Kode `A = YtY + alpha * (Yi.T @ Yi)` adalah terjemahan harfiah persamaan di atas. Begitu pula `b = (1 + alpha) * Yi.sum(axis=0)`, yang merupakan $Y^{\top}C^u\mathbf{p}(u)$ — karena $p_{ui}$ bernilai 1 hanya pada film yang disukai, jumlahnya menyusut jadi "jumlahkan vektor film yang disukai, kali $(1+\alpha)$".

**`np.linalg.solve(A, b)` dan bukan `np.linalg.inv(A) @ b`.** Keduanya menjawab pertanyaan yang sama, tapi `solve` menyelesaikan sistem persamaan secara langsung tanpa pernah membentuk matriks inversnya. Lebih cepat, dan yang lebih penting, **lebih stabil secara numerik** — menghitung invers memperbesar galat pembulatan, terutama saat matriksnya nyaris singular. Ini kaidah umum yang layak dibawa ke mana pun: kalau yang kamu butuhkan adalah $A^{-1}b$, pakai `solve`, jangan `inv`.

**`if len(item) == 0: X[u] = 0`** menangani pengguna yang tak pernah memberi rating ≥ 4. Tanpa penjagaan ini, `A` menjadi $\lambda I$ dan `b` menjadi nol — `solve` tetap berhasil dan mengembalikan nol juga, tapi memanggilnya cuma buang waktu. Penjagaan ini juga membuat maksudnya tersurat: **pengguna ini memang tidak punya profil.**

## Paruh kedua: peran ditukar

```python
        XtX = X.T @ X + I_reg
        for i in range(N_FILM):
            pengguna = C_film.indices[C_film.indptr[i]:C_film.indptr[i + 1]]
            if len(pengguna) == 0:
                Y[i] = 0
                continue
            Xu = X[pengguna]
            A = XtX + alpha * (Xu.T @ Xu)
            b = (1 + alpha) * Xu.sum(axis=0)
            Y[i] = np.linalg.solve(A, b)
```

Blok ini **cermin persis** dari blok sebelumnya — `Y` ditukar `X`, pengguna ditukar film, CSR ditukar CSC. Itulah arti *alternating*: bekukan satu sisi, selesaikan sisi lain secara tertutup, lalu tukar.

Perhatikan `X` yang dipakai di sini adalah `X` yang **baru saja diperbarui** di paruh pertama, bukan salinan lama. Berbeda dengan MF yang menuntut pembaruan serentak, ALS memang dirancang berurutan — tiap paruh menyelesaikan submasalahnya secara optimal terhadap keadaan sisi lain saat ini.

## Perbandingan langsung: MF versus iALS

| Aspek | Matrix Factorization | Implicit ALS |
|---|---|---|
| Masukan | Nilai rating 0,5–5,0 | Biner: suka (≥4) atau tidak |
| Pasangan yang dipelajari | Hanya yang teramati | **Seluruh 5 juta pasangan** |
| Optimasi | SGD, dua juta langkah kecil | ALS, 15 iterasi solusi tertutup |
| Suku bias | Ada ($\mu, b_u, b_i$) | Tidak ada |
| Keluaran | Taksiran rating | Skor preferensi tak berskala |
| Dievaluasi dengan | RMSE, MAE | Precision, Recall, NDCG |
| $k$ terpilih | 100 | **16** |
| Baris kode inti | 8 baris pembaruan | 8 baris solusi tertutup |

**Kenapa $k$ iALS jauh lebih kecil?** Dari tabel penyetelan, $k=64$ justru lebih buruk (NDCG 0,1031) daripada $k=16$ (0,1231). Dua sebabnya: menyusun peringkat "suka/tidak suka" butuh representasi lebih ringkas daripada menebak angka presisi, dan dengan cuma 610 pengguna, model besar cepat kehilangan daya generalisasi.

---

# Bagian 6 — Baseline dan Penyajian Top-N

## Baseline popularitas

```python
skor_pop = np.tile(jumlah_rating_film.astype(float), (N_PENGGUNA, 1))
```

Satu baris. `np.tile` menyalin array 8.246 elemen sebanyak 610 kali ke bawah, menghasilkan matriks 610 × 8.246 yang **setiap barisnya identik**.

Itulah definisi "tidak dipersonalisasi" dalam bentuk kode: skor film sama persis untuk semua orang. Bentuk matriksnya sengaja dibuat sama dengan model lain supaya bisa masuk ke fungsi evaluasi yang sama tanpa perlakuan khusus — perbandingannya jadi benar-benar setara.

## Menyajikan top-N

```python
def rekomendasi_topn(skor, idx_pengguna, top_n=10, kandidat=None, sudah=None, nama_kolom='skor'):
    kandidat = pool if kandidat is None else kandidat
    sudah = sudah_ditonton if sudah is None else sudah

    s = np.full(N_FILM, -np.inf)
    s[kandidat] = skor[idx_pengguna, kandidat].astype(float)
    s[sudah[idx_pengguna]] = -np.inf
    urutan = np.lexsort((-jumlah_rating_film, -s))[:top_n]
```

**Pola `-np.inf` sebagai penyaring.** Alih-alih menyalin skor lalu membuang yang tidak memenuhi syarat, kodenya mulai dari array yang **seluruhnya minus tak hingga**, lalu mengisi hanya posisi yang boleh direkomendasikan.

Kenapa `-np.inf` dan bukan `-1` seperti di Mode A? Karena di sini skornya bisa datang dari model mana pun, dan skalanya berbeda-beda jauh: skor MF melampaui 5,3, skor CBF tak pernah lebih dari 1, skor popularitas berupa cacahan hingga ratusan. Tak ada satu angka "cukup kecil" yang aman untuk semuanya. `-np.inf` dijamin kalah dari bilangan apa pun.

Dua penyaringan berurutan:
1. `s[kandidat] = ...` — hanya film di *candidate pool* yang dapat skor sungguhan
2. `s[sudah[idx_pengguna]] = -np.inf` — film yang sudah ditonton dikembalikan ke minus tak hingga

Urutannya penting: penyaringan kedua harus **setelah** yang pertama, kalau tidak film yang sudah ditonton akan tertimpa skornya lagi.

`np.lexsort` muncul lagi dengan aturan yang sama seperti Mode A: **kunci terakhir adalah kunci utama**, jadi urut menurut skor lalu popularitas sebagai pemecah seri.

```python
    hasil['relevan?'] = ['YA' if i in relevan_uji.get(idx_pengguna, set()) else '-' for i in urutan]
```

`.get(idx_pengguna, set())` memakai nilai bawaan himpunan kosong — supaya pengguna yang tak punya item relevan pada data uji tidak menyebabkan `KeyError`, melainkan menghasilkan sepuluh tanda `-`.

Kolom inilah yang membuat keluaran top-N di laporan **bisa dinilai**, bukan sekadar dipandang. Tanpa penanda ini, keempat daftar akan terlihat sama masuk akalnya — padahal iALS menghasilkan empat `YA` sementara CBF dan MF nol.

---

# Penutup

Seluruh model di proyek ini, diringkas dalam satu tabel:

| Model | Inti perhitungan | Ada pelatihan? | Deterministik? |
|---|---|---|---|
| Popularitas | `np.tile` dari jumlah rating | Tidak | Ya |
| CBF | Dua perkalian matriks | Tidak | Ya |
| MF | 2 juta langkah SGD | Ya | Ya, berkat `SEED` |
| iALS | 15 iterasi solusi tertutup | Ya | Ya, berkat `SEED` |

Yang menarik: model paling sederhana secara kode (popularitas, satu baris) mengalahkan dua model yang jauh lebih rumit. Dan model pemenang (iALS) bukan yang paling banyak parameternya — $k=16$, seperenam dari MF.

Kalau ada satu pelajaran yang dibawa pulang dari membaca seluruh kode ini: **kerumitan bukan jaminan apa-apa.** Yang menentukan adalah apakah yang dioptimalkan model benar-benar hal yang kamu pedulikan.
