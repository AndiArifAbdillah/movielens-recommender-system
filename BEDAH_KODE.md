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

## Berikutnya

Bedah kode untuk **Matrix Factorization (Bab 5.2)** dan **Implicit ALS (Bab 5.3)** belum ditulis di dokumen ini. Sementara ini, penjelasan konseptual keduanya — termasuk penurunan aturan pembaruan SGD dan solusi tertutup ALS — tersedia di [`BELAJAR.md`](BELAJAR.md) Tingkat 4 dan 5.
