# Panduan Belajar: Memahami Proyek Ini Luar Dalam

Dokumen ini adalah kurikulum untuk membedah proyek ini dari nol sampai kamu bisa menjelaskan setiap barisnya tanpa membuka catatan.

**Ditulis dengan satu asumsi:** kamu bisa Python dasar (variabel, `for`, `def`, list, dict) dan tidak lebih dari itu. Semua konsep lain — TF-IDF, faktorisasi matriks, ALS, NDCG — dijelaskan dari awal.

> **Pendamping dokumen ini.** [`BEDAH_KODE.md`](BEDAH_KODE.md) menjelaskan kodenya baris demi baris. Dokumen yang kamu baca sekarang membangun **pemahaman konsep** lewat latihan; dokumen itu jadi **rujukan implementasi** saat kamu berhadapan langsung dengan kodenya. Keduanya paling enak dipakai berdampingan.

> **Cara memakai dokumen ini.** Jangan dibaca sekali jalan lalu merasa paham. Rasa paham saat membaca itu menipu — otak mengenali penjelasan yang runtut dan menyangkanya sebagai pengetahuan. Yang benar-benar mengendap hanyalah yang kamu kerjakan sendiri. Setiap tingkat punya **Latihan** yang harus benar-benar diketik dan dijalankan, dan **Ujian tingkat** yang harus bisa kamu jawab dengan mulut sendiri, bukan dengan menunjuk kode.

---

## Peta Perjalanan

| Tingkat | Topik | Perkiraan waktu | Bisa dilewati? |
|---|---|---|---|
| 0 | Persiapan — jalankan dulu proyeknya | 1–2 jam | Tidak |
| 1 | Memahami masalahnya, belum kodenya | 2–3 jam | Tidak |
| 2 | Bahasa datanya: pandas dan matriks renggang | 3–4 jam | Ya, kalau pandas sudah lancar |
| 3 | Content-Based Filtering: TF-IDF dan cosine | 4–5 jam | Tidak |
| 4 | Matrix Factorization dan gradient descent | 6–8 jam | Tidak — ini jantungnya |
| 5 | Implicit ALS | 5–6 jam | Tidak |
| 6 | Evaluasi — bagian yang paling sering disepelekan | 4–5 jam | Tidak |
| 7 | Menyatukan semuanya | 2–3 jam | Tidak |

**Total realistis: 27–36 jam.** Kalau dicicil 1,5 jam sehari, sekitar tiga minggu. Jangan dikebut dalam dua hari — konsep di Tingkat 4 dan 5 butuh waktu mengendap.

**Kalau waktumu mepet** (misalnya ada interview minggu depan): kerjakan Tingkat 0, 1, 4, dan 6 saja. Itu inti yang paling sering ditanya.

---

## Tingkat 0 — Jalankan Dulu, Pahami Belakangan

**Tujuan:** memastikan proyeknya benar-benar hidup di komputermu, bukan sekadar teks yang kamu baca.

### Langkah

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
python Proyek_Akhir_Sistem_Rekomendasi.py
```

Tunggu sekitar 3 menit. Dataset diunduh otomatis.

### Latihan 0.1 — Cocokkan angkanya

Di akhir keluaran ada tabel `RINGKASAN AKHIR`. Bandingkan dengan tabel di README. **Harus sama persis** sampai empat angka di belakang koma, karena semua keacakan dikunci pada `SEED = 42`.

Kalau berbeda, berhenti dulu dan cari tahu kenapa. Bisa jadi versi pustakamu berbeda. Ini bukan sekadar kerapian: reproduktibilitas adalah syarat pertama pekerjaan ilmiah yang bisa dipercaya.

### Latihan 0.2 — Kenali medan

Buka `Proyek_Akhir_Sistem_Rekomendasi.ipynb`. Jangan dibaca dulu, cukup **gulung dari atas ke bawah** sambil mencatat di kertas: ada berapa bab besar, dan apa judul masing-masing.

### Latihan 0.3 — Rusak sesuatu dengan sengaja

Buka `parts/p1.py`, ubah `SEED = 42` menjadi `SEED = 7`. Jalankan `python build_notebook.py`. Amati mana angka yang berubah dan mana yang tidak.

**Pertanyaan:** kenapa hasil Content-Based Filtering **tidak** berubah sama sekali, sementara MF dan iALS berubah?

<details>
<summary>Petunjuk jawaban</summary>

CBF sama sekali tidak mengandung komponen acak — TF-IDF dan cosine similarity bersifat deterministik. MF dan iALS memulai dari matriks faktor yang diisi angka acak, dan MF juga mengacak urutan data setiap epoch. Setelah menjawab, kembalikan `SEED` ke 42.
</details>

### ✅ Ujian Tingkat 0

Kamu lulus kalau bisa: menjalankan proyek dari nol, menemukan letak angka Precision@10 pada keluaran, dan menjelaskan kenapa hasilnya identik setiap kali dijalankan.

---

## Tingkat 1 — Pahami Masalahnya Dulu, Bukan Kodenya

**Tujuan:** mengerti *kenapa* sistem rekomendasi sulit. Kalau kamu langsung lompat ke kode, kamu hanya akan hafal sintaks tanpa tahu apa yang sedang diperjuangkan.

### Konsep 1.1 — Kenapa tidak diurutkan berdasarkan rating tertinggi saja?

Ini pertanyaan pertama yang muncul di kepala semua orang, dan jawabannya membuka seluruh proyek.

Coba jalankan ini:

```python
import pandas as pd
r = pd.read_csv('data/ml-latest-small/ratings.csv')
m = pd.read_csv('data/ml-latest-small/movies.csv')

info = r.groupby('movieId').agg(jumlah=('rating','size'), rata=('rating','mean'))
info = info.merge(m[['movieId','title']], on='movieId')
print(info.sort_values('rata', ascending=False).head(10))
```

Yang muncul adalah film-film antah-berantah dengan rata-rata 5,0 — karena **hanya satu orang** yang menilainya. Ada **289 film** seperti itu di dataset.

Pelajarannya: rata-rata dari satu sampel bukan estimasi yang bisa dipercaya. Ini persoalan statistik dasar yang muncul di mana-mana, dan proyek ini menanganinya lewat **regularisasi bias item** (Tingkat 4).

### Konsep 1.2 — Ekor panjang (*long tail*)

```python
rpf = r.groupby('movieId').size().sort_values(ascending=False)
print('Film dengan <5 rating :', (rpf < 5).sum(), f'({(rpf<5).mean()*100:.1f}%)')
print('Film dengan 1 rating  :', (rpf == 1).sum())
print('Film yang menyerap 50% rating:', (rpf.cumsum() <= len(r)*0.5).sum() + 1)
```

659 film (6,8%) menyerap separuh seluruh rating. 62,5% film punya kurang dari lima rating.

Konsekuensinya bercabang dua, dan keduanya membentuk rancangan proyek ini:

1. Untuk film di ekor, *collaborative filtering* **tidak punya bahan** untuk belajar. Karena itu ada *candidate pool* (minimal 5 rating).
2. Model yang cuma menyodorkan film populer akan terlihat lumayan di atas kertas. Karena itu **baseline popularitas wajib ada** sebagai pembanding — tanpa itu kita tidak bisa membedakan "model belajar selera" dari "model ikut arus".

### Konsep 1.3 — Kerenggangan (*sparsity*)

```python
nU, nI = r.userId.nunique(), r.movieId.nunique()
print(f'Sel yang mungkin : {nU*nI:,}')
print(f'Sel yang terisi  : {len(r):,}')
print(f'Kerenggangan     : {(1 - len(r)/(nU*nI))*100:.2f}%')
```

**98,30% kosong.** Bayangkan tabel raksasa dengan 610 baris dan 9.724 kolom, dan hanya 1,7% selnya berisi angka. Seluruh tugas sistem rekomendasi adalah **menebak isi sel yang kosong**.

### Konsep 1.4 — Cold start

Dua bentuk:
- **Item baru** — film yang belum punya rating sama sekali. *Collaborative filtering* buta total terhadapnya. Di data uji proyek ini, **8,4% rating** menyangkut film seperti ini.
- **Pengguna baru** — orang yang belum pernah menilai apa pun. Tidak punya profil.

Inilah alasan CBF tetap punya nilai meski akurasinya kalah: ia hanya butuh metadata, bukan riwayat interaksi.

### Latihan 1.1

Cari sendiri di dataset: berapa film yang ada di `movies.csv` tetapi **tidak pernah muncul** di `ratings.csv`? (Jawaban: 18.) Lalu jelaskan kenapa `katalog` di proyek ini dibangun dari `train_penuh`, bukan dari `movies.csv`.

### Latihan 1.2 — Latihan berpikir, tanpa kode

Bayangkan kamu punya layanan streaming dengan 50.000 film dan 1 juta pengguna. Tuliskan di kertas: tiga cara berbeda untuk memilih 10 film yang ditampilkan di beranda seseorang. Untuk masing-masing, tulis satu kelemahan seriusnya.

Setelah selesai, bandingkan dengan empat pendekatan di proyek ini. Kemungkinan besar kamu sudah menemukan sendiri dua di antaranya.

### ✅ Ujian Tingkat 1

Jawab dengan mulut, bukan dengan menunjuk kode:

1. Kenapa merekomendasikan film dengan rata-rata rating tertinggi adalah ide buruk?
2. Apa itu *long tail*, dan kenapa ia menyulitkan *collaborative filtering*?
3. Kenapa proyek ini repot-repot menyertakan baseline popularitas, padahal baseline itu jelas bukan solusi yang mau dipakai?

---

## Tingkat 2 — Bahasa Datanya

**Bisa dilewati** kalau `groupby`, `merge`, dan `map` di pandas sudah jadi refleks. Kalau belum, jangan dilewati — seluruh tahap persiapan data proyek ini dibangun dari empat operasi itu.

### Konsep 2.1 — Empat operasi pandas yang dipakai proyek ini

```python
# 1. groupby + agg — meringkas per kelompok
r.groupby('movieId').agg(jumlah=('rating','size'), rata=('rating','mean'))

# 2. merge — menggabungkan tabel lewat kunci bersama
movies.merge(tag_per_film, on='movieId', how='left')

# 3. map — menerjemahkan nilai lewat kamus (dipakai untuk encoding id)
train.userId.map(pengguna_ke_idx)

# 4. transform — hasil groupby yang panjangnya sama dengan tabel asal
df.groupby('userId').movieId.transform('size')
```

Yang keempat paling sering membingungkan pemula. Bedanya dengan `agg`: `agg` **meringkas** (610 pengguna → 610 baris), `transform` **menyiarkan kembali** ke setiap baris asal (100.836 baris tetap 100.836 baris, tapi tiap baris kini tahu berapa total rating pemiliknya). Fungsi `bagi_temporal()` bergantung penuh pada perilaku ini.

### Konsep 2.2 — Matriks renggang (*sparse matrix*)

Matriks interaksi kita 610 × 8.246. Kalau disimpan biasa:

```
610 × 8.246 × 8 byte = 38,38 MB
```

Padahal hanya 80.896 sel yang terisi. Format **CSR** (*Compressed Sparse Row*) hanya menyimpan yang terisi: **0,62 MB**, 62 kali lebih hemat.

Cara CSR bekerja — pahami ini dengan contoh kecil:

```python
import numpy as np, scipy.sparse as sp

padat = np.array([[0, 0, 3],
                  [4, 0, 0],
                  [0, 5, 6]])
jarang = sp.csr_matrix(padat)

print('data   :', jarang.data)     # [3 4 5 6]  <- nilai tak-nol saja
print('indices:', jarang.indices)  # [2 0 1 2]  <- nomor kolom tiap nilai
print('indptr :', jarang.indptr)   # [0 1 2 4]  <- di mana tiap baris mulai
```

Bacanya: baris 0 memakai `data[0:1]` (posisi `indptr[0]` sampai `indptr[1]`), yaitu nilai 3 di kolom 2. Baris 2 memakai `data[2:4]`, yaitu nilai 5 di kolom 1 dan 6 di kolom 2.

**Kenapa ini penting untuk dipahami, bukan sekadar dipakai?** Karena fungsi `latih_ials()` mengakses `indptr` dan `indices` secara langsung:

```python
item = C_pengguna.indices[C_pengguna.indptr[u]:C_pengguna.indptr[u + 1]]
```

Baris itu berarti: *"ambil daftar film yang disukai pengguna u"*. Tanpa paham CSR, baris itu tampak seperti sihir.

### Latihan 2.1

Buat matriks interaksi mini dengan tangan: 3 pengguna, 4 film, 5 rating karanganmu sendiri. Bangun `csr_matrix`-nya, lalu cetak `data`, `indices`, dan `indptr`. Tebak isinya **sebelum** mencetak, lalu periksa tebakanmu.

### Latihan 2.2

Tanpa menjalankan kode, ramalkan hasil ini, lalu buktikan:

```python
df = pd.DataFrame({'u': [1,1,1,2,2], 'nilai': [10,20,30,40,50]})
print(df.groupby('u').nilai.agg('size'))       # bentuknya seperti apa?
print(df.groupby('u').nilai.transform('size')) # bentuknya seperti apa?
```

### ✅ Ujian Tingkat 2

Jelaskan dengan kalimatmu sendiri: apa isi `indptr` pada CSR, dan kenapa `latih_ials` memakainya alih-alih menulis `C[u].nonzero()`.

---

## Tingkat 3 — Content-Based Filtering

**Tujuan:** paham bagaimana teks (genre dan tag) berubah menjadi angka, dan bagaimana "mirip" diukur secara matematis.

### Konsep 3.1 — Vektor dan perkalian titik

Sebelum TF-IDF, pastikan tiga hal ini kokoh.

**Vektor** hanyalah daftar angka: `v = [3, 0, 4]`.

**Perkalian titik (*dot product*)** — kalikan pasangan yang sejajar lalu jumlahkan:

```
a = [1, 2, 3]
b = [4, 5, 6]
a · b = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

**Norma (panjang vektor)** — teorema Pythagoras yang diperluas:

```
‖v‖ = √(3² + 0² + 4²) = √25 = 5
```

Kerjakan tiga hitungan di atas dengan tangan sekarang. Sungguh — dengan pensil. Ini pondasi seluruh Tingkat 3, 4, dan 5.

### Konsep 3.2 — Cosine similarity

$$\text{sim}(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\lVert \mathbf{v}_i \rVert \, \lVert \mathbf{v}_j \rVert}$$

Yang diukur adalah **sudut** antara dua vektor, bukan jaraknya. Nilainya 1 kalau arahnya sama persis, 0 kalau tegak lurus (tidak berbagi apa pun).

Kenapa sudut, bukan jarak? Karena panjang vektor mencerminkan *seberapa banyak token* yang dimiliki sebuah film, bukan *seperti apa* film itu. Film dengan 30 tag tidak seharusnya dianggap berbeda jauh dari film dengan 3 tag hanya gara-gara jumlah tagnya.

**Trik yang dipakai proyek ini:** kalau setiap vektor sudah dinormalisasi lebih dulu sehingga panjangnya persis 1, maka penyebutnya menjadi 1 × 1 = 1, dan

```
cosine similarity = perkalian titik biasa
```

Itulah guna baris `normalize(...)` di Tahap 5 persiapan data. Berkat itu, kemiripan **seluruh pasangan film** bisa dihitung dengan satu perkalian matriks: `tfidf @ tfidf.T`.

### Konsep 3.3 — TF-IDF

Setiap film diubah menjadi satu "dokumen" berisi token:

```
Toy Story → "adventure animation children comedy fantasy
             adventure animation children comedy fantasy pixar pixar fun"
```

(Genre sengaja ditulis dua kali — alasannya di Latihan 3.3.)

**TF (*term frequency*)** — berapa kali token muncul di dokumen ini. `pixar` muncul 2 kali karena dua pengguna memberi tag itu.

**IDF (*inverse document frequency*)** — seberapa langka token itu di seluruh katalog:

$$\text{idf}(t) = \log\frac{N}{\text{df}(t)} + 1$$

dengan `N` jumlah film dan `df(t)` jumlah film yang memuat token `t`.

Intuisinya: token `drama` ada di 4.361 film, jadi mengetahui sebuah film bergenre drama **hampir tidak memberi tahu apa-apa**. Sementara `pixar` hanya ada di segelintir film, jadi token itu sangat informatif. IDF secara otomatis menekan yang pertama dan mengangkat yang kedua.

Buktinya ada di keluaran notebook — bobot TF-IDF tertinggi untuk *Toy Story*:

| Token | Bobot |
|---|---|
| pixar | 0,714 |
| fun | 0,349 |
| animation | 0,314 |
| children | 0,307 |
| fantasy | 0,293 |
| adventure | 0,253 |

`pixar` menang telak atas `animation`, padahal `animation` muncul dua kali (digandakan) dan `pixar` juga dua kali. Yang membedakan murni faktor IDF.

### Latihan 3.1 — Hitung dengan tangan, lalu buktikan dengan kode

Ambil tiga film khayalan dengan dua token saja (`action`, `comedy`):

```
Film A: action action     → vektor [2, 0]
Film B: action comedy     → vektor [1, 1]
Film C: comedy comedy     → vektor [0, 2]
```

Hitung dengan pensil: `sim(A,B)`, `sim(A,C)`, `sim(B,C)`. (Abaikan IDF dulu.)

<details>
<summary>Kunci jawaban</summary>

- `sim(A,B) = (2×1 + 0×1) / (2 × √2) = 2/2,828 = 0,707`
- `sim(A,C) = (2×0 + 0×2) / (2 × 2) = 0` — tidak berbagi token sama sekali
- `sim(B,C) = (1×0 + 1×2) / (√2 × 2) = 2/2,828 = 0,707`

Perhatikan `sim(A,B) = sim(B,C)` — masuk akal, karena B berada persis di tengah antara A dan C.
</details>

Sekarang buktikan dengan kode:

```python
import numpy as np
A, B, C = np.array([2,0]), np.array([1,1]), np.array([0,2])
def cos(x, y): return x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
print(cos(A,B), cos(A,C), cos(B,C))
```

### Latihan 3.2 — Reproduksi temuan *The Godfather*

Jalankan `rekomendasi_serupa('Godfather, The')` di notebook. Tiga film teratas punya kemiripan **1,000** — bukan 0,99, tapi persis 1.

**Pertanyaan:** kenapa bisa persis 1? Apa artinya bagi kualitas rekomendasi?

<details>
<summary>Petunjuk jawaban</summary>

Kemiripan 1,000 berarti kedua vektor menunjuk arah yang **persis sama** — kedua film punya himpunan token yang identik (`crime drama crime drama`, tanpa tag pembeda). Sistem sama sekali tidak punya dasar untuk menyatakan mana yang lebih layak direkomendasikan; urutannya akhirnya ditentukan pemecah seri (popularitas).

Ini bukan bug, melainkan **batas atas informasi** yang tersedia. Dengan 19 genre untuk 9.742 film, tabrakan seperti ini tak terhindarkan. Inilah alasan pokok kenapa CBF kalah telak pada metrik ketepatan.
</details>

### Latihan 3.3 — Eksperimen

Di `parts/p3.py`, cari baris pembentuk `content_soup`:

```python
movies['content_soup'] = ((movies.token_genre + ' ') * 2 + movies.token_tag).str.strip()
```

Ubah pengalinya dari `2` menjadi `1`, lalu `5`. Jalankan ulang, amati tabel "Perbandingan varian CBF pada data validasi".

**Pertanyaan:** ke arah mana metriknya bergerak, dan kenapa? Apa yang terjadi pada film yang **tidak punya tag sama sekali** ketika pengalinya dinaikkan? (Kembalikan ke `2` setelah selesai.)

### ✅ Ujian Tingkat 3

1. Kenapa cosine similarity memakai sudut, bukan jarak Euclidean?
2. Kenapa normalisasi L2 membuat cosine similarity setara dengan perkalian titik?
3. Kenapa IDF memberi `pixar` bobot lebih besar daripada `animation`?
4. Kenapa CBF di proyek ini punya Coverage@10 tertinggi (40,9%) tetapi Precision@10 terendah (0,0104)? Kedua fakta itu berhubungan — jelaskan hubungannya.

---

## Tingkat 4 — Matrix Factorization dan Gradient Descent

**Ini jantung proyek.** Kalau hanya ada satu tingkat yang boleh kamu kuasai betul, pilih yang ini. Sediakan waktu 6–8 jam dan jangan buru-buru.

### Konsep 4.1 — Ide besarnya

Kita punya matriks rating $R$ berukuran 610 × 8.246 yang 98% kosong. Klaim faktorisasi matriks:

> Matriks raksasa yang renggang itu bisa **dihampiri** oleh perkalian dua matriks kecil yang padat.

$$R \approx P \times Q^{\top}$$

dengan $P$ berukuran 610 × k dan $Q$ berukuran 8.246 × k. Pada proyek ini $k = 100$.

Kenapa masuk akal? Karena selera manusia tidak acak. Orang yang suka *The Godfather* cenderung suka *Goodfellas* — ada **struktur tersembunyi** yang jauh lebih sederhana daripada 5 juta sel matriks. Faktorisasi mencoba menangkap struktur itu dalam k angka per pengguna dan k angka per film.

**Yang sering disalahpahami:** tidak ada seorang pun yang menentukan arti tiap dimensi. Tidak ada yang menulis "dimensi 3 = kadar aksi". Angka-angka itu **muncul sendiri** dari proses optimasi. Kadang bisa ditafsirkan sesudahnya, sering kali tidak — dan itulah sebabnya *collaborative filtering* sulit dijelaskan ke pengguna.

### Konsep 4.2 — Kenapa perlu suku bias

Rumus mentah $\hat{r}_{ui} = \mathbf{p}_u^{\top}\mathbf{q}_i$ punya kelemahan besar. Dua kenyataan yang tidak ada hubungannya dengan kecocokan selera:

1. Sebagian orang **memang** pemurah dalam menilai. Di dataset ini, rata-rata rating per pengguna terentang dari 1,3 sampai 5,0.
2. Sebagian film **memang** lebih bagus menurut hampir semua orang.

Kalau kedua hal ini dipaksa dipelajari oleh vektor laten, kapasitas model habis untuk urusan yang sepele. Solusinya menambahkan tiga suku:

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^{\top} \mathbf{q}_i$$

| Suku | Arti | Contoh |
|---|---|---|
| $\mu$ | rata-rata rating global | 3,5141 |
| $b_u$ | pengguna ini biasanya menilai berapa di atas/bawah $\mu$ | −2,088 sampai +1,276 |
| $b_i$ | film ini biasanya dinilai berapa di atas/bawah $\mu$ | −1,703 sampai +0,931 |
| $\mathbf{p}_u^{\top}\mathbf{q}_i$ | kecocokan selera personal | sisanya |

**Dan inilah kejutannya.** Dari hasil evaluasi proyek ini:

| Model | RMSE | Perbaikan atas tebakan $\mu$ |
|---|---|---|
| Rata-rata global ($\mu$ saja) | 1,0688 | 0% |
| **Bias saja ($\mu + b_u + b_i$)** | **0,8937** | **16,38%** |
| MF penuh (100 faktor laten) | 0,8796 | 17,70% |

Suku bias — yang tidak mengandung setitik pun personalisasi — menyumbang **16,4 dari 17,7 poin**. Seratus faktor laten hanya menambah 1,3 poin sisanya.

Renungkan ini. Sebagian besar "keberhasilan" model prediksi rating sebenarnya berasal dari menjawab dua pertanyaan sederhana: *siapa yang menilai* dan *film apa yang dinilai*. Selera personal adalah lapisan yang jauh lebih tipis daripada dugaan orang. Temuan ini bukan hal baru — inilah salah satu pelajaran utama kompetisi *Netflix Prize*.

### Konsep 4.3 — Fungsi objektif

Apa yang sebenarnya sedang dicari komputer? Ini:

$$\min_{p,q,b} \sum_{(u,i) \in \mathcal{K}} \left( r_{ui} - \hat{r}_{ui} \right)^2 + \lambda \left( \lVert \mathbf{p}_u \rVert^2 + \lVert \mathbf{q}_i \rVert^2 + b_u^2 + b_i^2 \right)$$

Bacalah pelan-pelan, sepotong demi sepotong:

- $\sum_{(u,i) \in \mathcal{K}}$ — **hanya pada rating yang benar-benar teramati.** Ingat baik-baik bagian ini. Nanti di Tingkat 6 kamu akan lihat bahwa dari sinilah seluruh kelemahan MF berasal.
- $(r_{ui} - \hat{r}_{ui})^2$ — selisih tebakan dengan kenyataan, dikuadratkan supaya meleset jauh dihukum jauh lebih berat.
- $\lambda(\dots)$ — **regularisasi**: denda untuk angka yang membesar tak terkendali.

### Konsep 4.4 — Kenapa regularisasi diperlukan

Tanpa denda $\lambda$, model bisa membuat $b_i$ untuk film berdata satu rating menjadi sangat besar demi mencocokkan satu titik data itu dengan sempurna. Itu **menghafal**, bukan **belajar**.

Dengan denda, model harus membayar untuk setiap angka besar, sehingga film berdata sedikit ditarik mendekati rata-rata global. Statistikawan menyebut ini ***shrinkage***.

**Tapi lihat apa yang terjadi di proyek ini.** Lima film dengan $b_i$ tertinggi setelah pelatihan:

| Film | Jumlah rating latih | $b_i$ |
|---|---|---|
| Yojimbo | 11 | 0,931 |
| Paths of Glory | 8 | 0,905 |
| Guess Who's Coming to Dinner | 9 | 0,880 |
| His Girl Friday | 12 | 0,878 |
| Lawrence of Arabia | 39 | 0,871 |

Semuanya film berdata sedikit. Nilai $\lambda = 0{,}05$ dipilih karena memberi **RMSE validasi terbaik**, tetapi ternyata terlalu lemah untuk menarik film-film ini kembali ke tengah.

Simpan fakta ini di kepala. Di Tingkat 6 kita akan lihat akibat fatalnya.

### Konsep 4.5 — Gradient descent, dari nol

Bagaimana komputer meminimumkan fungsi objektif itu? Dengan **menuruni bukit**.

Bayangkan kamu berdiri di lereng bukit dalam kabut tebal dan ingin turun. Kamu tidak bisa melihat lembah, tapi bisa merasakan kemiringan tanah di bawah kaki. Strateginya: melangkah kecil ke arah paling curam menurun, lalu ulangi.

- **Kemiringan** = gradien (turunan fungsi objektif terhadap parameter)
- **Ukuran langkah** = *learning rate* $\eta$ (di sini 0,01)

*Stochastic* gradient descent berarti: alih-alih menghitung kemiringan dari seluruh 80.896 rating sekaligus, kita **melangkah setelah setiap satu rating**. Lebih berisik, tapi jauh lebih cepat.

### Konsep 4.6 — Menurunkan aturan pembaruan sendiri

Ini bagian yang membedakan orang yang paham dari orang yang cuma menyalin. **Kerjakan dengan pensil.**

Ambil satu rating. Galatnya:

$$e_{ui} = r_{ui} - \hat{r}_{ui} = r_{ui} - (\mu + b_u + b_i + \mathbf{p}_u^{\top}\mathbf{q}_i)$$

Fungsi objektif untuk satu titik ini:

$$L = e_{ui}^2 + \lambda\left(\lVert\mathbf{p}_u\rVert^2 + \lVert\mathbf{q}_i\rVert^2 + b_u^2 + b_i^2\right)$$

Turunkan terhadap $\mathbf{p}_u$. Pakai aturan rantai:

$$\frac{\partial L}{\partial \mathbf{p}_u} = 2 e_{ui} \cdot \frac{\partial e_{ui}}{\partial \mathbf{p}_u} + 2\lambda \mathbf{p}_u = 2 e_{ui} \cdot (-\mathbf{q}_i) + 2\lambda \mathbf{p}_u = -2\left(e_{ui}\mathbf{q}_i - \lambda \mathbf{p}_u\right)$$

Gradient descent melangkah **berlawanan** arah gradien, dan faktor 2 diserap ke dalam $\eta$:

$$\boxed{\mathbf{p}_u \leftarrow \mathbf{p}_u + \eta\left(e_{ui}\mathbf{q}_i - \lambda \mathbf{p}_u\right)}$$

Sekarang cocokkan dengan kodenya di `latih_mf()`:

```python
galat = nilai - (mu + bu[u] + bi[i] + p_u @ q_i)
bu[u] += lr * (galat - reg * bu[u])
bi[i] += lr * (galat - reg * bi[i])
P[u]  += lr * (galat * q_i - reg * p_u)
Q[i]  += lr * (galat * p_u - reg * q_i)
```

Persis sama. **Tidak ada sihir di dalamnya** — hanya turunan yang diterjemahkan ke Python.

### Latihan 4.1 — Turunkan sisanya sendiri

Dengan pensil, turunkan aturan pembaruan untuk $\mathbf{q}_i$ dan $b_u$. Cocokkan dengan kode.

Petunjuk untuk $b_u$: turunan $e_{ui}$ terhadap $b_u$ adalah $-1$, jadi hasilnya lebih sederhana daripada versi vektor.

### Latihan 4.2 — Tulis MF dari nol

Jangan menyalin dari proyek. Buka editor kosong dan tulis sendiri untuk matriks 4 × 4:

```python
import numpy as np

R = np.array([[5, 3, 0, 1],
              [4, 0, 0, 1],
              [1, 1, 0, 5],
              [0, 0, 5, 4]], dtype=float)   # 0 = belum dirating

# TUGASMU:
# 1. Buat P (4×2) dan Q (4×2) berisi angka acak kecil
# 2. Untuk 500 epoch, untuk setiap sel yang TIDAK nol:
#    - hitung galat
#    - perbarui P[u] dan Q[i] dengan lr=0.01, reg=0.02
# 3. Cetak P @ Q.T dan bandingkan dengan R
```

Kalau berhasil, sel yang tidak nol akan mendekati nilai aslinya, dan sel yang nol akan terisi tebakan yang masuk akal. **Momen ketika kamu melihat sel kosong terisi angka wajar adalah momen kamu benar-benar paham faktorisasi matriks.**

Kalau macet, baru intip `latih_mf()` — tapi usahakan menulis sendiri dulu setidaknya 30 menit.

### Latihan 4.3 — Buktikan bias itu penting

Di `latih_mf()`, matikan suku bias dengan mengomentari dua baris pembaruannya, lalu jalankan ulang. Amati perubahan RMSE.

**Ramalkan dulu arahnya sebelum menjalankan**, lalu periksa. Meramal sebelum melihat hasil adalah cara tercepat menemukan lubang di pemahamanmu.

### Latihan 4.4 — Rasakan overfitting

Buka `parts/p4.py`, ubah `grid_mf` menjadi satu konfigurasi dengan regularisasi sangat kecil:

```python
grid_mf = [{'n_faktor': 100, 'reg': 0.001, 'lr': 0.01}]
```

Amati kurva pembelajaran. RMSE latih akan terjun bebas sementara RMSE validasi mandek atau memburuk. **Itulah wajah overfitting.** Kembalikan seperti semula setelah selesai.

### ✅ Ujian Tingkat 4

1. Kenapa matriks 610 × 8.246 bisa dihampiri dua matriks 610 × 100 dan 8.246 × 100?
2. Apa arti $b_u$ dan $b_i$, dan berapa besar sumbangan keduanya terhadap RMSE?
3. Turunkan aturan pembaruan $\mathbf{p}_u$ di papan tulis, tanpa contekan.
4. Apa fungsi $\lambda$, dan apa akibatnya kalau terlalu kecil?
5. Kenapa disebut *stochastic* gradient descent, bukan gradient descent biasa?

---

## Tingkat 5 — Implicit ALS

**Tujuan:** paham kenapa model kedua ini ada, dan kenapa justru ia yang menang.

### Konsep 5.1 — Persoalan yang berbeda

MF menjawab: *"kalau pengguna ini menonton film itu, dia akan memberi rating berapa?"*

Padahal yang dibutuhkan sistem rekomendasi: *"sepuluh film mana yang harus saya tampilkan?"*

**Kedua pertanyaan itu tidak sama.** Ini titik terpenting seluruh proyek.

Kenapa berbeda? Karena fungsi objektif MF hanya menjumlahkan pada $(u,i) \in \mathcal{K}$ — rating yang **teramati**. Kalau model menaruh film asing dengan prediksi 5,3 di puncak daftar, tidak ada denda apa pun, sebab film itu tidak ada di data latih. Model tidak pernah diberi tahu bahwa tebakannya konyol.

Lihat sendiri buktinya. Top-10 dari MF untuk pengguna penggemar film laga (`userId = 452`, yang memberi rating 5,0 untuk *The Matrix*, *Braveheart*, *The Godfather*):

| # | Judul | Tahun | Prediksi | Tepat? |
|---|---|---|---|---|
| 1 | Sunset Blvd. | 1950 | 5,29 | – |
| 2 | High Noon | 1952 | 5,29 | – |
| 3 | Guess Who's Coming to Dinner | 1967 | 5,26 | – |
| … | (semuanya klasik era 1940–60-an) | | | – |

Nol dari sepuluh. Semuanya film klasik berdata sedikit yang $b_i$-nya melambung — persis film-film yang kamu lihat di tabel bias pada Tingkat 4.

### Konsep 5.2 — Umpan balik implisit

iALS menyerang dari sudut yang sama sekali berbeda. Rating diterjemahkan jadi sinyal biner:

$$p_{ui} = \begin{cases} 1 & \text{kalau rating} \geq 4 \text{ (suka)} \\ 0 & \text{untuk semua pasangan lain} \end{cases}$$

Perhatikan "**semua pasangan lain**" — termasuk film yang belum pernah dilihat pengguna. Ini perbedaan mendasar dari MF.

Tapi ada masalah jelas: pengguna tidak menonton sebuah film bisa berarti dua hal berbeda — tidak suka, atau belum tahu film itu ada. Kita tidak bisa membedakannya. Solusi Hu, Koren & Volinsky: beri **bobot keyakinan**.

$$c_{ui} = 1 + \alpha \, p_{ui}$$

- Pasangan teramati (suka): bobot $1 + \alpha = 11$ → "saya sangat yakin"
- Pasangan tidak teramati: bobot 1 → "saya menduga tidak suka, tapi tidak yakin"

Yang tak teramati **tetap ikut dihitung**, hanya dengan suara yang lebih pelan. Inilah kuncinya: model belajar bahwa film populer yang dilewatkan pengguna kemungkinan besar memang tidak diminati. MF tidak pernah punya kesempatan mempelajari hal itu.

Fungsi objektifnya mencakup **seluruh** 610 × 8.246 ≈ 5 juta pasangan:

$$\min_{x,y} \sum_{u,i} c_{ui}\left(p_{ui} - \mathbf{x}_u^{\top}\mathbf{y}_i\right)^2 + \lambda\left(\sum_u \lVert \mathbf{x}_u \rVert^2 + \sum_i \lVert \mathbf{y}_i \rVert^2\right)$$

Bandingkan dengan fungsi objektif MF di Tingkat 4. Perbedaannya cuma satu: batas penjumlahan. Tapi akibatnya besar sekali.

### Konsep 5.3 — Kenapa ALS, bukan SGD

Lima juta pasangan pada setiap langkah SGD jelas tidak praktis. Tapi ada jalan keluar yang elegan.

**Kunci:** kalau $Y$ (faktor film) dianggap **tetap**, fungsi objektif terhadap $X$ (faktor pengguna) menjadi *kuadratik* — dan persoalan kuadratik punya **solusi tertutup**. Tidak perlu menuruni bukit selangkah demi selangkah; langsung lompat ke dasar lembah dengan menyelesaikan sistem persamaan linear:

$$\mathbf{x}_u = \left(Y^{\top}Y + Y^{\top}(C^u - I)Y + \lambda I\right)^{-1} Y^{\top} C^u \mathbf{p}(u)$$

Lalu perannya ditukar: $X$ dibekukan, $Y$ diperbarui. Bolak-balik. Itulah arti ***alternating***.

### Konsep 5.4 — Trik yang membuatnya cepat

Perhatikan struktur rumusnya:

- $Y^{\top}Y$ — **tidak bergantung pada pengguna mana pun**, jadi dihitung **sekali saja** untuk seluruh 610 pengguna.
- $Y^{\top}(C^u - I)Y$ — karena $c_{ui} - 1 = 0$ untuk pasangan tak teramati, suku ini **hanya melibatkan film yang benar-benar disukai pengguna u**. Untuk pengguna biasa itu cuma puluhan film, bukan 8.246.

Lihat bagaimana dua fakta itu menjelma menjadi kode:

```python
YtY = Y.T @ Y + I_reg                       # sekali untuk semua pengguna
for u in range(N_PENGGUNA):
    item = C_pengguna.indices[C_pengguna.indptr[u]:C_pengguna.indptr[u+1]]  # film yang disukai u
    Yi = Y[item]                            # hanya sebagian kecil dari Y
    A = YtY + alpha * (Yi.T @ Yi)           # bagian mahal jadi murah
    b = (1 + alpha) * Yi.sum(axis=0)
    X[u] = np.linalg.solve(A, b)
```

Berkat itu, satu iterasi ALS di data ini hanya butuh sepersekian detik — padahal fungsi objektifnya mencakup lima juta pasangan.

### Latihan 5.1 — Buktikan solusi tertutupnya

Untuk pembaca yang nyaman dengan kalkulus vektor: turunkan fungsi objektif terhadap $\mathbf{x}_u$, samakan dengan nol, lalu selesaikan. Kamu akan sampai persis pada rumus di atas.

Kalau kalkulus vektor terasa berat, cukup pahami idenya: **turunan sama dengan nol = titik terendah**. Untuk fungsi kuadratik, titik itu bisa dicari langsung dengan aljabar, tidak perlu iterasi.

### Latihan 5.2 — Mainkan alpha

Di `parts/p5.py`, ubah `grid_ials` menjadi satu konfigurasi dan coba `alpha` = 1, 10, 100. Amati NDCG@10 validasi.

**Pertanyaan:** apa arti $\alpha = 1$? Apa yang terjadi pada model ketika $\alpha$ sangat besar? (Petunjuk: lihat kembali rumus $c_{ui}$ dan pikirkan seberapa "pelan" suara pasangan tak teramati menjadi.)

### Latihan 5.3 — Kenapa k lebih kecil?

Konfigurasi terpilih: MF pakai $k = 100$, iALS hanya $k = 16$. Dari tabel penyetelan, $k = 64$ untuk iALS justru **lebih buruk** (NDCG 0,1031 vs 0,1231).

**Pertanyaan:** kenapa tugas menyusun peringkat butuh representasi yang lebih ringkas daripada tugas menebak angka rating? Apa hubungannya dengan hanya adanya 610 pengguna?

### ✅ Ujian Tingkat 5

1. Apa perbedaan umpan balik eksplisit dan implisit?
2. Apa arti $c_{ui} = 1 + \alpha p_{ui}$, dan kenapa pasangan tak teramati tetap diberi bobot 1 alih-alih dibuang?
3. Kenapa ALS bisa memakai solusi tertutup sementara MF harus memakai SGD?
4. Jelaskan trik $Y^{\top}Y$ dan kenapa ia membuat ALS praktis.

---

## Tingkat 6 — Evaluasi

**Tujuan:** memahami bagian yang paling sering disepelekan orang, padahal justru di sinilah temuan terpenting proyek ini berada.

### Konsep 6.1 — RMSE dan MAE

$$\text{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|} \sum (r_{ui} - \hat{r}_{ui})^2} \qquad \text{MAE} = \frac{1}{|\mathcal{T}|} \sum |r_{ui} - \hat{r}_{ui}|$$

Bedanya cuma satu: RMSE mengkuadratkan sebelum merata-rata. Akibatnya meleset 2 bintang dihitung **empat kali** lebih buruk daripada meleset 1 bintang, sedangkan MAE menganggapnya dua kali lebih buruk.

Kalau RMSE jauh lebih besar dari MAE, artinya model sesekali meleset sangat jauh meski rata-ratanya bagus.

### Konsep 6.2 — Metrik peringkat, dihitung dengan tangan

Andaikan sistem merekomendasikan 5 film, dan tanda ✓ berarti pengguna benar-benar menyukainya. Pengguna itu total menyukai 8 film pada data uji.

```
Posisi:  1    2    3    4    5
Hasil:   ✗    ✓    ✗    ✓    ✗
```

**Precision@5** — berapa bagian dari yang disodorkan ternyata benar:

$$\frac{2}{5} = 0{,}400$$

**Recall@5** — berapa bagian dari yang disukai berhasil tertangkap:

$$\frac{2}{8} = 0{,}250$$

**NDCG@5** — seperti precision, tapi **peduli posisi**. Hit di posisi atas lebih berharga.

Hitung DCG (hit dibagi $\log_2(\text{posisi}+1)$):

$$\text{DCG} = \frac{0}{\log_2 2} + \frac{1}{\log_2 3} + \frac{0}{\log_2 4} + \frac{1}{\log_2 5} + \frac{0}{\log_2 6} = 0 + 0{,}631 + 0 + 0{,}431 + 0 = 1{,}062$$

IDCG = nilai terbaik yang **mungkin dicapai**, yaitu kalau slot teratas terisi penuh oleh film relevan. Karena pengguna ini menyukai 8 film sementara slotnya cuma 5, kelima slot bisa terisi semua — jadi IDCG menjumlahkan **lima** suku, bukan dua:

$$\text{IDCG} = \frac{1}{\log_2 2} + \frac{1}{\log_2 3} + \frac{1}{\log_2 4} + \frac{1}{\log_2 5} + \frac{1}{\log_2 6} = 1{,}000 + 0{,}631 + 0{,}500 + 0{,}431 + 0{,}387 = 2{,}948$$

$$\text{NDCG@5} = \frac{1{,}062}{2{,}948} = 0{,}360$$

> **Perhatikan baik-baik:** banyaknya suku IDCG adalah $\min(|\mathcal{R}_u|, K)$ — bukan jumlah hit yang benar-benar terjadi. Ini kekeliruan yang paling sering terjadi saat orang menghitung NDCG sendiri. Kalau IDCG dihitung dari jumlah hit, NDCG akan selalu terlihat tinggi dan kehilangan maknanya. Cocokkan dengan kodenya di `evaluasi_ranking()`:
>
> ```python
> idcg = idcg_kumulatif[min(len(item_relevan), K) - 1]
> ```

**Kerjakan hitungan ini dengan kalkulator sekarang.** Setelah sekali menghitung NDCG dengan tangan, kamu tidak akan pernah lagi bingung membacanya.

**Coverage@K** bukan metrik ketepatan melainkan **keberagaman**: berapa persen katalog yang pernah muncul di rekomendasi siapa pun. Baseline populer hanya 3,2% — ia menyodorkan sepuluh judul yang sama kepada semua orang, dan 96,8% katalog tak pernah punya kesempatan.

### Konsep 6.3 — Temuan terpenting proyek ini

Sekarang semua kepingan bertemu. Lihat tabel ini:

| Model | RMSE (makin kecil makin baik) | NDCG@10 (makin besar makin baik) |
|---|---|---|
| Popularitas | – | 0,0751 |
| **Matrix Factorization** | **0,8796 — terbaik** | **0,0278 — kalah dari popularitas** |
| Implicit ALS | tidak bisa menghitung rating | **0,1073 — terbaik** |

**Model dengan RMSE terbaik justru menghasilkan rekomendasi terburuk.** Bahkan kalah dari daftar film populer yang tidak dipersonalisasi sama sekali.

Rantai sebabnya sudah kamu telusuri sepanjang panduan ini:

1. Fungsi objektif MF hanya menjumlahkan pada rating **teramati** (Tingkat 4.3)
2. Karena itu $\lambda$ kecil terasa optimal bagi RMSE — tidak ada yang menghukumnya (Tingkat 4.4)
3. $\lambda$ kecil membuat film berdata sedikit memperoleh $b_i$ melambung (tabel Yojimbo dkk.)
4. Film-film itulah yang membanjiri top-10 (daftar *Sunset Blvd.* dkk. di Tingkat 5.1)
5. Precision@10 pun ambruk

Ini bukan kegagalan implementasi. Ini **reproduksi temuan Cremonesi, Koren & Turrin (2010)**, salah satu makalah paling banyak dikutip di bidang sistem rekomendasi.

Pelajaran yang bisa dibawa ke pekerjaan mana pun: **metrik yang kamu optimalkan harus metrik yang benar-benar kamu pedulikan.** Kalau tidak, model akan dengan patuh menjadi hebat pada hal yang salah.

### Konsep 6.4 — Kebocoran data, dibuktikan dengan angka

Proyek ini membagi data secara **temporal per pengguna**: 80% riwayat awal untuk latih, 20% terakhir untuk uji. Bukan acak.

Kenapa repot-repot? Karena pembagian acak membuat model belajar dari rating tahun 2018 untuk menebak rating tahun 2000. Sistem sungguhan tidak pernah punya kemewahan itu — ia selalu menebak ke depan.

Seberapa besar bedanya? Saya jalankan **model iALS yang persis sama**, hanya cara membagi datanya yang diubah:

| Cara membagi | Precision@10 | NDCG@10 |
|---|---|---|
| Temporal (dipakai proyek) | 0,0828 | 0,1073 |
| Acak (bocor waktu) | **0,1883** | **0,2633** |

**Pembagian acak melambungkan Precision@10 sebesar 128% dan NDCG@10 sebesar 145% — tanpa satu pun perbaikan pada modelnya.**

Angka yang sama persis, model yang sama persis, kode yang sama persis. Yang berubah hanya kejujuran protokolnya.

Kalau suatu hari kamu melihat hasil sistem rekomendasi yang tampak terlalu bagus, pertanyaan pertama yang harus kamu ajukan adalah: **bagaimana datanya dibagi?**

### Latihan 6.1 — Reproduksi sendiri

Tulis ulang eksperimen di atas. Petunjuk: ganti `bagi_temporal()` dengan versi yang mengacak baris lebih dulu (`df.sample(frac=1, random_state=42)`) sebelum mengambil 80% pertama tiap pengguna. Jangan lupa mengembalikannya setelah selesai.

### Latihan 6.2 — Hitung metrik dengan tangan

Ambil daftar top-10 iALS untuk `userId = 452` dari notebook. Ada empat penanda `YA` di posisi 2, 4, 5, dan 6. Pengguna itu punya 39 film relevan pada data uji.

Hitung dengan kalkulator: Precision@10, Recall@10, dan NDCG@10 untuk pengguna ini saja. Lalu buktikan dengan memanggil `evaluasi_ranking()` untuk satu pengguna itu.

### Latihan 6.3 — Geser ambang "suka"

Di `parts/p3.py`, ubah `AMBANG_SUKA` dari `4.0` menjadi `3.5`, lalu `4.5`. Jalankan ulang.

**Ramalkan dulu**: ke arah mana Precision@10 bergerak, dan kenapa? (Petunjuk: dengan ambang lebih longgar, lebih banyak film dianggap "relevan" — apa artinya bagi peluang menebak benar secara kebetulan?) Ini melatih kepekaan bahwa **angka metrik bergantung pada definisi yang kita pilih sendiri**, bukan kebenaran mutlak.

### ✅ Ujian Tingkat 6

1. Hitung NDCG@5 untuk pola hit `✓ ✗ ✗ ✓ ✗` dengan 6 item relevan, tanpa contekan.
2. Kenapa MF menang RMSE tapi kalah NDCG? Jelaskan rantai sebabnya dari fungsi objektif.
3. Kenapa pembagian data acak melambungkan skor, dan seberapa besar di proyek ini?
4. Kenapa Coverage@10 disertakan padahal ia tidak mengukur ketepatan sama sekali?

---

## Tingkat 7 — Menyatukan Semuanya

### Ujian akhir

Jawab dengan suara keras, seolah sedang diwawancarai. Kalau ada yang tersendat, kembali ke tingkat terkait.

**Dasar**
1. Jelaskan proyek ini dalam dua menit kepada orang yang tidak tahu apa-apa soal *machine learning*.
2. Kenapa dataset ini sulit? Sebut tiga sifatnya beserta angkanya.

**Content-Based**
3. Bagaimana genre dan tag berubah menjadi angka?
4. Kenapa IDF diperlukan? Beri contoh dari dataset ini.
5. Kenapa CBF punya cakupan tertinggi tapi ketepatan terendah?

**Collaborative**
6. Tulis rumus prediksi MF di papan tulis dan jelaskan tiap sukunya.
7. Turunkan aturan pembaruan $\mathbf{p}_u$.
8. Apa bedanya iALS dengan MF? Kenapa yang satu memakai SGD dan satunya ALS?
9. Kenapa iALS menang, sementara MF bahkan kalah dari baseline populer?

**Evaluasi**
10. Kenapa RMSE bukan metrik yang tepat untuk rekomendasi top-N?
11. Kenapa pembagian temporal, bukan acak? Seberapa besar bedanya?
12. Kalau atasanmu minta menaikkan Precision@10, apa dua hal pertama yang kamu coba?

### Kalau semua bisa dijawab

Proyek ini benar-benar sudah jadi milikmu. Langkah berikutnya:

1. **Bangun ulang dari nol** di folder kosong, tanpa melihat kode yang ada. Boleh melihat laporan untuk mengingat urutan tahapan. Ini ujian paling jujur.
2. **Bikin hybrid recommender** — CBF untuk pengguna dan film baru, beralih ke iALS setelah interaksi cukup. Kedua kelemahannya saling menambal; datanya sudah menunjukkan itu.
3. **Bikin prototipe Streamlit** — input `userId`, keluar top-10. Repo ini jadi punya demo yang bisa diklik.
4. **Coba MovieLens 25M** — 250 kali lebih besar. Kode NumPy ini akan kewalahan, dan justru di situ kamu belajar kenapa pustaka seperti `implicit` ada.

---

## Lampiran A — Peta Konsep ke Kode

| Konsep | Di notebook | Fungsi kunci |
|---|---|---|
| Long tail, sparsity | Bab 3.3 | — |
| Normalisasi token | Tahap 3–4 | `normalkan_token()` |
| TF-IDF | Tahap 5 | `TfidfVectorizer` |
| Matriks renggang | Tahap 8 | `sp.csr_matrix` |
| Split temporal | Tahap 7 | `bagi_temporal()` |
| Cosine similarity | Bab 5.1 | `rekomendasi_serupa()` |
| Profil pengguna CBF | Bab 5.1 | `bangun_skor_cbf()` |
| Matrix factorization | Bab 5.2 | `latih_mf()` |
| Implicit ALS | Bab 5.3 | `latih_ials()` |
| Metrik peringkat | Bab 5.0 | `evaluasi_ranking()` |
| RMSE / MAE | Bab 5.0 | `evaluasi_rating()` |

---

## Lampiran B — Glosarium

| Istilah | Arti singkat |
|---|---|
| **Cold start** | Tidak ada data untuk pengguna atau item baru |
| **Cosine similarity** | Kemiripan berdasarkan sudut antarvektor, bukan jarak |
| **Coverage** | Berapa persen katalog yang pernah direkomendasikan |
| **Faktor laten** | Dimensi tersembunyi yang dipelajari model, tanpa arti yang ditentukan manusia |
| **IDF** | Bobot yang menekan token umum dan mengangkat token langka |
| **Long tail** | Sedikit item sangat populer, sangat banyak item nyaris tak tersentuh |
| **NDCG** | Metrik peringkat yang memberi bobot lebih pada posisi atas |
| **Overfitting** | Model menghafal data latih dan gagal menggeneralisasi |
| **Regularisasi** | Denda untuk parameter besar, penangkal overfitting |
| **Shrinkage** | Menarik estimasi berdata sedikit mendekati rata-rata |
| **Sparsity** | Persentase sel kosong pada matriks interaksi |
| **TF-IDF** | Pembobotan token gabungan frekuensi dan kelangkaan |
| **Umpan balik implisit** | Sinyal perilaku (klik, tonton), bukan penilaian eksplisit |

---

## Lampiran C — Kesalahpahaman yang Umum

**"RMSE rendah berarti rekomendasinya bagus."**
Justru inilah yang dibantah proyek ini dengan angka. MF punya RMSE terbaik dan NDCG terburuk.

**"Faktor laten punya arti yang bisa dibaca manusia."**
Kadang bisa ditafsirkan sesudahnya, tetapi tidak ada yang menentukannya di muka. Jangan berjanji ke pengguna bahwa "dimensi 7 adalah kadar romantis".

**"Makin banyak faktor laten makin bagus."**
iALS dengan $k = 64$ lebih buruk daripada $k = 16$ di proyek ini. Dengan 610 pengguna, model besar kehilangan daya generalisasi.

**"Cosine similarity mengukur jarak."**
Ia mengukur sudut. Dua vektor bisa berjauhan tetapi punya cosine similarity 1,0 kalau arahnya sama.

**"Pembagian data acak itu netral dan aman."**
Untuk data yang punya dimensi waktu, pembagian acak membocorkan masa depan dan melambungkan skor sampai lebih dari dua kali lipat.

**"Baseline itu formalitas."**
Baseline populer di proyek ini mengalahkan dua dari tiga model sungguhan. Tanpa pembanding itu, MF akan terlihat lumayan.

---

## Lampiran D — Sumber Belajar Eksternal

**Makalah** (tiga yang menjadi pondasi proyek ini — baca setelah Tingkat 4)
- Koren, Bell & Volinsky (2009), *Matrix Factorization Techniques for Recommender Systems* — paling mudah dibaca, mulai dari sini
- Hu, Koren & Volinsky (2008), *Collaborative Filtering for Implicit Feedback Datasets* — dasar iALS
- Cremonesi, Koren & Turrin (2010), *Performance of Recommender Algorithms on Top-N Recommendation Tasks* — sumber temuan utama proyek ini

**Aljabar linear** (kalau Tingkat 4 terasa berat)
- 3Blue1Brown, *Essence of Linear Algebra* di YouTube — 15 video, visual, sangat membantu memahami vektor dan perkalian matriks

**Buku**
- Aggarwal, *Recommender Systems: The Textbook* — rujukan menyeluruh
- Falk, *Practical Recommender Systems* — lebih ringan, banyak sisi rekayasa

---

## Penutup

Panduan ini tidak akan membuatmu paham dengan cara dibaca. Ia hanya berguna kalau latihannya dikerjakan — terutama **Latihan 4.2** (menulis MF dari nol) dan **Latihan 6.2** (menghitung NDCG dengan tangan). Dua latihan itu yang paling banyak memindahkan pemahaman dari halaman ini ke kepalamu.

Kalau ada satu bagian yang macet berhari-hari, itu justru pertanda bagus: kamu sedang menyentuh batas pemahamanmu, dan di situlah belajar sesungguhnya terjadi.

Selamat mengerjakan.
