# BAB 3 PERANCANGAN SISTEM

Bab ini menjelaskan bagaimana agen pencarian dibangun, dilatih, dan dievaluasi. Subbab 3.1 memberikan gambaran umum rancangan penelitian beserta diagram alurnya (Gambar 3.1). Subbab 3.2 sampai 3.7 kemudian menjelaskan setiap blok pada diagram tersebut sesuai urutannya, yaitu agen (3.2), *dataset* (3.3), lingkungan pencarian (3.4), *reward* VEGR (3.5), algoritma pelatihan GRPO (3.6), dan skenario eksperimen (3.7). Subbab 3.8 mendefinisikan protokol evaluasi, dan Subbab 3.9 melaporkan detail implementasi dan komputasi.

## 3.1 Gambaran Umum Rancangan Penelitian

Penelitian ini merupakan eksperimen terkontrol. Model, data, lingkungan, *hyperparameter*, dan protokol evaluasi dibuat tetap. Yang berubah antar-*run* hanyalah fungsi *reward* dan, pada dua varian, prosedur optimasi. Perbandingannya juga berpasangan: *run* dengan *seed* yang sama melatih 256 pertanyaan yang sama dengan urutan yang sama, apa pun *reward*-nya (Subbab 3.6). Kontrol ini memiliki dua batasan yang perlu dinyatakan sejak awal, yaitu sebagai berikut.

1. Perbandingan utama, yaitu R1 (VEGR) dibandingkan dengan R0 (*outcome-only*), mengubah **seluruh paket *reward* sekaligus**: enam komponen ditambahkan (F1, cakupan, presisi, *retrieval recall*, biaya pencarian, dan penalti duplikasi). Rancangan ini tidak dapat menunjukkan komponen mana yang menyebabkan suatu perbedaan.
2. *Reward* yang lebih padat juga mengubah jumlah langkah pelatihan yang menghasilkan pembaruan. R0 melewati sekitar separuh langkahnya sebagai grup dengan variansi nol (Subbab 3.6), sehingga *run* R0 melakukan 128–149 pembaruan, sedangkan *run* R1 melakukan sekitar 240 pembaruan (Tabel 4.1). Sebagian pembaruan R1 berasal dari grup yang *reward*-nya hanya berbeda sangat sedikit, misalnya 0,033 akibat satu pencarian tambahan, karena normalisasi dengan simpangan baku grup mengubah selisih sekecil apa pun menjadi *advantage* berskala satuan.

Oleh karena itu, perbedaan antara R1 dan R0 diatribusikan kepada paket VEGR secara utuh, bersama dengan sinyal pelatihan yang dihasilkannya, dan bukan kepada satu komponen tertentu.

Gambar 3.1 menunjukkan satu langkah pelatihan. Agen menerima sebuah pertanyaan HotpotQA beserta 10 paragrafnya, melakukan hingga 3 aksi `SEARCH` pada paragraf-paragraf tersebut, lalu mengakhiri episode dengan aksi `ANSWER[...] CITATIONS[...]`. Episode yang selesai dinilai oleh *reward* VEGR terhadap jawaban emas dan *supporting facts* emas. GRPO kemudian membandingkan *reward* dari 8 episode pada pertanyaan yang sama dan memperbarui *adapter* LoRA milik agen. Seluruh langkah ini diulang untuk setiap skenario eksperimen (Subbab 3.7).

![Figure 3.1](../figures/fig3_1_system.png)

**Gambar 3.1.** Diagram alur satu langkah pelatihan

Rancangan ini dipandu oleh tiga pertanyaan penelitian (Bab 1), yaitu sebagai berikut.

- **RQ1.** Apakah pelatihan GRPO meningkatkan performa agen pencarian 7B pada HotpotQA dibandingkan dengan model tanpa pelatihan?
- **RQ2.** Apakah *verifiable evidence-grounded reward* (VEGR) mencegah perilaku jalan pintas dari *reward* berbasis jawaban saja, tanpa kehilangan akurasi?
- **RQ3.** Bagaimana pengaruh varian *reward* (penalti bergerbang/*gated penalties* dan bobot cakupan yang lebih besar) serta perubahan *optimizer* (*advantage* tanpa normalisasi standar deviasi dan *dynamic sampling*) terhadap *grounding* dan perilaku pencarian?

Pada RQ2, perilaku jalan pintas berarti menjawab tanpa menyertakan sitasi bukti, yang terlihat sebagai tingkat sitasi yang mendekati 0 pada *holdout*. Subbab 3.8 menetapkan metrik utama untuk setiap pertanyaan.

## 3.2 Agen Pencarian

Agen adalah sebuah LLM yang berperan sebagai *policy*: pada setiap giliran, agen membaca pertanyaan dan seluruh hasil pencarian sejauh ini, lalu mengeluarkan satu aksi. Agen ini mengikuti *loop* bergaya ReAct [1] yang disederhanakan. Agen tidak menuliskan penalaran bebas, tetapi harus memilih antara mencari atau menjawab dengan tata bahasa aksi yang ketat, sehingga setiap keluaran dapat diperiksa dengan aturan.

**Model dasar.** *Policy* yang digunakan adalah **Qwen2.5-7B-Instruct** [12], yaitu model *open-weight* yang telah melalui *instruction tuning*, dimuat dari bobot lokal dalam format bf16. Model berukuran 7B cukup kecil untuk dilatih pada satu GPU 48 GB dengan LoRA. Karena sudah melalui *instruction tuning*, model tanpa pelatihan sudah dapat mencoba mengikuti format aksi, sehingga RL memiliki titik awal.

**Tata bahasa aksi.** Pada setiap giliran, agen harus mengeluarkan tepat satu aksi, yaitu sebagai berikut.

- `SEARCH[query]` menjalankan pencarian pada lingkungan (Subbab 3.4) dan menggunakan satu unit anggaran pencarian.
- `ANSWER[answer]`, opsional diikuti `CITATIONS[title|index; title|index; ...]`, memberikan jawaban akhir beserta kalimat-kalimat bukti yang disitasi. Setiap sitasi menunjuk satu kalimat dengan judul paragrafnya dan nomor kalimatnya. Aksi ini mengakhiri episode. Blok `CITATIONS` boleh dihilangkan atau dibiarkan kosong, dan setiap pasangan `title|index` hanya boleh muncul satu kali.

***Prompt*.** *Prompt* bersifat deterministik dan dibungkus dengan *chat template* milik model. Isinya adalah sebagai berikut.

1. Penjelasan peran agen ("You are a careful evidence-seeking QA agent").
2. Tata bahasa aksi beserta contoh (`ANSWER[United States] CITATIONS[Ursula K. Le Guin|1; The Left Hand of Darkness|1]`) dan aturan format: satu `title|index` per sitasi, dipisahkan dengan titik koma, tidak ada teks setelah kurung penutup, judul disalin persis, tidak pernah mengarang judul, dan hanya menyitasi dari hasil pencarian. Dua aturan terakhir adalah instruksi bagi model. Lingkungan menegakkan bahwa judul termasuk 10 paragraf pertanyaan dan menoleransi perbedaan penulisan judul yang kecil (Subbab 3.4).
3. Pertanyaan dan sisa anggaran pencarian.
4. Seluruh hasil pencarian sejauh ini. Untuk setiap pencarian, *prompt* menampilkan *query* lalu setiap dokumen yang dikembalikan dalam bentuk `[Title]` diikuti baris-baris `index: sentence`.

*Prompt* dibangun ulang dari awal pada setiap giliran, sehingga model selalu melihat seluruh riwayat pencarian. *Prompt* dikirim sebagai satu pesan pengguna tanpa pesan *system*, sehingga isi bagian *system* ditentukan oleh *chat template* model. Lampiran A memuat teks *prompt* untuk satu pertanyaan *holdout* nyata.

***Adapter*.** Pelatihan menggunakan LoRA [18] dengan *rank* r = 16, α = 32, dan *dropout* 0. *Adapter* diterapkan pada seluruh proyeksi *attention* dan MLP (q, k, v, o, gate, up, down). Bobot dasar tetap dibekukan, sehingga hanya sekitar 40 juta dari 7,66 miliar parameter yang dilatih dan pelatihan dapat dilakukan pada satu GPU 48 GB.

## 3.3 *Dataset* dan Pembagian Data

Penelitian ini menggunakan **HotpotQA** [6], versi 1.1, pada *distractor setting*. Pada pengaturan ini, setiap pertanyaan disertai 10 paragraf Wikipedia yang terbagi menjadi dua jenis, yaitu sebagai berikut.

- **Paragraf emas** (*gold paragraphs*), yaitu 2 paragraf yang digunakan anotator untuk menyusun pertanyaan dan jawabannya. Kedua paragraf inilah yang memuat bukti (*evidence*) jawaban. Istilah "emas" pada tugas akhir ini selalu berarti label rujukan yang benar dari anotasi HotpotQA: jawaban emas, paragraf emas, dan kalimat emas.
- **Paragraf pengecoh** (*distractor paragraphs*), yaitu 8 paragraf yang diambil dengan pencarian bigram tf-idf menggunakan pertanyaan sebagai *query*, sehingga tampak relevan tetapi tidak memuat bukti jawaban [6]. Kesepuluh paragraf diacak urutannya, sehingga agen harus menemukan paragraf emas di antara paragraf pengecoh.

Setiap pertanyaan juga dianotasi dengan ***supporting facts***, yaitu kalimat-kalimat di dalam paragraf emas yang dibutuhkan untuk menjawabnya, dalam bentuk pasangan (judul paragraf, indeks kalimat). Pasangan ini disebut kalimat emas. Anotasi inilah yang memungkinkan *reward* bukti tanpa *judge*: benar atau tidaknya suatu sitasi dapat diperiksa dengan mencocokkannya terhadap kalimat emas.

Baris data yang gagal validasi dilewati dan dicatat beserta alasannya, misalnya kalimat kosong atau indeks *supporting fact* di luar jangkauan.

**Pembagian data.** Pembagian data dibuat dengan sebuah skrip dan dicatat dalam sebuah *manifest* bersama dengan *hash* SHA-256 dari berkas sumber (Tabel 3.5).

- **Data latih dan validasi.** Keduanya diambil dari berkas pelatihan HotpotQA menggunakan *hash* deterministik. Setiap ID pertanyaan dipetakan ke sebuah nilai dalam [0; 1) dari 8 *byte* pertama *hash* SHA-256 atas ID tersebut. Pertanyaan masuk ke data latih jika nilainya di bawah 0,95, dan ke data validasi jika sebaliknya. Pembagian ini menghasilkan **85.573 pertanyaan latih** dan **4.452 pertanyaan validasi**, tidak bergantung pada urutan berkas, dan dapat direproduksi secara persis.
- ***Holdout*.** *Holdout* adalah himpunan beku berisi **300 pertanyaan**, yaitu 300 baris valid pertama dari berkas dev HotpotQA, yang disimpan beserta *hash manifest*-nya. Karena diambil berdasarkan urutan berkas dan bukan secara acak, komposisi jenis dan tingkat kesulitan pertanyaannya belum tentu sama dengan seluruh berkas dev.

**Penggunaan *holdout*.** Bobot *reward* tidak pernah di-*tuning*. Pengaturan pelatihan, yaitu *learning rate* (dari 1 × 10⁻⁵ menjadi 2 × 10⁻⁵), ukuran grup (dari 4 menjadi 8), dan jumlah pertanyaan pelatihan (dari 64 menjadi 256), ditetapkan setelah uji *smoke* dan *run* kedua (yang kemudian menjadi R1 *seed* 7). Keduanya dievaluasi pada *holdout*, dan *run* kedua juga pada data validasi. Pengaturan ini sama untuk setiap varian *reward*, sehingga tidak menguntungkan satu varian tertentu. Selain itu, R3, R4, R0-dyn, dan *seed* ketiga (*seed* 9) untuk R0, R1, dan R2 diputuskan setelah hasil *holdout* *seed* 7 dan 8 diketahui (Subbab 3.7). Angka *holdout* untuk R3, R4, dan R0-dyn karena itu bukan estimasi yang sepenuhnya tak bias, dan perbandingannya diperlakukan sebagai eksploratif.

**Sampel pelatihan.** Setiap *run* pelatihan mengambil **256 pertanyaan** dari data latih. Sampel dan urutannya hanya bergantung pada *seed* *run*, sehingga *seed* yang berbeda melihat pertanyaan yang berbeda, sedangkan *run* dengan *seed* yang sama melihat pertanyaan yang sama (Subbab 3.6).

## 3.4 Lingkungan Pencarian

Lingkungan pencarian adalah sesi pencarian yang terbatas dan dapat di-*reset* untuk satu pertanyaan. Lingkungan menjalankan aksi agen, mengembalikan hasil pencarian, dan memvalidasi jawaban akhir beserta sitasinya.

**Korpus dan *retriever*.** Korpus yang dapat dicari adalah **kumpulan tertutup berisi 10 paragraf milik pertanyaan itu sendiri**, yang selalu memuat kedua paragraf emas. Hal ini membuat *retrieval* jauh lebih mudah dibandingkan dengan pencarian *open-domain*: pada episode yang selesai, model tanpa pelatihan sudah memunculkan paragraf untuk 83% *supporting facts* emas (*retrieval recall* 0,829, Tabel 4.1). Dengan demikian, tugas ini lebih dekat dengan pemahaman bacaan (*reading comprehension*) dengan alat penelusur halaman dibandingkan dengan pencarian web, dan istilah "agen pencarian" pada tugas akhir ini selalu berarti pencarian pada kumpulan tertutup tersebut.

*Retrieval* menggunakan Okapi BM25 [13] dengan parameter bawaan pustaka `rank_bm25`. Setiap paragraf diindeks sebagai judulnya ditambah seluruh kalimatnya, yang ditokenisasi menjadi token kata Unicode dengan *case folding*. Setiap pencarian mengembalikan **3 dokumen teratas**. Nilai yang sama dipecahkan berdasarkan urutan dokumen asli, sehingga *retrieval* bersifat deterministik. Setiap dokumen yang dikembalikan ditampilkan secara utuh dengan kalimat-kalimat bernomor.

**Anggaran pencarian.** Satu episode mengizinkan paling banyak **3 pencarian**.

**Validasi sitasi.** Judul yang disitasi dicocokkan dengan judul 10 paragraf milik pertanyaan tersebut. Jika tidak ada judul yang persis sama, pencocokan diulang dengan mengabaikan perbedaan huruf besar/kecil, normalisasi Unicode, tanda baca, dan spasi, selama hasilnya menunjuk ke tepat satu judul. Lingkungan kemudian memeriksa bahwa judul tersebut termasuk 10 paragraf pertanyaan dan bahwa indeks kalimatnya berada dalam jangkauan paragraf itu.

Lingkungan tidak memeriksa apakah paragraf yang disitasi pernah dikembalikan oleh pencarian dalam episode tersebut. *Prompt* meminta agen menyitasi hanya dari hasil pencarian, tetapi aturan itu tidak ditegakkan. Pada data *holdout*, setidaknya 4 episode menyitasi kalimat emas dari paragraf yang tidak pernah diambil, yaitu 2 pada B0, 1 pada R2 *seed* 7, dan 1 pada R4 *seed* 7. Angka ini merupakan batas bawah, karena hanya episode yang cakupannya melebihi *retrieval recall*-nya yang dapat terdeteksi tanpa korpus lengkap.

**Jenis kegagalan.** Sebuah episode dinyatakan **tidak valid** jika terjadi salah satu kondisi berikut.

- *Kegagalan protokol*: keluaran tidak sesuai dengan tata bahasa aksi, misalnya terdapat teks tambahan, sitasi ditempelkan pada aksi `SEARCH`, atau sitasi yang sama ditulis dua kali.
- *Kegagalan bukti*: sebuah sitasi merujuk pada judul yang tidak termasuk 10 paragraf pertanyaan tersebut atau pada indeks kalimat di luar jangkauan. Dalam praktiknya, hal ini hampir selalu berarti judul yang dikarang.
- *Kegagalan anggaran*: agen menghabiskan seluruh pencariannya dan tetap tidak menjawab.

Kode juga mengenali dua jenis kegagalan lain, yaitu galat pencarian di luar habisnya anggaran dan jawaban kosong, tetapi keduanya tidak pernah terjadi pada episode *holdout* maupun *rollout* pelatihan yang dilaporkan. Episode yang tidak valid menerima penalti sebesar −1 selama pelatihan (Subbab 3.5) dan dihitung salah selama evaluasi (Subbab 3.8).

**Celah yang diketahui.** Jawaban tanpa sitasi (`ANSWER[x]` atau `ANSWER[x] CITATIONS[]`) diterima sebagai jawaban yang valid oleh *parser* maupun lingkungan, dan *prompt* juga mengizinkan `CITATIONS[]` untuk kasus ketika tidak ada sitasi yang dapat divalidasi. Celah ini sengaja tidak ditutup agar setiap *run* dilatih dan dievaluasi pada lingkungan yang sama. Bab 4 melaporkan bagaimana *reward outcome-only* memanfaatkan celah ini.

## 3.5 *Reward* yang Diusulkan: VEGR

***Verifiable Evidence-Grounded Reward* (VEGR)** menilai setiap episode yang selesai. Setiap komponennya dihitung secara berbasis aturan dari anotasi emas HotpotQA, sehingga tidak melibatkan *LLM judge*.

Misalkan *G* adalah himpunan *supporting facts* emas (judul, indeks kalimat) dan *C* adalah himpunan kalimat yang disitasi oleh agen. Untuk episode yang valid:

$$
R_{\text{VEGR}} = \text{EM} + \text{F1} + \text{Cov} + \text{Prec} + \text{RetR} - 0.1\cdot\text{Cost} - 0.1\cdot\text{Dup}
$$

Tabel 3.1 mendefinisikan setiap komponen.

**Tabel 3.1.** Komponen VEGR

| Komponen | Definisi | Rentang |
|---|---|---|
| EM | *exact match* ternormalisasi antara jawaban prediksi dan jawaban emas | {0; 1} |
| F1 | F1 tingkat token antara jawaban prediksi dan jawaban emas | [0; 1] |
| Cov (cakupan bukti) | \|G ∩ C\| / \|G\| | [0; 1] |
| Prec (presisi sitasi) | \|G ∩ C\| / \|C\|, didefinisikan 0 jika C kosong | [0; 1] |
| RetR (*retrieval recall*) | proporsi *supporting facts* emas yang paragrafnya muncul pada hasil pencarian mana pun | [0; 1] |
| Cost (biaya pencarian) | jumlah pencarian yang digunakan / anggaran (3) | [0; 1] |
| Dup (penalti *query* duplikat) | (jumlah *query* − jumlah *query* unik) / jumlah *query* | [0; 1) |

Episode yang **tidak valid** menerima **R = −1**, terlepas dari komponen lainnya. Penalti inilah yang membuat VEGR menegakkan tata bahasa aksi. Karena judul yang dikarang termasuk kegagalan bukti, agen tidak dapat memperoleh presisi dengan menyitasi dokumen yang tidak ada di antara 10 paragraf pertanyaan. Sitasi ke paragraf milik pertanyaan yang belum diambil tetap dinilai seperti sitasi lainnya (Subbab 3.4). Karena RetR dihitung per *supporting fact*, paragraf emas yang memuat dua fakta berbobot dua kali lipat dibandingkan dengan paragraf emas yang memuat satu fakta.

Komponen-komponen tersebut terbagi ke dalam tiga kelompok, masing-masing ditujukan untuk jenis kegagalan yang berbeda, yaitu sebagai berikut.

- **Komponen jawaban (EM, F1)** memberi *reward* atas kebenaran jawaban. Keduanya mengikuti metrik jawaban HotpotQA [6]. F1 memberikan nilai parsial, sehingga *reward* menjadi tidak sejarang EM saja.
- **Komponen bukti (Cov, Prec)** memberi *reward* atas penyitasian kalimat yang tepat. Presisi saja dapat dipenuhi dengan menyitasi satu kalimat yang aman, dan cakupan saja dapat dipenuhi dengan menyitasi semua kalimat, sehingga keduanya disertakan, sejalan dengan pemisahan *citation recall* dan *citation precision* pada ALCE [14].
- **Komponen pencarian (RetR, Cost, Dup)** memberi *reward* atas ditemukannya paragraf emas dan memberi penalti atas pencarian yang sia-sia atau berulang. Penalti tersebut diberi bobot kecil (0,1) agar dapat membentuk perilaku pencarian tanpa membuat agen enggan melakukan pencarian sama sekali.

**Asal rumus cakupan bukti dan presisi sitasi.** Kedua rumus komponen bukti mengikuti evaluasi *supporting facts* pada HotpotQA [6, Subbab 5.2]. HotpotQA menilai prediksi *supporting facts* sebagai himpunan kalimat yang dibandingkan dengan himpunan kalimat emas, lalu menghitung presisi P^(sup) dan *recall* R^(sup) per contoh. Kedua nilai tersebut digabung menjadi Sup-F1 dan, bersama metrik jawaban, menjadi Joint-F1. Pada VEGR, presisi sitasi sama dengan P^(sup), yaitu bagian dari kalimat yang disitasi yang merupakan kalimat emas. Cakupan bukti sama dengan R^(sup), yaitu bagian dari kalimat emas yang berhasil disitasi. Istilah "cakupan bukti" adalah penamaan dalam tugas akhir ini, karena HotpotQA menyebutnya *recall supporting facts*. Pemisahan yang sama, yaitu *recall* sitasi dibandingkan dengan presisi sitasi, juga digunakan oleh ALCE [14] untuk menilai teks bersitasi. Bedanya, ALCE menilai dukungan sitasi dengan model NLI, sedangkan VEGR mencocokkan sitasi secara langsung dengan kalimat emas.

**Contoh perhitungan cakupan bukti.** Pada pertanyaan *holdout* `5ae7a8175542993210983ed8`, agen R3 *seed* 8 menjawab "Pedro Rodríguez" dengan benar dan menyitasi tiga kalimat, yaitu `Formula One drivers from Mexico|1`, `2014 Bahrain Grand Prix|3`, dan `2009 Singapore Grand Prix|5` (*Trace* 3, Subbab 4.7). Pertanyaan ini memiliki tiga kalimat emas, dan hanya satu dari tiga sitasi tersebut yang merupakan kalimat emas. Perhitungannya adalah sebagai berikut.

1. |G| = 3 kalimat emas, |C| = 3 kalimat yang disitasi, dan |G ∩ C| = 1 kalimat yang terdapat pada keduanya.
2. Cakupan bukti = |G ∩ C| / |G| = 1/3 ≈ 0,333. Agen hanya menyebutkan satu dari tiga kalimat yang dibutuhkan.
3. Presisi sitasi = |G ∩ C| / |C| = 1/3 ≈ 0,333. Dua dari tiga sitasinya bukan kalimat emas.
4. Komponen lainnya: EM = 1, F1 = 1, RetR = 2/3 (paragraf untuk dua dari tiga kalimat emas terambil), Cost = 2/3 (dua pencarian), dan Dup = 0.
5. R_VEGR = 1 + 1 + 0,333 + 0,333 + 0,667 − 0,1 · 0,667 − 0 ≈ **3,267**. Dengan bobot cakupan 2 (R3), cakupan dihitung dua kali sehingga R ≈ 3,600.

Seandainya agen menyitasi ketiga kalimat emas dan tidak ada kalimat lain, cakupan dan presisi masing-masing bernilai 1 dan *reward*-nya naik sekitar 1,33 poin. Seandainya agen hanya menyitasi satu kalimat emas, presisi bernilai 1 tetapi cakupan tetap 1/3. Contoh terakhir ini menunjukkan mengapa presisi yang tinggi belum berarti bukti yang lengkap (Subbab 4.5).

Komponen-komponen VEGR secara individual bukanlah hal baru. Kontribusi penelitian ini adalah penggabungan komponen-komponen tersebut menjadi *reward* tanpa *judge* untuk agen pencarian *multi-turn*, bersama dengan evaluasi terkontrol atas apa yang diubah oleh kombinasi tersebut (Subbab 3.7). Seluruh bobot merupakan nilai bawaan yang tidak di-*tuning*. Hal ini merupakan keterbatasan yang dibahas pada Bab 5.

**Perbandingan dua jawaban benar.** Tabel 3.2 menilai dua episode *holdout* nyata untuk pertanyaan yang sama, yaitu "2014 S/S is the debut album of a South Korean boy group that was formed by who?". Keduanya menjawab "YG Entertainment" dengan benar. Episode A berasal dari R1 *seed* 8 dan episode B dari R0 *seed* 8. Dengan *reward outcome-only* (R0), keduanya menerima *reward* yang sama, yaitu 1. VEGR membedakan keduanya: episode A sepenuhnya ter-*grounding*, sedangkan episode B tidak memiliki bukti.

**Tabel 3.2.** Komponen VEGR untuk dua jawaban benar

| | Episode A (R1 *seed* 8) | Episode B (R0 *seed* 8) |
|---|---|---|
| Aksi | `SEARCH[S/S 2014 debut South Korean boy group]` → `SEARCH[WINNER YG Entertainment]` → `ANSWER[YG Entertainment] CITATIONS[2014 S/S\|0; Winner (band)\|0]` | `SEARCH[2014 S/S debut album South Korean boy group formed by]` → `ANSWER[YG Entertainment]` |
| EM | 1 | 1 |
| F1 | 1 | 1 |
| Cakupan bukti | 1 (kedua kalimat emas disitasi) | 0 (tidak ada yang disitasi) |
| Presisi sitasi | 1 | 0 |
| *Retrieval recall* | 1 (paragraf kedua fakta emas diambil) | 0,5 (paragraf satu dari dua fakta emas) |
| − 0,1 · biaya pencarian | − 0,1 · 2/3 = − 0,067 | − 0,1 · 1/3 = − 0,033 |
| − 0,1 · penalti duplikasi | 0 | 0 |
| **Total VEGR** | **4,933** | **2,467** |
| *Reward outcome-only* (R0) | 1 | 1 |

**Pembanding dan varian *reward*.** R0 adalah *reward* dasar yang berbeda dari VEGR pada enam komponen sekaligus (Subbab 3.1), sedangkan R2 dan R3 masing-masing mengubah satu aspek VEGR, yaitu sebagai berikut.

- **R0, *outcome-only* (*reward* dasar).** R = EM. Seluruh bobot lainnya bernilai 0, tetapi episode yang tidak valid tetap menerima −1. R0 mengambil gagasan *reward* berbasis EM dari Search-R1 [2], tetapi bukan replikasinya: Search-R1 tidak memakai *reward* format dan hanya mengembalikan pesan agar model "berpikir ulang" ketika aksinya tidak valid, sedangkan R0 memberi −1 pada setiap episode yang tidak valid, termasuk kegagalan bukti.
- **R2, penalti bersyarat (*gated*).** Varian ini sama dengan VEGR, kecuali bahwa penalti biaya pencarian dan duplikasi hanya diterapkan ketika EM = 1. Tujuannya adalah agar pencarian tidak diberi penalti pada pertanyaan yang belum berhasil dijawab oleh agen.
- **R3, cakupan ×2.** Varian ini adalah VEGR dengan bobot cakupan bukti dinaikkan dari 1 menjadi 2. Varian ini menyasar pola sitasi "presisi tetapi tidak lengkap" yang dijelaskan pada Bab 4.

**Apa yang diberi *reward* oleh R0 terkait sitasi.** Pada R0, episode yang valid memperoleh nilai EM-nya (0 atau 1), baik dengan maupun tanpa sitasi. Sitasi tidak pernah dapat menambah *reward*, tetapi sitasi yang gagal validasi membuat episode tersebut menjadi kegagalan bukti yang bernilai −1. Dengan demikian, berdasarkan definisinya, **R0 menghukum sitasi yang tidak valid tetapi tidak pernah memberi imbalan atas sitasi yang benar**. Model dasar sering menghasilkan sitasi yang tidak valid: 85 dari 130 kegagalan *holdout* B0 merupakan kegagalan bukti (Subbab 4.2).

Dari definisi ini, hipotesis penelitian ini adalah bahwa R0 dapat mendorong *policy* untuk berhenti menyitasi. Log *rollout* pelatihan memungkinkan hipotesis ini diperiksa sebagian. Log tersebut tidak menyimpan daftar sitasi, sehingga *rollout* valid hanya dapat dibagi menjadi dua kelompok, yaitu **valid dengan ≥1 sitasi emas** (presisi sitasi > 0) dan **valid tanpa sitasi emas**, yang mencakup *rollout* tanpa sitasi sama sekali maupun *rollout* yang hanya menyitasi kalimat bukan emas. Sebuah "upaya menyitasi" dihitung sebagai kegagalan bukti ditambah *rollout* valid dengan ≥1 sitasi emas. Pada 32 langkah pertama R0 *seed* 8 (8 × 32 = 256 *rollout* pelatihan), 176 *rollout* merupakan upaya menyitasi. Sebanyak 90 di antaranya (51%) merupakan kegagalan bukti (−1), dan 86 *rollout* valid dengan ≥1 sitasi emas memperoleh rata-rata 0,50. Sebanyak 42 *rollout* valid tanpa sitasi emas memperoleh rata-rata 0,36. Rata-rata *reward* sebuah upaya menyitasi karena itu adalah (86 × 0,50 − 90) / 176 ≈ **−0,27**, dibandingkan dengan **+0,36** untuk kelompok tanpa sitasi emas. Pada *seed* ini, data konsisten dengan hipotesis tersebut.

Statistik yang sama tidak membedakan *run* yang kehilangan sitasi dari yang tidak. Pada blok pertama R0 *seed* 9, yang juga kehilangan sitasi, hanya 23% upaya menyitasi yang tidak valid, dan rata-rata *reward* sebuah upaya (+0,29) sedikit lebih tinggi daripada kelompok tanpa sitasi emas (0,26). Pada R0 *seed* 7, yang tetap menyitasi, 36% upaya tidak valid dan rata-ratanya (−0,05) lebih rendah daripada kelompok tanpa sitasi emas (0,30). Dengan demikian, penalti −1 menjelaskan mengapa hilangnya sitasi *mungkin* terjadi pada R0, tetapi tidak menjelaskan mengapa hal itu terjadi pada *seed* 8 dan 9 dan tidak pada *seed* 7.

Perbedaan yang lebih mendasar terletak pada selisih *reward* di dalam grup. Pada R0, *rollout* valid dengan ≥1 sitasi emas hanya memperoleh sedikit lebih banyak daripada *rollout* valid tanpa sitasi emas (selisih rata-rata 0,1–0,4 pada blok-blok awal), dan selisih ini berasal dari jawaban yang lebih sering benar, bukan dari sitasinya. VEGR memberi *reward* secara langsung atas sitasi yang benar melalui komponen cakupan dan presisinya. Pada blok pertama R1 *seed* 8, *rollout* valid dengan ≥1 sitasi emas memperoleh rata-rata 3,35 dan *rollout* valid tanpa sitasi emas 1,37, yaitu selisih sekitar 2 poin *reward*. Subbab 4.5 menyajikan data per blok untuk *run*-*run* tersebut.

## 3.6 Algoritma Pelatihan GRPO

Pelatihan menggunakan ***policy gradient* bergaya GRPO dengan normalisasi grup**, yang mengikuti *Group Relative Policy Optimization* (GRPO) [3] dan diperluas untuk episode *multi-turn*. Algoritma ini berbeda dari GRPO penuh dalam hal-hal yang dinyatakan pada bagian "Fungsi objektif" di bawah.

***Rollout*.** Untuk setiap pertanyaan pelatihan, *policy* saat ini menjalankan **G = 8** episode penuh, dengan *sampling* pada *temperature* 1,0 dan top-p 1,0. Setiap aksi dibatasi hingga 128 token baru. Setiap episode menerima satu nilai *reward* skalar dari fungsi *reward* milik *run* tersebut (R0–R4, R0-dyn).

***Advantage*.** Setiap *reward* dinormalisasi di dalam grupnya:

$$
A_i = \frac{r_i - \operatorname{mean}(r_{1..G})}{\operatorname{std}(r_{1..G}) + 10^{-6}}
$$

Seluruh aksi dalam satu episode berbagi *advantage* episode tersebut. Jika kedelapan *reward* dalam satu grup bernilai sama, setiap *advantage* bernilai nol dan grup tersebut tidak membawa sinyal pembelajaran. **Grup dengan variansi nol** seperti ini **dilewati**, yaitu tidak dilakukan pembaruan. Jumlah langkah yang dilewati dicatat sebagai diagnostik (Bab 4). Sebaliknya, selama *reward* dalam grup tidak seluruhnya sama, pembagian dengan simpangan baku membuat *advantage* berskala satuan berapa pun besar selisihnya, sehingga grup yang hanya berbeda karena satu pencarian tambahan (0,1 × 1/3 ≈ 0,033 pada VEGR) menghasilkan pembaruan sebesar grup yang berbeda karena jawaban benar atau salah.

**Fungsi objektif.** Pembaruan menggunakan *clipped surrogate* bergaya PPO [15] dengan ε = 0,2. *Log-probability* setiap *rollout* adalah **rata-rata atas seluruh token yang dihasilkan agen dalam episode tersebut**, mencakup seluruh aksinya. Token *prompt* dan observasi dikecualikan. Dengan demikian, rasio ρ dihitung satu kali per *rollout*.

$$
\mathcal{L} = -\,\mathbb{E}\left[\min\left(\rho\,A,\; \operatorname{clip}(\rho,\,1-\varepsilon,\,1+\varepsilon)\,A\right)\right],
\qquad \rho = \exp\left(\log\pi_\theta - \log\pi_{\theta_\text{old}}\right)
$$

**Tidak ada penalti KL** terhadap model referensi. *Rollout* yang tidak valid tetap berada di dalam *batch*, sehingga penalti −1 secara aktif mendorong *policy* menjauh dari aksi yang tidak sesuai format.

**Perilaku fungsi objektif dalam praktik.** Terdapat tiga sifat implementasi yang menyebabkan fungsi objektif ini bukan fungsi objektif GRPO bergaya PPO yang penuh, yaitu sebagai berikut.

- **Satu pembaruan per grup.** Delapan *rollout* dari setiap pertanyaan digunakan untuk tepat satu langkah *optimizer*. Tidak ada *epoch* berulang dan tidak ada penggunaan ulang *minibatch*, sehingga *policy* lama dan baru identik pada saat pembaruan, ρ ≈ 1, dan *clipping* rasio PPO tidak pernah aktif. Fraksi *clip* yang tercatat bernilai **0 pada seluruh 16 *run***. Selisih *log-probability* lama/baru yang tercatat tetap kecil tetapi tidak nol. *Log-probability* lama dihitung saat koleksi *rollout* dalam mode evaluasi, sedangkan yang baru dihitung dalam mode pelatihan dengan *gradient checkpointing*, keduanya dalam bf16. Karena bobotnya identik, selisih tersebut berasal dari perbedaan jalur komputasi numerik. Pada praktiknya, pembaruan ini adalah *policy gradient* bergaya REINFORCE [35] dengan *baseline* rata-rata grup dan *advantage* yang dinormalisasi per grup.
- **Tanpa komponen KL.** DAPO [17] dan Dr. GRPO [16] juga menghilangkan penalti KL.
- **Rasio tingkat sekuens.** Rasio menggunakan *rata-rata* *log-probability* token dari setiap episode, sehingga dihitung per sekuens, bukan per token. Hal ini menskalakan gradien setiap *rollout* dengan satu per panjangnya, yaitu normalisasi panjang yang diidentifikasi oleh Dr. GRPO sebagai sumber bias [16].

*Clipping* rasio dengan ε = 0,2 tetap dipertahankan dalam kode dan dalam deskripsi di atas demi kelengkapan, tetapi tidak berperan pada *run* yang dilaporkan.

**Optimasi.** AdamW dengan *learning rate* 2 × 10⁻⁵ tanpa *scheduler* maupun *warmup*, *gradient-norm clipping* pada 1,0, bf16, dan *gradient checkpointing*. *Loss* setiap *rollout* dibagi dengan ukuran grup (8) sebelum gradiennya diakumulasi. Satu *run* melakukan satu kali lintasan atas 256 pertanyaannya (256 langkah, satu pertanyaan per langkah), kecuali R0-dyn (512 langkah; lihat di bawah). Karena setiap *run* hanya terdiri atas satu *epoch*, kemajuan pelatihan pada Bab 4 dilaporkan per blok 32 langkah. *Gradient clipping* aktif pada sebagian besar langkah: norma gradien sebelum *clipping* melebihi 1,0 pada 86–100% langkah pembaruan untuk R0–R3 dan pada 59–72% untuk R4.

***Seed*.** *Run* menggunakan *seed* 7, 8, dan 9. Sebuah *seed* menentukan sampel dan urutan 256 pertanyaan pelatihan sekaligus *state* acak yang digunakan untuk *sampling* *rollout*. Dengan demikian, perbedaan antar-*seed* mencerminkan derau data maupun derau optimasi. Pelatihan pada GPU tidak deterministik secara bit: R0 *seed* 8 dan R0-dyn *seed* 8 melatih pertanyaan yang sama dengan urutan yang sama, menghasilkan *rollout* yang identik pada langkah pertama, tetapi *reward*-nya sudah berbeda sejak langkah kedua, setelah pembaruan pertama. Hasil sebuah *seed* karena itu tidak dapat diulang secara persis, dan *seed* yang sama dapat menghasilkan lintasan pelatihan yang berbeda.

**Varian *optimizer*.** Dua varian mengubah prosedur optimasi alih-alih *reward*, yaitu sebagai berikut.

- **GRPO tanpa normalisasi std**, digunakan pada R4. *Advantage* dihitung tanpa pembagian dengan simpangan baku, A_i = r_i − mean(r), yang merupakan salah satu dari dua perubahan yang diusulkan pada Dr. GRPO [16]. R4 **bukan Dr. GRPO penuh**: R4 tetap mempertahankan normalisasi panjang per sekuens yang dijelaskan di atas. Tanpa pembagian ini, grup dengan sebaran *reward* yang besar menghasilkan *advantage* yang lebih besar daripada grup yang selisih *reward*-nya kecil. Karena *gradient clipping* aktif pada sebagian besar langkah dan AdamW menormalkan skala gradien, perubahan ini terutama menggeser bobot relatif antar-grup. Pada R4, *clipping* aktif pada lebih sedikit langkah (59–72%) dibandingkan dengan *run* lainnya.
- ***Dynamic sampling***, digunakan pada R0-dyn, mengikuti gagasan DAPO [17]. Grup dengan variansi nol tidak dihitung ke dalam anggaran langkah. *Trainer* terus mengambil pertanyaan, dari kumpulan berisi hingga 512 pertanyaan, sampai telah melakukan 256 pembaruan atau kumpulan tersebut habis. Karena setiap langkah hanya memuat satu pertanyaan dan grup dengan variansi nol sudah dilewati pada *run* standar, implementasi ini setara dengan melanjutkan pelatihan pada pertanyaan tambahan sampai 256 pembaruan tercapai. Hal ini berbeda dengan DAPO, yang mengisi ulang sebuah *batch* berisi banyak grup agar ukuran *batch* efektifnya tetap. Pemilihan pertanyaan mengurutkan kumpulan pertanyaan berdasarkan *hash* lalu memotongnya, sehingga 256 pertanyaan pertama R0-dyn *seed* 8 sama, dengan urutan yang sama, dengan 256 pertanyaan R0 *seed* 8. Komponen DAPO lainnya (*clip-higher*, *loss* tingkat token) tidak diimplementasikan.

***Pseudocode*.**

```
for step, question in enumerate(sample(train_split, 256, seed)):
    # multi-turn
    episodes = [run_episode(policy, question) for _ in range(8)]
    # VEGR or variant; invalid → −1
    rewards  = [reward_fn(ep, gold(question)) for ep in episodes]
    if all rewards equal:
        log_skip(step); continue          # zero-variance group
    # R4: no division
    A = (rewards − mean(rewards)) / (std(rewards) + 1e-6)
    for each episode i:
        # mean over all action tokens
        ρ = exp(mean_token_logp_θ(actions_i) − mean_token_logp_old(actions_i))
        loss += −min(ρ·A_i, clip(ρ, 0.8, 1.2)·A_i) / 8
    update LoRA with AdamW (lr 2e-5, grad clip 1.0)
```

## 3.7 Skenario Eksperimen

Tabel 3.3 memuat seluruh skenario eksperimen. Setiap skenario menjalankan alur pada Gambar 3.1 dengan fungsi *reward* atau prosedur optimasi yang berbeda.

**Tabel 3.3.** Matriks eksperimen

| ID | Nama | *Reward* | *Optimizer* | *Seed* |
|---|---|---|---|---|
| B0 | *Baseline* tanpa pelatihan | — (tanpa pelatihan) | — | — |
| R0 | *Outcome-only* | EM, tidak valid −1 | GRPO | 7, 8, 9 |
| R1 | **VEGR** | VEGR penuh | GRPO | 7, 8, 9 |
| R2 | *Gated* | VEGR, penalti hanya ketika EM = 1 | GRPO | 7, 8, 9 |
| R3 | Cakupan ×2 | VEGR, bobot cakupan 2 | GRPO | 7, 8, 9 |
| R4 | GRPO tanpa normalisasi std | VEGR penuh | GRPO, tanpa pembagian std (normalisasi panjang dipertahankan) | 7, 8, 9 |
| R0-dyn | *Outcome-only* + *dynamic sampling* | EM, tidak valid −1 | GRPO, pelatihan dilanjutkan hingga 256 pembaruan | 8 |

Perbandingan-perbandingan tersebut dipetakan ke pertanyaan penelitian sebagai berikut.

- **RQ1 (utama untuk R0, R1, R2; eksploratif untuk R3, R4):** setiap metode yang dilatih dibandingkan dengan B0.
- **RQ2 (utama):** R1 dibandingkan dengan R0. *Optimizer*, *seed*, dan pertanyaan pelatihan sama, sedangkan paket *reward* berbeda (Subbab 3.1).
- **RQ3 (eksploratif):**
  - R2 dan R3 dibandingkan dengan R1 untuk menelaah rancangan *reward*.
  - R4 dibandingkan dengan R1 untuk menelaah penghapusan normalisasi simpangan baku.
  - R0-dyn dibandingkan dengan R0 *seed* 8 untuk menelaah apakah pelatihan yang lebih panjang mengubah perilaku sitasi R0.

**Jumlah *seed* yang berbeda antar-skenario.** Jumlah *seed* sengaja dibedakan sesuai peran setiap skenario. Metode yang dibandingkan secara statistik (R0–R4) dilatih dengan tiga *seed*, yaitu jumlah minimal yang dianjurkan untuk melaporkan variansi antar-*run* [10]. B0 tidak dilatih, sehingga hanya dievaluasi satu kali dengan *greedy decoding* yang deterministik. R0-dyn dijalankan pada satu *seed* saja (*seed* 8) sebagai **eksperimen diagnostik**, bukan sebagai salah satu metode utama, dengan alasan sebagai berikut.

- **Apa yang dapat diuji.** Pada 256 langkah pertamanya, R0-dyn melatih pertanyaan yang sama, dengan urutan yang sama dan konfigurasi algoritma yang sama, seperti R0 *seed* 8. Perbedaannya hanya bahwa pelatihan R0-dyn dilanjutkan pada 256 pertanyaan berikutnya sampai 256 pembaruan tercapai. Karena pelatihan tidak deterministik secara bit, kedua *run* tetap menempuh lintasan yang berbeda sejak langkah kedua: pada langkah 1–96, R0 *seed* 8 melewati 29 langkah dan R0-dyn 19 langkah, dan pada langkah 1–256 masing-masing 128 dan 107 langkah. Dengan demikian, R0-dyn dapat menunjukkan dua hal, yaitu apakah hilangnya sitasi pada R0 *seed* 8 terulang pada *run* kedua dengan konfigurasi awal yang sama, dan apakah pelatihan tambahan dengan lebih banyak pembaruan memulihkan sitasi. R0-dyn tidak dapat menunjukkan apakah perubahan *optimizer* mencegah hilangnya sitasi, karena hingga langkah ke-256 kedua *run* menjalankan prosedur yang sama. R0-dyn juga melihat dua kali lebih banyak pertanyaan, sehingga perbedaan akurasinya dengan R0 *seed* 8 tidak dapat diatribusikan pada *dynamic sampling* itu sendiri.
- **Alasan satu *seed* (waktu komputasi).** R0 *seed* 8 melewati 128 dari 256 langkahnya, sehingga mencapai 256 pembaruan diperkirakan membutuhkan hingga sekitar 512 pertanyaan, kira-kira dua kali lipat jumlah *rollout* *run* standar. Uji ini semula direncanakan pada dua *seed*, kemudian dipangkas menjadi satu agar *run* tiga *seed* dari metode-metode utama (R0–R4) tetap dapat diselesaikan dalam waktu komputasi yang tersedia. Pada pelaksanaannya, *run* ini mengambil seluruh **512** pertanyaan dan mencapai **250** pembaruan, kurang 6 dari target 256, karena 262 grup yang diambil memiliki variansi nol. Langkah-langkah pelatihannya memakan waktu total **2,64 jam**, dibandingkan dengan 1,1–2,7 jam untuk *run* standar.
- **Alasan *seed* 8.** Pada *seed* ini, pelatihan *outcome-only* kehilangan sitasi. Hanya *seed* tempat hal itu terjadi yang dapat menunjukkan apakah sitasi pulih dengan pelatihan tambahan.
- **Cara membacanya.** R0-dyn ditafsirkan sebagai studi kasus satu *run*, bukan sebagai perbandingan antarmetode yang didukung secara statistik.

**Terencana vs. *post hoc*.** B0, R0, R1, dan R2 pada *seed* 7 dan 8 telah ditetapkan sebelum *batch* multi-*seed* dijalankan. Pada saat itu, baru terdapat satu *run* R1 *seed* 7. Setelah hasil *holdout* *batch* tersebut dianalisis, ditambahkan dua hal. Pertama, *seed* ketiga (*seed* 9) untuk R0, R1, dan R2, agar setiap metode utama memenuhi standar minimal tiga *seed* [10]. Kedua, varian R3, R4, dan R0-dyn, yang merespons pengamatan tertentu, yaitu rendahnya cakupan bukti, serta langkah yang dilewati dan hilangnya sitasi pada R0. Penambahan *seed* 9 berarti jumlah *seed* untuk perbandingan utama juga diputuskan setelah hasil dua *seed* pertama diketahui. Seluruh *run* *seed* 9 dilaporkan apa adanya, tanpa ada *run* yang dibuang. R3, R4, dan R0-dyn merupakan **tindak lanjut eksploratif**, dan Bab 4 melaporkannya sebagai demikian, bukan sebagai uji konfirmatori.

## 3.8 Protokol Evaluasi

**Pengaturan.** Setiap model dievaluasi pada **300 pertanyaan *holdout*** beku yang sama, dengan *greedy decoding* dan **tanpa percobaan ulang**. Jika sebuah aksi tidak sesuai format, episode tersebut gagal.

**Metrik.** Metrik yang digunakan adalah sebagai berikut.

- **EM (seluruh episode)** adalah metrik akurasi utama. Setiap episode yang tidak valid dihitung 0, sehingga penyebutnya selalu 300.
- **Episode selesai** adalah jumlah episode valid dari 300 episode. Kegagalan dirinci berdasarkan jenisnya (bukti, protokol, anggaran).
- **Metrik kualitas** meliputi F1, cakupan bukti, presisi sitasi, *retrieval recall*, dan jumlah pencarian per episode. Setiap metrik dirata-ratakan hanya atas episode yang selesai, menggunakan definisi pada Tabel 3.1. Karena presisi bernilai 0 untuk episode tanpa sitasi, presisi sitasi mencampur "tidak menyitasi" dengan "menyitasi kalimat yang salah". Oleh karena itu, **presisi saat menyitasi**, yaitu presisi yang dirata-ratakan hanya atas episode selesai yang memiliki setidaknya satu sitasi, juga dilaporkan.
- **EM pada episode selesai** adalah EM yang dirata-ratakan hanya atas episode yang valid. Metrik ini memisahkan kualitas jawaban dari kepatuhan terhadap protokol.
- **Metrik pada himpunan bersama.** Jumlah episode selesai berbeda antar-*run* (170 untuk B0, 278–300 untuk *run* terlatih), sehingga rata-rata atas episode selesai dihitung pada himpunan pertanyaan yang berbeda. Untuk mengendalikan efek seleksi ini, EM, presisi sitasi, dan cakupan juga dihitung pada **136 pertanyaan yang diselesaikan oleh semua *run***.
- ***Precision-grounded* EM** menghitung sebuah episode hanya jika EM = 1, episode tersebut memiliki setidaknya satu sitasi, dan setiap sitasinya merupakan *supporting fact* emas (presisi 1). Metrik ini ketat terhadap sitasi yang *salah*, tetapi longgar terhadap sitasi yang *hilang*: jawaban benar yang menyitasi satu dari dua kalimat emas tetap dihitung sepenuhnya ter-*grounding*. Metrik ini dilaporkan bersama tiga ukuran yang lebih ketat atau lebih standar, yaitu sebagai berikut.
  - ***Lenient grounded* EM:** EM = 1 dan setidaknya satu sitasi emas.
  - **Sup-EM:** himpunan yang disitasi sama dengan himpunan emas (presisi 1 dan cakupan 1), seperti pada evaluasi *supporting fact* HotpotQA [6].
  - **Joint-EM dan Joint-F1:** metrik gabungan yang mengikuti HotpotQA. Joint-EM mensyaratkan EM dan Sup-EM sekaligus. Joint-F1 yang dilaporkan merupakan aproksimasi: metrik HotpotQA resmi mengalikan presisi jawaban dengan presisi *supporting fact* dan *recall* jawaban dengan *recall supporting fact*, sedangkan tugas akhir ini memakai F1 jawaban pada kedua sisi (P = F1 × presisi sitasi, R = F1 × cakupan).
- **Tingkat sitasi** adalah proporsi episode selesai yang memiliki setidaknya satu sitasi.
- **Diagnostik perilaku** berasal dari log episode dan log pelatihan. Diagnostik ini mencakup proporsi episode yang mengulang *query*, distribusi jumlah sitasi dan indeks kalimat yang disitasi, tingkat ketidakvalidan selama pelatihan, serta jumlah langkah pelatihan yang dilewati (variansi nol).

**Metrik utama per pertanyaan penelitian.** Untuk RQ1, metrik utamanya adalah EM atas seluruh 300 episode. Untuk RQ2, terdapat dua metrik utama: *precision-grounded* EM untuk sisi *grounding* dan EM atas 300 episode untuk sisi akurasi, dengan tingkat sitasi sebagai indikator langsung jalan pintas. Metrik lainnya bersifat pendukung. Klausul "tanpa kehilangan akurasi" pada RQ2 merupakan pertanyaan non-inferioritas, tetapi tidak ada margin non-inferioritas yang ditetapkan sebelum eksperimen. Oleh karena itu, klausul tersebut dinilai dari batas bawah SK selisih EM: batas bawah itu menunjukkan penurunan akurasi terbesar yang masih konsisten dengan data, dan penurunan yang lebih kecil dari itu tidak dapat disingkirkan.

**Statistik.** Mengikuti Agarwal dkk. [10] dan Henderson dkk. [9], hasil dari satu *run* tidak dianggap konklusif. Selang kepercayaan (SK) yang digunakan adalah sebagai berikut.

- **Selang utama: *paired bootstrap* [36] yang memperhitungkan *seed* (hierarkis).** Setiap dari 10.000 *resample* mengambil 3 *seed* dengan pengembalian untuk setiap metode, dan 300 pertanyaan *holdout* dengan pengembalian yang digunakan bersama oleh metode-metode yang dibandingkan. Dengan demikian, selang persentil 95% mencakup variasi dari pemilihan *seed* maupun pemilihan pertanyaan. Sebuah perbedaan dinyatakan signifikan hanya jika selang ini tidak memuat 0. Dengan hanya 3 *seed*, pengambilan ulang *seed* hanya menghasilkan 10 kombinasi berbeda per metode, sehingga distribusinya sangat diskret dan cenderung meremehkan variansi antar-*seed*. Oleh karena itu, untuk R1 − R0, seluruh 9 pasangan *seed* (R1 *seed* *i* dibandingkan dengan R0 *seed* *j*) juga dilaporkan sebagai pelengkap yang transparan.
- **Selang sekunder: *paired bootstrap* hanya atas pertanyaan.** Skor setiap pertanyaan dirata-ratakan atas tiga *seed* milik metode tersebut, dan hanya pertanyaan yang di-*resample* (10.000 *resample*, *seed* RNG 0). Selang ini mengabaikan variansi *seed* dan lebih sempit; selang ini dilaporkan hanya sebagai rujukan. Perbandingan antara satu *run* dengan satu *run* lainnya (misalnya R0-dyn *seed* 8 dibandingkan dengan R0 *seed* 8) hanya dapat diuji dengan cara ini, sehingga selangnya tidak mencerminkan variabilitas antar-*run*.
- Hasil setiap metode dilaporkan per *seed* dan sebagai rata-rata ± simpangan baku antar-*seed*. Proporsi atas *seed*, seperti berapa banyak *seed* yang kehilangan sitasi, diberikan dengan selang Clopper–Pearson 95% [37].
- **Perbandingan utama dan eksploratif.** Perbandingan utama adalah R0, R1, dan R2 dibandingkan dengan B0 (RQ1) serta R1 − R0 (RQ2). Seluruh perbandingan lainnya bersifat **eksploratif**, yaitu R3 dan R4 dibandingkan dengan B0, R2, R3, dan R4 dibandingkan dengan R1, serta R0-dyn dibandingkan dengan R0 *seed* 8. Perbandingan tersebut tidak dikoreksi untuk perbandingan berganda, dan R3, R4, serta R0-dyn dirancang setelah hasil *holdout* diketahui (Subbab 3.7). Dengan tiga *seed*, hasil eksploratif yang tidak signifikan dilaporkan sebagai **tidak konklusif**, bukan sebagai bukti tidak adanya efek.

## 3.9 Implementasi dan Komputasi

**Perangkat lunak.** Python 3.11 (dikelola dengan `uv`), Transformers 4.57.6, PEFT 0.21.2, dan `rank_bm25`. Versi PyTorch berbeda antar-*run*: *run* pertama (B0 dan R1 *seed* 7) menggunakan 2.14.1+cu130, sedangkan seluruh *run* berikutnya menggunakan 2.11.0+cu128, karena *driver* GPU pada mesin berikutnya hanya mendukung CUDA 12.8. Hal ini dilaporkan sebagai keterbatasan ketika melakukan perbandingan dengan R1 *seed* 7. Pelatihan dan evaluasi dijalankan secara *offline* dari bobot lokal.

**Perangkat keras dan waktu komputasi.** Seluruh pelatihan dan evaluasi dijalankan pada satu GPU yang disewa melalui layanan *cloud* RunPod. Tabel 3.4 merangkum spesifikasi perangkat keras dan waktu komputasinya. Waktu pelatihan dihitung dari jumlah waktu per langkah yang tercatat pada log pelatihan setiap *run*.

**Tabel 3.4.** Perangkat keras dan waktu komputasi

| Aspek | Nilai |
|---|---|
| GPU | 1 × NVIDIA L40S, memori 48 GB |
| Presisi numerik | bf16, dengan *gradient checkpointing* |
| Memori GPU puncak saat pelatihan | 17–19 GiB |
| Waktu pelatihan per *run* standar (256 langkah) | 1,1–2,7 jam |
| Waktu pelatihan R0-dyn (512 langkah) | 2,6 jam |
| Waktu satu *run* termasuk evaluasi *holdout* | sekitar 2 jam |
| Total waktu pelatihan 16 *run* | sekitar 33 jam GPU |

**Reproduktibilitas.** Setiap *run* didefinisikan oleh sebuah berkas konfigurasi dan menyimpan metadata (versi pustaka, *seed*, konfigurasi), ringkasan hasil, log setiap episode evaluasi, dan metrik pelatihan per langkah. Setiap *batch* eksperimen juga memiliki catatan bertanggal. Tabel 3.5 memuat lokasi artefak-artefak tersebut di repositori.

**Tabel 3.5.** Lokasi artefak di repositori

| Artefak | Lokasi |
|---|---|
| Lingkungan pencarian | `src/needle/environment/search.py` |
| Fungsi *reward* (VEGR dan varian) | `src/needle/rewards/__init__.py` |
| Pembangun *prompt* | `src/needle/inference/runner.py` |
| Pelatihan GRPO | `scripts/train_grpo.py`, `src/needle/training/grpo.py` |
| Pembagian data dan *manifest* | `scripts/prepare_splits.py`, `configs/evaluation/` |
| Konfigurasi *run* | `configs/training/` |
| Hasil per *run* (episode dan metrik pelatihan) | `results/<tanggal>/<run>/` |
| Catatan eksperimen | `logs/` |
| Skrip analisis, tabel, dan gambar | `scripts/make_thesis_figures.py`, `scripts/make_training_dynamics.py`, `scripts/thesis_analysis_checks.py`, `scripts/peer_review_reanalysis.py` |

**Batasan reproduktibilitas.** *Adapter* LoRA dari 15 dari 16 *run* terlatih disimpan secara lokal dan tidak dimasukkan ke repositori karena ukurannya (sekitar 470 MB per *adapter*). *Adapter* R1 *seed* 7 tidak disimpan dan telah hilang, tetapi episode *holdout* dan log pelatihannya masih tersedia. Metadata *run* mencatat versi pustaka tetapi tidak mencatat *commit* git, sehingga versi kode setiap *run* hanya dapat diketahui dari catatan *batch*. Pelatihan pada GPU juga tidak deterministik secara bit (Subbab 3.6), sehingga menjalankan ulang sebuah konfigurasi tidak menjamin angka yang persis sama.
