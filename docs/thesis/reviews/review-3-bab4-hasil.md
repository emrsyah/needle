# Review 3 — Bab 4 Hasil dan Analisis

*Draf kerja hasil bantuan AI. Penulis tetap wajib memverifikasi setiap temuan sebelum menindaklanjutinya.*

- **Berkas yang direview:** `docs/thesis/chapters-id/bab4-hasil-dan-analisis.md` (359 baris)
- **Standar penilaian:** Tugas Akhir S1 (novelty bukan tujuan). Fokus review: kebenaran angka, interpretasi selang kepercayaan (SK), klaim mekanisme, serta konsistensi tabel, gambar, dan istilah.
- **Sumber data mentah yang dicek:**
  - `results/2026-10-13-peer-review-reanalysis/output.md`
  - `docs/thesis/figures/figures_data.json`
  - `logs/2026-10-10-grpo-run2.md`, `logs/2026-10-10-r0-r1-r2-batch.md`, `logs/2026-10-11-fixes-batch.md`, `logs/2026-10-13-peer-review-reanalysis.md`
  - `episodes.jsonl` dan `metrics.jsonl` mentah, saya hitung ulang secara lokal dan *read-only*
  - `scripts/peer_review_reanalysis.py`, `scripts/train_grpo.py`, `configs/training/*.json`

## Ringkasan

Bab 4 sudah kuat untuk ukuran S1. Hampir semua angka cocok dengan data mentah, termasuk seluruh isi Tabel 4.1, 4.2, 4.3, 4.5, dan 4.6 serta kelima *trace*. Bab ini juga sudah memisahkan perbandingan utama dari yang eksploratif, memakai SK yang memperhitungkan *seed*, dan memuat bagian "hal yang tidak didukung". Ancaman validitasnya juga ditulis jujur, termasuk Goodhart, paket *reward*, dan versi PyTorch.

Masalah terbesarnya ada pada **interpretasi**, bukan pada angka:

1. Eksperimen R0-dyn, sebagaimana diimplementasikan, **tidak dapat menguji** hipotesis yang dinyatakan di Subbab 4.5 dan 4.6.
2. Penjelasan mekanisme "R0 berhenti menyitasi karena −1 untuk sitasi salah" disajikan sebagai sebab yang terbukti. Padahal data rollout yang sama tidak membedakan *run* yang runtuh dari yang tidak. Label pada Tabel 4.4 juga tidak sesuai dengan cara kategorinya dihitung.
3. Beberapa klaim kesetaraan dan klaim "signifikan" pada RQ2 dan RQ3 masih terlalu kuat.
4. Ada satu klaim faktual yang bertentangan dengan Tabel 4.1, dan satu SK yang tidak cocok dengan log.

---

## Temuan MAJOR

### M1. R0-dyn tidak menguji hipotesis "lebih banyak pembaruan informatif mencegah keruntuhan"

- **Lokasi:**
  - bab4:184 ("R0-dyn menguji hipotesis ini … tetap runtuh dengan jadwal yang sama")
  - bab4:216–223 ("*dynamic sampling* tidak mencegah jalan pintas tanpa sitasi")
  - bab4:235, bab4:329, bab4:344
  - terkait juga bab3:249
- **Observasi:**
  - Di `scripts/train_grpo.py:152–167`, *dynamic sampling* hanya mengubah satu hal: pelatihan tidak berhenti setelah 256 pertanyaan, melainkan terus mengambil pertanyaan sampai 256 pembaruan tercapai. Grup dengan variansi nol tetap tidak menghasilkan gradien, persis seperti pada R0 biasa.
  - Selisih konfigurasi `r0_outcome_seed8.json` dengan `r0_dynsample_seed8.json` hanya `num_train_examples` (256 → 512) dan `dynamic_sampling_target_updates: 256`.
  - Saya membandingkan `metrics.jsonl` kedua *run*. Urutan `question_id` **identik pada 256 langkah pertama**. Seed rollout juga sama (`rollout_seed(seed, qid:step, member)`).
  - `train_grpo.py` tidak memakai *LR scheduler* maupun *warmup*. Panjang jadwal (256 vs 512) karena itu tidak mengubah *learning rate*.
  - Meskipun begitu, jumlah langkah yang dilewati pada langkah 1–96 berbeda: 29 pada R0 dan 19 pada R0-dyn. Penyebabnya bisa nondeterminisme GPU atau perbedaan kode. Antrean `fixes` menarik *commit* `1ebe62c`, sedangkan R0 s8 dijalankan lebih awal, dan `run_metadata.json` tidak mencatat *commit*. Perbedaan versi kode tidak dapat disingkirkan.
  - Keruntuhan terjadi pada langkah 65–96, yaitu saat kedua *run* masih menjalankan konfigurasi algoritma yang sama.
- **Mengapa penting:** Penguji yang membaca kode akan melihat bahwa R0-dyn pada dasarnya adalah **replikasi R0 s8 (di bawah nondeterminisme) ditambah 256 pertanyaan pelatihan lanjutan**. Dengan demikian:
  - kalimat "pembaruan informatif yang lebih banyak tidak **mencegah** keruntuhan" (:184) tidak teruji, karena pada saat keruntuhan jumlah pembaruannya belum berbeda;
  - yang benar-benar teruji adalah versi **pemulihan** dari hipotesis "penguncian kebiasaan" di :184: setelah 256 pertanyaan tambahan dengan pembaruan informatif, sitasi tetap tidak kembali. Ini temuan yang sah dan sebaiknya dinyatakan secara eksplisit;
  - kenaikan +8 EM (:218–219) **tidak dapat diatribusikan** pada *dynamic sampling*. Pada satu *run*, kenaikan ini juga tidak dapat dipisahkan dari variabilitas antar-*run* (lihat m4).
  - Ironisnya, hasil ini justru berguna dengan cara lain: keruntuhan R0 s8 **terulang pada *run* kedua** dengan konfigurasi awal yang sama.
- **Perbaikan konkret:**
  - Nyatakan di Subbab 4.6 dan 3.x bahwa pada 256 pertanyaan pertama R0-dyn identik secara algoritmik dengan R0 s8, dan bahwa keruntuhan terjadi sebelum kedua *run* berbeda.
  - Bingkai ulang R0-dyn sebagai "replikasi R0 s8 + pelatihan diperpanjang". Hasilnya adalah: (a) keruntuhan terulang, dan (b) pelatihan tambahan tidak memulihkan sitasi, sementara EM-nya lebih tinggi pada satu *run* ini.
  - Ganti klaim "mencegah" dengan "memulihkan" pada :184, :223, :235, :329, dan "Hal yang tidak didukung" :344.
  - Catat juga bahwa varian ini berbeda dari *dynamic sampling* DAPO, yang mengisi ulang *batch* berisi banyak grup. Di sini ukuran *batch* adalah satu grup.

### M2. Mekanisme "−1 untuk sitasi salah, 0 untuk sitasi benar" disajikan sebagai sebab yang terbukti, padahal data tidak membedakannya

- **Lokasi:**
  - bab4:112 ("bukan eksploitasi … melainkan konsekuensi dari definisi *reward*", "sekitar separuh dari seluruh upaya menyitasi tidak valid")
  - bab4:151 (perhitungan −0,27 vs +0,36)
  - bab4:126 ("karena alasan yang mengikuti definisinya sendiri")
  - bab4:327 ("penyebabnya dapat ditelusuri")
  - bab4:336 ("*Rollout* pelatihan menunjukkan penyebabnya")
  - Kontras dengan bab4:155, yang sudah tepat menyebutnya "interpretasi".
- **Observasi (hitung ulang saya dari `metrics.jsonl` per blok 32 langkah, memakai definisi skrip yang sama):**

| *Run* | Blok | E[*reward* \| mencoba menyitasi] | *Reward* "tanpa sitasi" | Runtuh? |
|---|---|---|---|---|
| R0 s8 | 1–32 | −0,27 | 0,36 | ya |
| R0 s9 | 1–32 | **+0,29** | 0,26 | ya |
| R0 s7 | 1–32 | **−0,05** | 0,30 | **tidak** |
| R0 s7 | 33–64 | 0,31 | 0,45 | **tidak** |
| R1 s8 | 1–32 | **1,10** | **1,37** | **tidak** |

  - Pada R0 s9 blok pertama, mencoba menyitasi justru sedikit *lebih* menguntungkan, padahal *run* ini runtuh.
  - Pada R0 s7 dan bahkan R1 s8 blok pertama, mencoba menyitasi *kurang* menguntungkan, padahal keduanya tidak runtuh.
  - Jadi statistik "rata-rata upaya menyitasi" yang dipakai di :151 tidak memisahkan *run* yang runtuh dari yang tidak.
  - Proporsi sitasi tidak valid pada blok pertama juga bervariasi: s8 90/176 = 51%, s9 48/209 = 23%, s7 62/174 ≈ 36%. Klaim "sekitar separuh" (:112) hanya berlaku untuk s8.
  - Perbandingan yang lebih relevan bagi GRPO adalah selisih *reward* antara rollout valid bersitasi dan tanpa sitasi di dalam grup. Pada R0 selisihnya kecil (0,14), sedangkan pada R1 sekitar 2 (:153). Argumen ini lebih masuk akal, tetapi tetap korelasional dan hanya berdasar dua *seed* R0 dan satu *seed* R1.
- **Mengapa penting:** Ini adalah klaim mekanistik utama di RQ2 dan di Kesimpulan. Penguji yang menghitung ulang dengan R0 s7 atau R1 s8 akan menemukan kontradiksi.
- **Perbaikan konkret:**
  - Turunkan klaim menjadi **hipotesis yang konsisten dengan data**. Contoh: "Kami menduga … Data *rollout* konsisten dengan dugaan ini pada R0 s8, tetapi statistik yang sama juga muncul pada R0 s7 dan R1 s8 yang tidak runtuh".
  - Ganti kalimat :112, :126, :327, dan :336.
  - Hapus "sekitar separuh", atau tulis "51% pada s8, 23% pada s9".
  - Pindahkan inti argumen ke selisih *reward* di dalam grup (valid bersitasi vs tanpa sitasi: 0,14 vs ~2), dan nyatakan bahwa *seed* 7 belum terjelaskan.
  - Opsional, tanpa GPU: tambahkan baris R0 s7 ke Tabel 4.4 sebagai kontras.

### M3. Label Tabel 4.4 ("bersitasi / tanpa sitasi") tidak sesuai dengan cara kategorinya dihitung

- **Lokasi:** Tabel 4.4 (bab4:135–149), bab4:151, bab4:153, dan bab3:121–130.
- **Observasi:**
  - `scripts/peer_review_reanalysis.py:154–156` membagi rollout valid berdasarkan `citation_precision > 0`. Komentar di skrip menyebut "*rollouts log no citation list*".
  - Jadi kolom "Valid, tanpa sitasi" sebenarnya berarti "**valid tanpa satu pun sitasi emas**". Kategori ini mencakup rollout yang menyitasi kalimat yang sudah diambil tetapi bukan kalimat emas.
  - Pada *holdout*, kategori campuran ini tidak kecil: 35/170 episode selesai B0, 62/278 R0 s7, dan 35/287 R1 s8 menyitasi tetapi presisinya 0 (hitung ulang saya dari `episodes.jsonl`).
  - Akibatnya, "+0,36 untuk tidak menyitasi" (:151) dan "1,3 untuk yang tanpa sitasi" (:153) mencampur dua perilaku. Klaim "jawaban bersitasi lebih sering benar" juga sebenarnya berarti "jawaban dengan ≥1 sitasi emas lebih sering benar". Hubungan ini terkacaukan (*confounded*): menemukan kalimat emas dan menjawab benar sama-sama dipengaruhi oleh keberhasilan pencarian.
  - Kolom tabel juga tidak berjumlah 256, karena kegagalan protokol dan anggaran dihilangkan. Contohnya, R0 s9 langkah 225–256 terdiri atas 217 valid, 38 *budget*, dan 1 protokol.
- **Mengapa penting:** Angka −0,27 vs +0,36 adalah bukti utama mekanisme. Jika kategorinya salah label, bukti tersebut tidak mengukur "menyitasi vs tidak menyitasi".
- **Perbaikan konkret:**
  - Ganti header menjadi "Valid, ≥1 sitasi emas" dan "Valid, tanpa sitasi emas (termasuk tanpa sitasi)".
  - Tulis ulang :151 dan :153 dengan istilah tersebut.
  - Tambahkan kolom "gagal protokol/anggaran" agar baris berjumlah 256, atau jelaskan hal ini di *caption*.
  - Catat batasannya: log *rollout* tidak menyimpan daftar sitasi. Di bagian lain, "tidak menyitasi" sah untuk *holdout* (tingkat sitasi 0%), tetapi untuk pelatihan hanya disimpulkan dari presisi 0 dan tidak adanya kegagalan bukti.

### M4. RQ2 "Akurasinya sama": non-signifikansi diperlakukan sebagai kesetaraan

- **Lokasi:**
  - bab4:100 (judul paragraf "**Akurasinya sama.**")
  - bab4:126 ("menyamai akurasi")
  - bab4:108 ("presisi setara B0", tanpa SK)
  - Bandingkan dengan bab4:198, yang sudah benar ("tidak konklusif, bukan bukti … setara") tetapi hanya diterapkan pada RQ3.
- **Observasi:**
  - R1 − R0 ΔEM = +1,0 dengan SK *seed* [−3,6; +5,4] (output.md:66, cocok).
  - Selang ini tidak menyingkirkan selisih sekitar 4–5 poin ke salah satu arah, dan tidak ada margin non-inferioritas yang ditetapkan di awal.
  - RQ2 di bab3:29 berbunyi "tanpa kehilangan akurasi". Pertanyaan ini secara implisit adalah pertanyaan non-inferioritas.
- **Mengapa penting:** Ini adalah standar ganda yang mudah ditangkap penguji: RQ3 diperlakukan ketat, RQ2 tidak.
- **Perbaikan konkret:**
  - Ganti "Akurasinya sama" dengan "**Tidak ada perbedaan akurasi yang terdeteksi**".
  - Tambahkan kalimat: "Batas bawah SK menunjukkan bahwa kehilangan akurasi lebih dari ~3,6 poin EM tidak didukung data. Kehilangan yang lebih kecil tidak dapat disingkirkan".
  - Terapkan hal yang sama di :126 dan :335. Pada :108, ganti "setara" dengan "serupa (0,621 vs 0,633; tanpa uji)".
  - Judul Gambar 4.3 ("*Same accuracy, different grounding*", sudah dicek) membuat klaim kesetaraan yang sama. Ganti misalnya dengan "Akurasi serupa, *grounding* berbeda".

### M5. "Kenaikan penalaran hanya 4–6 poin" membandingkan himpunan pertanyaan yang berbeda

- **Lokasi:** bab4:90–92 dan bab4:334 (poin 1).
- **Observasi:**
  - "EM selesai" dihitung atas 170 episode untuk B0 dan ~280–300 episode untuk *run* terlatih, sehingga himpunan pertanyaannya berbeda. Kalimat "pada pertanyaan yang sebelumnya sudah dapat diselesaikan model" (:92) dengan demikian tidak tepat.
  - Data himpunan bersama sudah tersedia (output.md:33–53). Pada 136 pertanyaan yang diselesaikan semua *run*:
    - EM B0 = 37,5;
    - EM R1 = 49,3 / 42,6 / 44,9 (rata-rata 45,6, yaitu **+8,1**);
    - rentang seluruh *run* terlatih 39,7–49,3 (+2,2 hingga +11,8).
  - Istilah "kemampuan penalaran" juga merupakan klaim kausal tentang konstruk yang tidak diukur.
- **Mengapa penting:** Ini adalah klaim utama RQ1 dan muncul di Kesimpulan. Arah kesimpulannya tetap benar (sebagian besar kenaikan berasal dari protokol), tetapi besarnya kurang tepat. Selain itu, analisis himpunan bersama sudah dipakai untuk presisi di :114 tetapi tidak dipakai untuk EM.
- **Perbaikan konkret:**
  - Laporkan EM pada 136 pertanyaan bersama di samping "EM selesai".
  - Ganti "4–6 poin" dengan "sekitar 5–8 poin (EM selesai: +4,4 hingga +5,9; himpunan bersama: rata-rata R1 +8,1)".
  - Ganti "kemampuan penalaran" dengan "akurasi jawaban pada episode yang selesai".

### M6. Klaim faktual yang bertentangan dengan Tabel 4.1: cakupan R1 "di bawah B0 pada dua dari tiga *seed*"

- **Lokasi:** bab4:186.
- **Observasi:**
  - Menurut Tabel 4.1 (bab4:42–44, cocok dengan `logs/2026-10-11-fixes-batch.md`), cakupan R1 adalah s7 = 0,473, s8 = 0,513, dan s9 = 0,333, sedangkan B0 = 0,472.
  - Artinya, hanya **satu** dari tiga *seed* yang berada di bawah B0. Seed 7 praktis sama dengan B0, dan seed 8 berada di atasnya.
- **Mengapa penting:** Kontradiksi ini bisa langsung dicek penguji pada halaman yang sama.
- **Perbaikan konkret:** Tulis "berada di bawah B0 hanya pada *seed* 9. Rata-ratanya (0,440) di bawah B0 karena *seed* 9". Pertimbangkan juga menambahkan perbandingan pada himpunan bersama: B0 0,463 vs rata-rata R1 0,460 (output.md:37, 41–43).

### M7. Klaim RQ3 (R3 "terbaik pada Joint-EM") lebih kuat daripada status eksploratifnya

- **Lokasi:** bab4:206–212 (cukup berhati-hati), tetapi kemudian bab4:209, bab4:234, dan bab4:337 (poin 4, di bawah "Hal yang **didukung**").
- **Observasi:**
  - R3 − R1 Joint-EM = +6,3 [+0,2; +13,1] dan P-gEM = −8,0 [−16,1; −0,2]. Kedua selang ini nyaris menyentuh nol.
  - Analisis RQ3 dengan selang *seed* memuat 3 perbandingan (R2/R3/R4 − R1) × 3 metrik = 9 selang tanpa koreksi, dan R3 dirancang *post hoc*. Dengan α = 0,05, kira-kira 0,45 selang diperkirakan "signifikan" secara kebetulan, dan dua yang muncul berada tepat di batas.
  - :212 sudah menyatakan "tidak ada peringkat yang kokoh", tetapi :234 dan :337 kembali menyajikannya sebagai temuan yang didukung.
- **Mengapa penting:** Ini tidak konsisten dengan aturan yang ditetapkan sendiri di bab3:228 dan bab4:198.
- **Perbaikan konkret:**
  - Pindahkan poin 4 di :337 menjadi "temuan eksploratif yang mengarah ke …".
  - Pada :234, tambahkan "(eksploratif; kedua selang nyaris mencakup nol)".
  - Hindari kata "terbaik" tanpa kualifikasi.

### M8. Gambar 4.5 bertentangan dengan teks (catatan [FIGURE UPDATE NEEDED] masih terbuka)

- **Lokasi:** bab4:82–86.
- **Observasi:**
  - Gambar menampilkan selang yang hanya me-*resample* pertanyaan (cocok dengan `figures_data.json`, `bootstrap`). Di gambar itu, R0 − B0 ΔP-gEM [−8,4; −1,2] **tidak mencakup nol**, sedangkan teks :94 menyatakan selisih ini "tidak signifikan" berdasarkan selang *seed*.
  - Interval R1 − R0 di gambar ([+20,9; +28,1]) juga jauh lebih sempit daripada selang utama ([+8,8; +37,3]).
  - Judul gambar, label sumbu ("ΔGrounded EM"), dan anotasinya masih berbahasa Inggris.
- **Mengapa penting:** Ini adalah satu-satunya gambar yang menampilkan SK, dan gambar itu memvisualkan selang yang menurut teks sendiri tidak boleh dipakai untuk inferensi.
- **Perbaikan konkret:**
  - Laksanakan catatan :86, yaitu buat ulang gambar dengan selang *seed* sebagai batang utama.
  - Terjemahkan seluruh label ke Bahasa Indonesia.
  - Hapus catatan berbahasa Inggris dari naskah sebelum dikumpulkan.

---

## Temuan MINOR

### m1. SK R2 s8 − R1 s8 tidak cocok dengan log

- **Lokasi:** bab4:200.
- **Masalah:** Teks menulis [+1,3; +11,0], sedangkan `logs/2026-10-10-r0-r1-r2-batch.md:50` menulis **[+1,3, +10,7]**. Hitung ulang saya dengan RNG lain menghasilkan [+1,0; +10,7].
- **Perbaikan:** Gunakan +10,7 atau sebutkan sumber hitung ulangnya.

### m2. SK per *run* terhadap B0 tidak punya sumber tersimpan

- **Lokasi:** bab4:88.
- **Masalah:** SK R2 s9 − B0 [+10,7; +21,3] dan R0-dyn s8 − B0 [+21,0; +31,7] tidak ditemukan di `logs/` maupun `figures_data.json`. Angkanya masuk akal (hitung ulang saya: [+11,0; +21,3] dan [+21,3; +31,7]), tetapi tidak dapat ditelusuri. R2 s8 − B0 [+21,0; +31,7] cocok dengan log batch.
- **Perbaikan:** Tambahkan SK per *run* ke `peer_review_reanalysis.py`/`output.md`, atau hapus SK dan pertahankan hanya selisihnya.

### m3. Bootstrap tiga *seed* punya keterbatasan yang tidak disebut

- **Lokasi:** bab4:67, bab4:94, bab4:349.
- **Masalah:**
  - Me-*resample* 3 *seed* dengan pengembalian hanya menghasilkan 10 multiset per metode. Hasilnya diskret dan cenderung meremehkan variansi antar-*seed* (faktor (n−1)/n).
  - Batas bawah R1 − R0 (+8,8) praktis sama dengan kasus "semua *seed* R0 = s7".
  - Kata "signifikan" untuk selang yang batasnya sangat dekat nol (:94 R2 − B0 Joint-EM [+0,7; +7,2]) perlu dibaca dengan hati-hati.
- **Perbaikan:**
  - Tambahkan satu kalimat di "Ancaman terhadap validitas" tentang keterbatasan ini.
  - Untuk R1 − R0, tambahkan argumen yang lebih transparan: **setiap *seed* R1 (28,0–37,7) berada di atas *seed* R0 terbaik (24,7)**, sehingga 9 dari 9 pasangan *seed* bernilai positif. Selisih terkecilnya adalah 3,3 poin (R1 s8 vs R0 s7).

### m4. Kenaikan R0-dyn +8,0 EM hanya didukung SK pertanyaan untuk satu *run*

- **Lokasi:** bab4:218 dan bab4:329 ("R0-dyn lebih akurat daripada R0 s8").
- **Masalah:** SK [+3,3; +12,7] tidak memperhitungkan variabilitas antar-*run*. Sebagai pembanding, rentang antar-*seed* R2 adalah 10,4 poin, dan nondeterminisme membuat R0-dyn dan R0 s8 sudah berbeda sejak awal (lihat M1).
- **Perbaikan:** Tulis "EM-nya 8 poin lebih tinggi pada satu *run*. Selang ini hanya mencerminkan pengambilan sampel pertanyaan, bukan variabilitas antar-*run*".

### m5. Proporsi keruntuhan hanya diberi selang untuk R0

- **Lokasi:** bab4:110 dan bab4:336.
- **Masalah:** R0 runtuh pada 2/3 *seed* (CP [0,09; 0,99]). R1 runtuh pada 0/3 *seed*, dan CP-nya tidak dilaporkan. Hitungan saya sendiri (CP eksak, 0/3): [0; 0,71]. Kedua selang tumpang-tindih lebar, sehingga "VEGR terhindar dari hilangnya sitasi pada ketiga *seed*" (:126) belum bisa diartikan sebagai laju keruntuhan yang lebih rendah.
- **Perbaikan:** Laporkan CP untuk 0/3 juga dan tambahkan satu kalimat tentang hal ini.

### m6. "Pemeriksaan independen pada Subbab 4.4" tidak independen

- **Lokasi:** bab4:351.
- **Masalah:** Pemeriksaan di Subbab 4.4 (himpunan bersama, :114) tetap memakai label emas yang sama, sehingga tidak mengurangi masalah Goodhart.
- **Perbaikan:** Hapus rujukan itu, atau sebut secara spesifik bahwa yang mengurangi masalah hanyalah Sup-EM/Joint-EM, yang juga memakai label yang sama.

### m7. Pola "satu sitasi" pada *seed* 9 muncul di dua metode, tetapi hanya dibahas untuk R1

- **Lokasi:** bab4:194.
- **Masalah:** Hitung ulang saya menunjukkan R4 s9 juga hanya menyitasi 1,04 kalimat per episode, dengan Sup-EM 3,0 (output.md:20) dan P-gEM tertinggi (39,0). Pola yang sama pada dua metode di *seed* yang sama mengarah ke efek sampel pertanyaan pelatihan atau *seed*, bukan efek metode.
- **Perbaikan:** Tambahkan satu kalimat tentang ini. Ini juga memperkuat peringatan bahwa P-gEM harus dibaca bersama Sup-EM.

### m8. Penjelasan bias posisi tidak memuat *base rate*

- **Lokasi:** bab4:186–192 dan Gambar 4.7.
- **Masalah:**
  - "93% sitasi ke tiga kalimat pertama" tidak bisa ditafsirkan tanpa distribusi posisi **kalimat emas** HotpotQA. Kalimat emas memang cenderung berada di awal paragraf.
  - Narasi "menyitasi satu kalimat yang aman" juga diilustrasikan dengan R1 s8, yang justru menyitasi 2 kalimat pada 182/287 episode (hitung ulang saya cocok: 97/182/8).
- **Perbaikan:** Tambahkan distribusi posisi kalimat emas sebagai pembanding, atau lunakkan klaimnya. Ilustrasi "satu kalimat" lebih cocok memakai R1 s9.

### m9. Klaim "kegagalan bukti tetap paling umum pada R1–R4" tidak berlaku per *run*

- **Lokasi:** bab4:319.
- **Masalah:** Pada R2 s8, kegagalan bukti 3 vs protokol 7, dan R3 s8 seri (8 vs 8) (Tabel 4.6). Klaim ini hanya benar secara agregat (R1 20 vs 13, R2 23 vs 18, R3 25 vs 16, R4 17 vs 9).
- **Perbaikan:** Tambahkan "secara agregat per metode".

### m10. Validasi bukan bukti independen

- **Lokasi:** bab4:96.
- **Masalah:** Menurut `logs/2026-10-10-grpo-run2.md`, *split* validasi adalah "*the split for choosing settings*". *Split* itu dipakai untuk memilih pengaturan *run* 2, dan bootstrap-nya memakai 5.000 *resample*, bukan 10.000.
- **Perbaikan:** Sebutkan kedua hal tersebut.

### m11. Nomor gambar tidak berurutan

- **Lokasi:** seluruh bab.
- **Masalah:** Urutan kemunculan gambar adalah 4.5 → 4.3 → 4.1 → 4.2 → 4.7 → 4.4 → 4.6, padahal pedoman TA umumnya mensyaratkan penomoran sesuai urutan kemunculan.
- **Perbaikan:** Nomori ulang.

### m12. Label gambar masih berbahasa Inggris

- **Lokasi:** Gambar 4.1, 4.2, 4.3, 4.4, dan 4.5 (sudah dicek).
- **Masalah:** Judul, sumbu, dan legenda masih berbahasa Inggris, misalnya "Zero-variance GRPO steps skipped", "Citation precision during training", "Grounded EM", dan "Per-seed holdout scores".
- **Perbaikan:** Terjemahkan sekalian saat menangani catatan *figure update*.

### m13. Status catatan [FIGURE UPDATE NEEDED]

- **:86 (Gambar 4.5):** valid dan wajib. Lihat M8.
- **:124 (Gambar 4.3, legenda "R4 Dr. GRPO"):** sudah dicek, dan legenda memang bertuliskan "R4 Dr. GRPO". Catatan ini valid, karena teks (:214) sendiri menyatakan R4 bukan Dr. GRPO. Judul gambarnya juga perlu diganti (lihat M4).
- **:176 (Gambar 4.1):** gambar yang dirender **tidak memiliki legenda**, dan labelnya hanya "R4 s7/s8/s9". Tidak ada yang perlu diubah selain menerjemahkan judul, jadi catatan ini dapat dihapus.
- **:229 (Gambar 4.4):** valid, karena panel kanan berjudul "Grounded EM".
- **Perbaikan:** Hapus keempat catatan berbahasa Inggris dari naskah final.

---

## NIT

1. **Terminologi.** Bab 4 hanya memakai "*outcome-only*" dan tidak pernah memakai "berbasis jawaban saja", sedangkan Bab 1, 3, dan 5 memakai "berbasis jawaban saja" (5, 2, dan 6 kali; hasil `grep`). Pilih satu istilah dan definisikan padanannya sekali.
   - "*Grounded EM*" di gambar, log, dan skrip berbeda dengan "*precision-grounded* EM" di teks.
   - "Keruntuhan", "jalan pintas", "hilangnya sitasi", dan "berhenti memberi sitasi" dipakai bergantian. Di :108, perilaku s7 disebut "bukan jalan pintas", di :112 hilangnya sitasi disebut "bukan eksploitasi", tetapi :223 dan :235 menyebutnya "jalan pintas". Samakan istilahnya.
2. **:157.** "*reward*-nya biner" kurang tepat, karena R0 bernilai {−1, 0, 1}. Tulis "*reward*-nya hanya mengambil tiga nilai".
3. **Tabel 4.2 *caption* (:56).** Joint-F1 bukan "persentase dari 300 pertanyaan".
4. **Tabel 4.3.** "Baris di bawah garis" tidak terlihat di Markdown. Pastikan garisnya ada di versi Word/PDF. Baris R0-dyn juga menaruh nilai di kolom "dengan *seed*". Ganti dengan "–" dan pindahkan +8,0 ke kolom "hanya pertanyaan".
5. **:151.** "Sitasi hilang pada langkah ke-97" sebenarnya resolusi blok. Tulis "sejak blok langkah 97–128".
6. **:114.** "Tingkat sitasi 99–100%": B0 98,8% dan R2 s9 98,6% (output.md:5, 14). Tulis "≥ 98,6%".
7. **:273.** "*Trace* diambil apa adanya", tetapi *Trace* 3 dan 4 menghilangkan aksi SEARCH. Tambahkan "(dipotong)".
8. **SK/"SK yang hanya me-*resample* pertanyaan".** Pastikan singkatan SK didefinisikan di Bab 3 atau di awal Bab 4.

---

## Pemeriksaan angka (spot-check)

**Ringkasan:** 48 baris pemeriksaan (sebagian baris mencakup banyak sel, misalnya seluruh Tabel 4.1). Hasilnya:

- 44 baris cocok;
- 1 baris tidak cocok dengan log (#36, m1);
- 1 klaim salah terhadap Tabel 4.1 (#32, M6);
- 1 klaim hanya benar untuk satu *seed* (#18, M2);
- 1 baris tidak dapat ditelusuri (#10, m2). SK R0-dyn s8 − B0 di :88 juga tidak dapat ditelusuri, tetapi tidak saya jadikan baris tersendiri.

Arti kolom "Status":

- **S** = cocok dengan sumber tersimpan (log, `output.md`, atau `figures_data.json`).
- **H** = cocok dengan hitung ulang saya dari `episodes.jsonl`/`metrics.jsonl`.

| # | Lokasi | Angka di teks | Sumber | Status |
|---|---|---|---|---|
| 1 | :28, Tabel 4.6 | B0 170 selesai; 85/37/8 | grpo-run2 log; figures_data `failures` | S, H |
| 2 | :30 | B0 EM selesai 39,4; presisi 0,633; *recall* 0,829 | grpo-run2 log | S |
| 3 | Tabel 4.1 | 17 baris EM/P-gEM/selesai/cakupan/presisi/*recall*/pencarian/dup | fixes-batch log tabel per *run* | S (semua cocok) |
| 4 | Tabel 4.1 | Kolom Pembaruan = 256 − dilewati (Tabel 4.5) | fixes log; figures_data `fig4_1` | S |
| 5 | Tabel 4.2 | Seluruh rata-rata ± sd (EM, EM selesai, P-gEM, Lenient, Sup-EM, Joint) | output.md:27–31 | S |
| 6 | Tabel 4.2 | Presisi R2 0,718 ± 0,026 (output.md: 0,721 "*when cited*") | fixes log (0,718); dihitung dari Tabel 4.1 | S, H (definisi berbeda, tidak salah) |
| 7 | Tabel 4.2 | Presisi R1 sd 0,015; R0 0,207 ± 0,358 | dihitung dari Tabel 4.1 | H |
| 8 | Tabel 4.3 | 9 baris × 3 kolom SK *seed* | output.md:61–69 | S |
| 9 | Tabel 4.3 | Kolom SK hanya pertanyaan (EM dan P-gEM, 10 baris) | figures_data `bootstrap` | S |
| 10 | :88 | R2 s9 +16,0 [+10,7; +21,3] | tidak ada sumber tersimpan; hitung ulang [+11,0; +21,3] | **tak tertelusur** |
| 11 | :88 | R2 s8 +26,3 [+21,0; +31,7] | r0-r1-r2-batch log | S |
| 12 | :90 | maks 22 episode gagal (R0 s7) | Tabel 4.1/4.6 | S |
| 13 | :92 | Invalid 46→29→13→13→7%; *reward* 0,97→2,81 | grpo-run2 log | S, H |
| 14 | :96 | Validasi 57,0 vs 29,3, +27,7 [+21,7; +33,7] | grpo-run2 log | S |
| 15 | :106 | 122 dan 135 jawaban benar tanpa sitasi | fixes log | S, H |
| 16 | :108 | 229/278 dup; R1 7–27 | fixes log | S |
| 17 | :110 | CP [0,09; 0,99] | output.md:71 | S |
| 18 | :112 | "sekitar separuh upaya menyitasi tidak valid" | s8 51%, s9 23%, s7 36% | **hanya benar untuk s8** (M2) |
| 19 | :114 | Himpunan bersama 136; 0,627 / 0,646 / 0,750–0,790 | output.md:33–43 | S |
| 20 | :116 | R1 EM 42,7–45,3; presisi 0,726–0,754 | Tabel 4.1 | S |
| 21 | :118 | Pita 38–49%; presisi R1–R4 0,63–0,76 | figures_data `points` | S |
| 22 | :128 | P-gEM 32,7 vs EM 43,6 (~11 poin) | output.md | S |
| 23 | Tabel 4.4 | 33 sel R0 s8/s9, R1 s8 | output.md:79–112 | S, H (label: lihat M3) |
| 24 | :151 | (86×0,50 − 90)/176 ≈ −0,27 | aritmetika | H |
| 25 | :153 | 85 → 214 rollout valid bersitasi | output.md:105, 112 | S |
| 26 | Tabel 4.5 | 16 nilai langkah dilewati | figures_data `fig4_1` | S |
| 27 | :170 | "8–20 langkah" | Tabel 4.5 | S |
| 28 | :178 | Presisi pelatihan s8: 0,568/0,499; 0,550/0,520; 0,126; 0,000; R1 0,62–0,76 | figures_data `fig4_2` | S, H |
| 29 | :178 | R0 s9 0,714/0,410/0,192; R0 s7 0,55–0,73 | `metrics.jsonl` | H |
| 30 | :184 | R0-dyn 0,550/0,596/0,250; 0,004 | `metrics.jsonl` | H |
| 31 | :186 | R1 s8: 97/182/8 kalimat; posisi 233/161/57 | figures_data `cited_index`; `episodes.jsonl` | S, H |
| 32 | :186 | Cakupan R1 "di bawah B0 pada dua dari tiga *seed*" | Tabel 4.1: hanya s9 | **salah** (M6) |
| 33 | Gbr 4.7 | 485 sitasi; 451 (93%) | figures_data | S, H |
| 34 | :194 | R1 s9 1,03; s7 1,49; s8 1,69 sitasi/episode; Sup-EM 0,7 | `episodes.jsonl`; output.md:11 | H, S |
| 35 | :198 | sd EM maks 5,4 | output.md:29 | S |
| 36 | :200 | R2 s8 − R1 s8 +6,0 [+1,3; **+11,0**] | log: [+1,3, **+10,7**] | **tidak cocok** (m1) |
| 37 | :200 | R2 s7 −1,7 vs R1; s9 38,3 vs 45,3 | Tabel 4.1 | S |
| 38 | :204 | R3 1,83–2,06 vs R1 1,03–1,69 sitasi/episode | `episodes.jsonl` | H |
| 39 | :209–210 | R3 Joint-EM 13,4, Joint-F1 38,0 | output.md:30 | S |
| 40 | :214 | sd EM R4 4,0 vs R1 1,5 | output.md | S |
| 41 | :218–220 | R0-dyn 48,7%; +8,0 [+3,3; +12,7]; 146 benar tanpa sitasi; 250 pembaruan | figures_data; fixes log | S |
| 42 | :267 | Kegagalan bukti 3–14 | Tabel 4.6 | S |
| 43 | :271 | '&' 2–5 per *run*; ≤ 1,7 poin | output.md:116–134 | S |
| 44 | :275–321 | Kelima *trace* (aksi, EM, presisi/cakupan, pesan galat) | `episodes.jsonl` | H (semua cocok) |
| 45 | :24 | *Reward* masih bergeser beberapa persepuluh poin di blok akhir | `metrics.jsonl` (mis. R1 s9 3,12 → 2,53) | H |
| 46 | :21–22 | ≈ 2,05 jam/*run*; total ≈ $49 | batch log 10,2 jam/5 *run*; 6,2 + 16 + 26,5 = 48,7 | S |
| 47 | :355 | PyTorch 2.14.1 → 2.11.0 | r0-r1-r2-batch log:20 | S |
| 48 | :349 | Lebar SK R2 − R1 ≈ 13 poin | 5,8 − (−7,1) = 12,9 | S |

---

## Kemungkinan pertanyaan sidang (ruang lingkup Bab 4)

1. "Apa sebenarnya beda R0-dyn dengan R0 s8 pada 96 langkah pertama, saat keruntuhan terjadi?" Lihat M1. Siapkan jawaban jujur bahwa keduanya identik secara algoritmik pada tahap itu.
2. "Anda bilang R0 berhenti menyitasi karena −1 untuk sitasi salah. Kalau begitu, mengapa R0 seed 7 tidak runtuh, dan mengapa di R1 blok pertama upaya menyitasi juga rata-rata lebih rugi?" Lihat M2/M3.
3. "SK R1 − R0 untuk EM adalah [−3,6; +5,4]. Atas dasar apa Anda menyatakan VEGR tidak mengorbankan akurasi? Berapa margin yang Anda anggap dapat diterima?" Lihat M4.
4. "*Bootstrap* yang me-*resample* tiga *seed*: berapa banyak kombinasi yang mungkin, dan apakah selang itu dapat dipercaya? Apakah ada cara yang lebih sederhana untuk menunjukkan bahwa R1 > R0 pada *grounding*?" Jawaban yang disarankan: 9/9 pasangan *seed* positif (m3).
5. "Metrik *grounding* Anda memakai label emas yang sama dengan *reward*. Bagaimana Anda tahu agen benar-benar lebih ter-*grounding* dan bukan sekadar mengoptimalkan metrik?" (Goodhart, :351, m6).

---

## Catatan keterbatasan review

- Gambar 4.1–4.5 sudah dilihat. Gambar 4.6 dan 4.7 belum dibuka secara visual.
- Skrip pada skill `peer-review` tidak dijalankan, yaitu *intake validator*, *claim-matrix*, dan *lint*. Otorisasi berasal dari pemilik naskah sendiri, dan pemrosesan dilakukan secara lokal.
- Tidak ada GPU yang dijalankan.
- Hitung ulang bootstrap saya memakai RNG Python yang berbeda dari skrip asli. Selisih ±0,3 poin pada batas SK karenanya wajar, dan tidak saya laporkan sebagai ketidakcocokan kecuali ada sumber tersimpan yang berbeda (m1).
