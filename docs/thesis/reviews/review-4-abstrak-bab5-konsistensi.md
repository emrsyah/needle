# Review 4: Abstrak, Bab 5, dan Konsistensi Lintas Bab

**Status:** draf kerja (*working draft*) dari reviewer berbantuan AI. Penulis tetap harus memeriksa setiap lokasi, angka, dan usulan kalimat sebelum dipakai.
**Cakupan:** `chapters-id/abstrak.md`, `chapters-id/bab5-kesimpulan-dan-saran.md`, dan konsistensi dengan Bab 1, 3, dan 4 (Bab 2 hanya dicek pada Subbab 2.9). Bab 2 dan Lampiran A tidak di-*review* secara isi.
**Tolok ukur:** Tugas Akhir S1. Kebaruan bukan syarat. Yang dinilai adalah apakah klaim sesuai dengan bukti dan apakah naskah konsisten.
**Catatan lokasi:** `abstrak.md` hanya berisi satu paragraf panjang (baris 3), jadi lokasinya ditulis sebagai `abstrak.md:3` disertai kutipan frasa. Angka bertanda "(dihitung reviewer dari Tabel X)" adalah hasil hitungan reviewer dari tabel yang ada di naskah, bukan angka yang dilaporkan naskah.

---

## 1. Ringkasan

Secara keseluruhan, naskah sangat konsisten secara numerik. Semua angka di abstrak dan Bab 5 yang saya cek ada di Bab 4 dengan nilai yang sama (lihat Bagian 3). Rata-rata pada Tabel 4.2, selisih pada Tabel 4.3, dan jumlah *update* pada Tabel 4.1 sesuai dengan Tabel 4.5. Kelemahan utama ada pada **kekuatan klaim**, bukan angka. Di beberapa tempat, abstrak dan Bab 5 menyatakan lebih tegas daripada Bab 4:

1. Kesetaraan akurasi R1–R0 ditulis "sama", padahal yang didukung data adalah "tidak dapat dibedakan".
2. Mekanisme hilangnya sitasi ditulis sebagai penyebab pasti ("karena"), padahal Bab 4 menyebutnya interpretasi.
3. "Mencegah" disimpulkan dari 3 *seed*.
4. Kenaikan 24,4 poin P-gEM dibaca sebagai kemampuan VEGR. Padahal sebagian besar selisih itu datang dari dua *seed* R0 yang bernilai 0, dan metriknya memakai label emas yang sama dengan *reward*.

Ada juga satu kontradiksi di metode, yaitu pernyataan bahwa *holdout* tidak pernah dipakai untuk memilih pengaturan, dan beberapa ketidakkonsistenan istilah.

## 2. Kekuatan

- **Kejujuran statistik di atas rata-rata TA S1.** Ada SK *bootstrap* yang memperhitungkan *seed*, pemisahan tegas antara perbandingan utama dan eksploratif (bab3:228, bab4:198), selang Clopper–Pearson untuk 2/3 *seed* (bab4:110), dan daftar "hal yang tidak didukung" (bab4:339–345).
- **Dekomposisi RQ1 yang jujur.** "EM selesai" memisahkan pembelajaran protokol dari peningkatan kemampuan menjawab (bab4:90–92, bab5:9).
- **Mekanisme R0 bisa ditelusuri.** Tabel 4.4 dan perhitungan nilai harapan (bab4:151) membuat klaim dapat diperiksa.
- **Ketertelusuran angka.** Bab 4 menyebut skrip dan berkas sumbernya (bab4:3).
- **Keterbatasan dan saran sudah konkret.** Ablasi tanpa komponen bukti (bab5:57), menutup celah tanpa sitasi (bab5:58), dan *baseline* berbasis *prompt* (bab5:61) adalah langkah yang tepat.
- **Ruang lingkup jelas.** Ada batasan "hal yang tidak diklaim" (bab1:46), dan istilah "agen pencarian" dijelaskan sebagai pencarian pada kumpulan tertutup (bab1:42, bab3:49).

## 3. Cek Silang Numerik: Abstrak dan Bab 5 terhadap Bab 4

| # | Klaim (lokasi) | Nilai | Sumber Bab 4 | Status |
|---|---|---|---|---|
| 1 | abstrak.md:3 "satu GPU 48 GB" | 48 GB | bab4:20 | Cocok |
| 2 | abstrak.md:3, bab5:7 "10 paragraf" | 10 | bab4:17 | Cocok |
| 3 | abstrak.md:3, bab5:7 "Enam belas *run* (sekitar $49)" | 16, $49 | bab4:22; bab3:277 | Cocok |
| 4 | abstrak.md:3, bab5:7 "lima konfigurasi ... tiga *seed* dan satu *run* diagnostik" | 5×3+1 | bab4:19, bab3:232–240 | Cocok |
| 5 | abstrak.md:3, bab5:7 "300 pertanyaan *holdout*" | 300 | bab4:18 | Cocok |
| 6 | abstrak.md:3, bab5:9 "sekitar 20 poin" | +20,0 s.d. +21,7 | Tabel 4.3, bab4:71–75 | Cocok |
| 7 | abstrak.md:3, bab5:9 B0 EM | 22,3% | bab4:28, bab4:38, bab4:60 | Cocok |
| 8 | abstrak.md:3, bab5:9 EM selesai | 39,4% → 45,2% | bab4:60, 62, 90 | Cocok. Abstrak tidak menyebut bahwa 45,2% adalah angka R1 (lihat m3) |
| 9 | bab5:9 "130 dari 300", "paling banyak 22" | 130; 22 | bab4:247–248, bab4:90 | Cocok |
| 10 | abstrak.md:3, bab5:13 ΔEM R1−R0 | +1,0 [−3,6; +5,4] | bab4:76 | Cocok |
| 11 | bab5:13 "43,6% vs. 42,6%" | R1 = 43,6; R0 = 42,6 | bab4:61–62 | Angka cocok, **urutannya terbalik** dengan subjek kalimat (lihat m1) |
| 12 | abstrak.md:3, bab5:14 ΔP-gEM R1−R0 | +24,4 [+8,8; +37,3] | bab4:76, bab4:102 | Cocok |
| 13 | abstrak.md:3, bab5:15 "2 dari 3 *seed*" | 2/3 | bab4:110 | Cocok |
| 14 | bab5:41 Clopper–Pearson | [0,09; 0,99] | bab4:110 | Cocok (dicek ulang reviewer: [0,094; 0,992]) |
| 15 | bab5:15 "sekitar separuh ... tidak valid" | ≈ 51% | bab4:139 (R0 s8: 90/176) | **Hanya berlaku untuk s8**. Pada R0 s9, 48/209 ≈ 23% (dihitung reviewer dari Tabel 4.4, bab4:143). Lihat M3 |
| 16 | abstrak.md:3, bab5:17 cakupan R1 vs B0 | 0,440 vs 0,472 | bab4:60, 62, 129 | Cocok |
| 17 | bab5:22 cakupan R3 | 0,44 → 0,57 | bab4:62, 64 | Cocok (dibulatkan) |
| 18 | abstrak.md:3, bab5:22 ΔJoint-EM R3−R1 | +6,3 [+0,2; +13,1] | bab4:78, 209 | Cocok |
| 19 | bab5:22 ΔP-gEM R3−R1 | −8,0 [−16,1; −0,2] | bab4:78, 206 | Cocok |
| 20 | bab5:21 "+6 poin EM ... R2" | +6,0 | bab4:200 | Cocok |
| 21 | bab5:23 R0-dyn "+8,0 poin EM" | +8,0 | bab4:80, 218 | Cocok |
| 22 | bab5:32 "$2,25 per *run*" | $2,25 | bab4:21 | Cocok |
| 23 | bab5:32 "15 dari 16 *run*" | 15/16 | bab3:287 | Cocok |
| 24 | bab5:38 "83% dokumen emas" | 0,829 | bab4:38 | Cocok |
| 25 | bab5:40 "256 pertanyaan, satu *epoch*" | 256 | bab4:15 | Cocok |
| 26 | bab5:42 "250 dari target 256 *update*" | 250 | bab4:54, 219 | Cocok |
| 27 | bab5:51 "$49 ... Tabel 3.2" | $49 | bab3:277 | Cocok |
| 28 | bab5:49 "sekitar seperempat jawaban benarnya ... agen VEGR terbaik" | – | **Tidak ada di Bab 4** | Tidak tertelusur (lihat m6) |
| 29 | bab5:16, bab1:56 dsb. "enam komponen" | 6 | bab3:9 | Cocok |
| 30a | abstrak.md:3, bab5:15 "sitasi yang tidak valid bernilai −1" | −1 | bab3:117, bab3:121, bab4:112 | Cocok |
| 30b | bab5:23 "jumlah pertanyaan pelatihan yang dua kali lipat" | 512 vs 256 | bab4:219 | Cocok |
| 30 | bab3:10, bab4:104, bab4:350 "sekitar 240 vs sekitar 130" | R1 = 245, R0 = 136 rata-rata (dihitung reviewer dari Tabel 4.1) | Tabel 4.1, 4.5 | Cocok. 256 − langkah dilewati = kolom *update* untuk ke-15 *run* |

Cek tambahan: rata-rata Tabel 4.2 (EM, P-gEM, cakupan untuk R0, R1, R2, R3, R4) dan selisih titik pada Tabel 4.3 semuanya sesuai dengan Tabel 4.1, setelah dihitung ulang dari nilai per *seed*.

## 4. Temuan

### MAJOR

**M1. Kesetaraan akurasi dinyatakan lebih kuat daripada yang didukung data ("akurasi yang sama", RQ2 "tanpa kehilangan akurasi → Ya")**

- **Lokasi:** abstrak.md:3 ("mencapai akurasi yang sama (+1,0 EM, SK 95% [−3,6; +5,4])"); bab5:11 ("Ya, dalam pengaturan penelitian ini"); bab5:13 ("mencapai akurasi yang sama"); bab1:56 ("mencapai akurasi yang sama"); bab4:100 (judul paragraf "**Akurasinya sama.**"); bab4:126 ("menyamai akurasi").
- **Observasi:** SK [−3,6; +5,4] masih cocok dengan R1 yang lebih buruk sampai 3,6 poin. Tidak ada margin non-inferioritas yang ditetapkan sebelumnya. Naskah sendiri memakai standar yang lebih ketat di tempat lain: bab4:100 ("Tidak ada perbedaan akurasi yang dapat diklaim") dan bab4:335 ("secara statistik tidak dapat dibedakan"). Aturan di bab3:228 dan bab4:198 (hasil tidak signifikan berarti "tidak konklusif, bukan bukti bahwa kedua metode setara") secara formal ditulis untuk perbandingan eksploratif, tetapi logikanya berlaku sama untuk R1 − R0.
- **Mengapa penting:** Frasa "tanpa kehilangan akurasi" dalam RQ2 adalah klaim non-inferioritas. Penguji statistik akan bertanya berapa margin yang dipakai. Menjawab "Ya" tanpa margin tidak sejalan dengan logika yang dipakai naskah sendiri untuk hasil yang tidak signifikan.
- **Perbaikan:** Ganti "akurasi yang sama" dengan "akurasi yang tidak dapat dibedakan secara statistik". Di bab5:11, ubah jawaban RQ2 menjadi, misalnya: "*Sebagian ya, dalam pengaturan penelitian ini. Tidak terdeteksi penurunan akurasi (+1,0 EM, SK [−3,6; +5,4]); data masih konsisten dengan penurunan hingga sekitar 3,6 poin, sehingga 'tanpa kehilangan akurasi' hanya dapat dinyatakan dalam batas presisi tersebut.*" Ganti juga judul paragraf bab4:100 menjadi "**Akurasinya tidak dapat dibedakan.**"

**M2. Kenaikan 24,4 poin P-gEM disajikan sebagai efek VEGR, padahal sebagian besar berasal dari dua *seed* R0 yang bernilai 0, dan metriknya memakai label yang sama dengan *reward***

- **Lokasi:** abstrak.md:3 ("VEGR menaikkan *precision-grounded* EM sebesar 24,4 poin"); bab5:14; bab1:56.
- **Observasi:**
  - P-gEM R0 per *seed* adalah 24,7 / 0 / 0 (bab4:39–41). Dibandingkan dengan satu-satunya *seed* R0 yang masih memberi sitasi, R1 hanya unggul sekitar 8 poin (32,7 vs 24,7; dihitung reviewer dari Tabel 4.1–4.2, deskriptif saja, satu *seed*). Lebarnya SK [+8,8; +37,3] mencerminkan hal ini.
  - Selisih ini juga dimungkinkan oleh celah lingkungan yang menerima jawaban tanpa sitasi (bab3:66, bab4:356).
  - VEGR dilatih pada label emas yang sama dengan metrik *grounding*. Bab 4 mengakui ini (bab4:351), dan bab5:43 juga, tetapi **abstrak tidak menyebutnya sama sekali**.
  - Di abstrak, pembanding untuk "menaikkan" tidak ditulis. Pembacanya bisa B0 (+19,7) atau R0 (+24,4).
- **Mengapa penting:** Ini adalah angka utama tesis. Penguji akan bertanya apakah angka ini mengukur "agen lebih ter-*grounding*" atau "R0 berhenti menyitasi karena celah lingkungan". Keduanya benar, tetapi abstrak hanya menyampaikan versi yang paling menguntungkan.
- **Perbaikan:**
  - Abstrak: "*Dibandingkan dengan reward berbasis jawaban saja, VEGR menaikkan precision-grounded EM sebesar 24,4 poin (SK [+8,8; +37,3]), terutama karena dua seed R0 berhenti memberi sitasi; metrik ini diukur terhadap supporting facts yang sama dengan yang dipakai reward.*"
  - Bab 5 butir *Grounding* (bab5:14): tambahkan satu kalimat bahwa terhadap R0 *seed* 7 (yang masih memberi sitasi) selisihnya jauh lebih kecil, dan bahwa metrik dan *reward* memakai label yang sama (rujuk ke Keterbatasan 6).

**M3. Mekanisme hilangnya sitasi ditulis sebagai penyebab pasti dan digeneralisasi dari satu *seed***

- **Lokasi:** abstrak.md:3 ("2 dari 3 *seed* berhenti memberi sitasi, **karena** sitasi yang tidak valid bernilai −1 ..."); bab5:15 ("*Rollout* pelatihan **menjelaskan penyebabnya** ... pada awal pelatihan **sekitar separuh** dari seluruh percobaan sitasi tidak valid"); bab4:112 (klaim serupa); bab1:56 ("Log pelatihan menunjukkan penyebabnya").
- **Observasi:**
  - Bab 4 sendiri lebih hati-hati: "Ini merupakan interpretasi atas log pelatihan dari dua *seed* R0 dan satu *seed* R1, dan interpretasi ini tidak menjelaskan mengapa *seed* 7 tetap memberi sitasi" (bab4:155).
  - Angka "sekitar separuh" dan perhitungan −0,27 vs +0,36 hanya berasal dari R0 s8 (bab4:139, 151; bab3:123–128). Untuk **R0 s9** pada blok 1–32, hanya 48 dari 209 percobaan sitasi yang tidak valid (≈ 23%). Nilai harapan dari mencoba menyitasi adalah (161 × 0,67 − 48)/209 ≈ **+0,29**, sedangkan tidak menyitasi bernilai +0,26. Jadi, pada blok pertama s9, **menyitasi justru sedikit lebih menguntungkan**. Baru pada blok 33–64 nilainya berbalik: (93 × 0,59 − 45)/138 ≈ +0,07 vs +0,24 (dihitung reviewer dari Tabel 4.4, bab4:143–144).
  - Penjelasan "*reward* tidak pernah memberi imbalan" juga berlaku sama untuk R0 s7, yang tidak runtuh.
- **Mengapa penting:** Mekanismenya masuk akal dan didukung data, tetapi "karena" di abstrak adalah klaim kausal dari dua *run* yang tidak dimanipulasi. Selain itu, bukti kuantitatif terkuatnya hanya berlaku untuk satu *seed*. Penguji yang menghitung ulang baris s9 akan menemukan ketidaksesuaian ini.
- **Perbaikan:**
  - Abstrak: "*... berhenti memberi sitasi. Log pelatihan konsisten dengan penjelasan bahwa sitasi yang tidak valid bernilai −1 sedangkan sitasi yang benar tidak diberi imbalan.*"
  - bab5:15: ganti "menjelaskan penyebabnya" dengan "konsisten dengan penjelasan berikut". Ganti "sekitar separuh" dengan "pada R0 *seed* 8 sekitar separuh, dan pada *seed* 9 sekitar seperempat (meningkat menjadi sepertiga pada blok kedua)".
  - Tambahkan satu kalimat di bab5:15: "*Penjelasan ini tidak menjawab mengapa seed 7 tetap memberi sitasi.*"
  - Lakukan perbaikan yang sama di bab4:112.

**M4. "Mencegah" disimpulkan dari 0 keruntuhan pada 3 *seed* R1**

- **Lokasi:** RQ2 (bab1:25, bab3:29), jawaban bab5:11 ("Ya"), bab4:126 ("terhindar dari hilangnya sitasi pada ketiga *seed*").
- **Observasi:** Bab 4 memberi SK untuk proporsi keruntuhan R0 (2/3), tetapi tidak untuk R1 (0/3). SK Clopper–Pearson 95% untuk 0/3 adalah [0; 0,71] (dihitung reviewer). Dengan kata lain, 3 *seed* tidak dapat menyingkirkan kemungkinan bahwa VEGR juga runtuh pada sebagian besar *seed* lain. Dukungan yang kuat untuk "mencegah" justru datang dari mekanismenya (Tabel 4.4, selisih *reward* ≈ 2 poin untuk sitasi yang valid), bukan dari frekuensi.
- **Mengapa penting:** Penguji yang melihat SK [0,09; 0,99] untuk R0 akan bertanya mengapa SK untuk R1 tidak dilaporkan.
- **Perbaikan:** Tambahkan di bab4:110 dan bab5:15 (atau Keterbatasan 4, bab5:41): "*Sebaliknya, tidak ada dari tiga seed R1 yang berhenti memberi sitasi (0/3, SK 95% Clopper–Pearson [0; 0,71]); dukungan bahwa VEGR mencegah keruntuhan terutama berasal dari mekanisme reward (Subbab 4.5), bukan dari frekuensi.*" Pertimbangkan mengganti "mencegah" di jawaban RQ2 menjadi "tidak mengalami ... pada ketiga *seed*, dan mekanismenya menjelaskan mengapa".

**M5. Kontradiksi: "*holdout* tidak pernah digunakan untuk memilih pengaturan" vs. varian yang dirancang setelah hasil *holdout***

- **Lokasi:** bab3:41 ("*Holdout* tidak pernah digunakan untuk memilih pengaturan. Seluruh pengaturan dipilih pada data validasi"), berlawanan dengan bab3:228, bab3:257, bab4:198, bab4:358 (R3, R4, dan R0-dyn "dirancang setelah hasil *holdout* R0/R1/R2 diketahui").
- **Observasi:** Kedua pernyataan tidak bisa sama-sama benar. Selain itu, menurut Tabel 3.2 (bab3:275–276), *seed* 9 dari R0, R1, dan R2 dijalankan di *batch* perbaikan, setelah hasil *seed* 7–8 dianalisis. Naskah tidak menyatakan apakah *seed* ketiga sudah direncanakan sebelumnya. Jika belum, jumlah *seed* untuk perbandingan **utama** juga ditentukan setelah melihat hasil (*optional stopping*).
- **Mengapa penting:** Kebocoran *holdout* adalah pertanyaan standar penguji. Dampaknya kecil karena perbandingan yang terdampak sudah diberi label eksploratif, tetapi pernyataan bab3:41 dalam bentuknya sekarang tidak akurat.
- **Perbaikan:** Ubah bab3:41 menjadi: "*Hyperparameter pelatihan tidak pernah dipilih menggunakan holdout. Namun, varian R3, R4, dan R0-dyn dirancang setelah hasil holdout R0/R1/R2 diketahui, sehingga perbandingannya diperlakukan sebagai eksploratif (Subbab 3.8).*" Di bab3:257, tambahkan satu kalimat yang menjelaskan apakah *seed* 9 untuk R0/R1/R2 sudah direncanakan sejak awal (misalnya dengan merujuk ke `logs/2026-10-10-r0-r1-r2-batch.md`) atau ditambahkan kemudian. Jika ditambahkan kemudian, sebutkan di Keterbatasan 5 (bab5:42).

### MINOR

**m1. Urutan angka terbalik di bab5:13**

- **Lokasi:** bab5:13 "*Reward* berbasis jawaban saja dan paket VEGR lengkap mencapai akurasi yang sama (43,6% vs. 42,6% EM; +1,0 ...)".
- **Observasi:** Subjek pertama dalam kalimat adalah R0, tetapi angka pertama adalah milik R1 (bab4:61–62: R0 = 42,6, R1 = 43,6).
- **Mengapa penting:** Pembaca akan mengira R0 = 43,6%.
- **Perbaikan:** "(42,6% vs. 43,6% EM; R1 − R0 = +1,0, SK [−3,6; +5,4])". Sekalian terapkan M1.

**m2. RQ3 menanyakan "perilaku pencarian" dan Tujuan 3 menyebut "stabilitas pelatihan", tetapi jawaban RQ3 di Bab 5 tidak membahas keduanya**

- **Lokasi:** bab1:26 (RQ3: "... terhadap *grounding* dan perilaku pencarian"); bab1:36 (Tujuan 3: "... *grounding*, cakupan bukti, dan stabilitas pelatihan"); jawaban di bab5:19–23 dan bab4:231–237.
- **Observasi:** Bab 5 hanya membahas EM, P-gEM, cakupan, dan Joint-EM. Datanya sebenarnya sudah ada di Bab 4: jumlah pencarian dan "Dup." (Tabel 4.1), sebaran antar-*seed* R4 (bab4:214), rentang R2 (bab4:200), dan langkah yang dilewati (Tabel 4.5).
- **Mengapa penting:** Penguji biasanya memeriksa apakah setiap RQ dan tujuan dijawab secara eksplisit.
- **Perbaikan:** Tambahkan satu butir di bab5:19–23, misalnya: "*Perilaku pencarian dan stabilitas: jumlah pencarian per episode varian R2–R4 (1,61–2,41) mirip dengan R1 (1,91–2,30), dan tidak ada varian yang menunjukkan pencarian berulang seperti R0 s7. Sebaran EM antar-seed R2 dan R4 lebih lebar daripada R1 (sd 5,4 dan 4,0 vs 1,5), tetapi tiga seed tidak cukup untuk menyatakannya sebagai perbedaan stabilitas.*" (angka dari Tabel 4.1–4.2; sesuaikan bila perlu).

**m3. Abstrak tidak menyebut bahwa angka "EM pada episode selesai 45,2%" adalah milik R1**

- **Lokasi:** abstrak.md:3 "EM pada episode yang selesai hanya naik dari 39,4% menjadi 45,2%".
- **Observasi:** Kalimat sebelumnya membahas "pelatihan" secara umum. Rentang untuk semua metode adalah 43,8–45,3% (bab4:90). Bab 5 sudah menulis "(R1)" (bab5:9).
- **Mengapa penting:** Pembaca abstrak bisa mengira 45,2% adalah angka untuk semua metode, padahal abstrak harus bisa dibaca tanpa bab lain.
- **Perbaikan:** "... dari 39,4% menjadi 43,8–45,3% (45,2% untuk VEGR)". Pertimbangkan juga menambahkan SK untuk kenaikan sekitar 20 poin (mis. "semua SK di atas +13,8", yaitu batas bawah terendah, R2 − B0, Tabel 4.3) agar ketidakpastiannya ikut tersampaikan.

**m4. Klaim peringkat R3 "Joint-EM tertinggi" kehilangan status eksploratif dan kerapuhannya**

- **Lokasi:** abstrak.md:3 ("memperoleh Joint-EM HotpotQA tertinggi"); bab5:22 ("Joint-EM HotpotQA terbaik"); bab1:59 ("menempati peringkat terbaik").
- **Observasi:** Bab 4 sendiri menyimpulkan "Kedua selang berada dekat dengan nol, sehingga tidak ada peringkat yang kokoh" (bab4:212). R3 dirancang setelah melihat *holdout* (bab3:257), dan R3 vs R2 tidak pernah diuji (13,4 vs 10,0, Tabel 4.2).
- **Mengapa penting:** Abstrak dan Bab 1 menyatakan peringkat yang ditolak sendiri oleh Bab 4. Penguji yang membaca bab4:212 akan melihat ketidakkonsistenan ini.
- **Perbaikan:** Abstrak: "*Dalam analisis eksploratif, bobot cakupan yang lebih besar menukar presisi dengan kelengkapan dan memperoleh rata-rata Joint-EM HotpotQA tertinggi, meskipun selangnya nyaris memuat nol.*" Bab 5 dan Bab 1: ganti "terbaik" dengan "rata-rata tertinggi (eksploratif)".

**m5. Keterbatasan di Bab 5 tidak memuat beberapa ancaman yang ada di Bab 4**

- **Lokasi:** bab5:36–45, dibandingkan dengan bab4:347–359.
- **Observasi:** Hal-hal berikut tidak masuk ke 5.3:
  - (a) Tidak ada *baseline* berbasis *prompt* (bab4:352). Ini relevan langsung untuk jawaban RQ1 karena kenaikannya sebagian besar adalah pembelajaran protokol. Saat ini hal ini hanya muncul sebagai saran (bab5:61).
  - (b) Perubahan versi PyTorch antara B0/R1 s7 dan *run* lain (bab4:355, bab3:261–266).
  - (c) Label *supporting facts* HotpotQA tidak lengkap (bab4:351).
  - (d) *Holdout* adalah 300 baris valid pertama dari berkas dev (bab3:41), bukan sampel acak.
  - (e) `run_metadata.json` tidak mencatat *commit* git (bab3:288).
  - (f) Alasan R0 s7 tidak runtuh tidak diketahui (bab4:155).
- **Mengapa penting:** Bab 1 menyatakan "Keterbatasan lainnya dibahas pada Bab 5" (bab1:49), jadi 5.3 seharusnya menjadi daftar lengkap.
- **Perbaikan:** Tambahkan butir 9–10 di 5.3. Contoh: "***Baseline. Tidak ada baseline berbasis prompt few-shot, sehingga tidak diketahui berapa bagian kenaikan RQ1 yang dapat dicapai tanpa pelatihan.***" dan "***Reproduktibilitas.** Versi PyTorch berubah setelah B0/R1 seed 7; commit git tidak tercatat per run; adapter R1 seed 7 hilang.*" Gabungkan (c) ke butir 6 dan (f) ke butir 4.

**m6. Angka "sekitar seperempat" di 5.4 tidak tertelusur dan merujuk ke agen yang tidak jelas**

- **Lokasi:** bab5:49 "Bahkan agen VEGR terbaik masih menyisakan sekitar seperempat jawaban benarnya dengan setidaknya satu sitasi yang bukan kalimat emas".
- **Observasi:** Angka ini tidak ada di Bab 4. Jika dihitung dari Tabel 4.1–4.2 (EM − P-gEM)/EM: rata-rata R1 = (43,6 − 32,7)/43,6 ≈ 25%. Untuk *run* VEGR dengan P-gEM tertinggi (R1 s9) ≈ 17%, dan untuk R4 s9 ≈ 16% (dihitung reviewer). Jadi "seperempat" sesuai dengan *rata-rata* R1, bukan dengan agen "terbaik". Selain itu, selisih EM − P-gEM juga mencakup jawaban benar tanpa sitasi, tidak hanya jawaban dengan sitasi non-emas (pengaruhnya kecil karena tingkat sitasi 99–100%, bab4:114).
- **Mengapa penting:** Setiap angka di Bab 5 seharusnya dapat ditelusuri ke Bab 4. Angka ini juga dipakai sebagai dasar argumen etika.
- **Perbaikan:** "*Rata-rata, sekitar seperempat jawaban benar R1 tidak precision-grounded (43,6% EM vs 32,7% P-gEM, Tabel 4.2), dan Sup-EM R1 hanya 11,9%, sehingga agen jarang menyitasi bukti secara lengkap.*"

**m7. Rujukan "pemeriksaan independen pada Subbab 4.4" tidak sesuai isi dan bertentangan dengan Bab 5**

- **Lokasi:** bab4:351 ("... serta pemeriksaan independen pada Subbab 4.4 mengurangi ... kekhawatiran ini") dan bab5:43 ("tidak ada pemeriksaan atribusi yang independen dari label emas yang dijalankan").
- **Observasi:** Satu-satunya pemeriksaan tambahan di 4.4 adalah analisis 136 pertanyaan yang sama-sama selesai (bab4:114). Analisis itu menguji efek seleksi dan tetap memakai label emas, jadi bukan pemeriksaan independen terhadap tumpang-tindih *reward*–metrik.
- **Mengapa penting:** Ancaman Goodhart adalah salah satu kelemahan utama tesis. Menyebut ada "pemeriksaan independen" padahal tidak ada akan terlihat sebagai klaim berlebihan, apalagi Bab 5 menyatakan sebaliknya.
- **Perbaikan:** Di bab4:351, ganti dengan "*... serta analisis pada 136 pertanyaan yang diselesaikan semua run (Subbab 4.4, yang menguji efek seleksi, bukan ketergantungan pada label emas) ...*".

**m8. Klaim "VEGR tidak menaikkan cakupan dibanding B0" belum dicek terhadap efek seleksi**

- **Lokasi:** abstrak.md:3, bab5:17, bab4:129, bab4:345.
- **Observasi:** Cakupan dirata-ratakan atas episode yang selesai. B0 hanya menyelesaikan 170 episode, yaitu subset yang kemungkinan lebih mudah. Bab 4 sudah mengoreksi efek ini untuk presisi (136 pertanyaan bersama, bab4:114), tetapi tidak untuk cakupan.
- **Mengapa penting:** Ini adalah klaim negatif yang muncul di abstrak. Jika pada 136 pertanyaan bersama cakupan R1 ternyata lebih tinggi, klaim tersebut berubah.
- **Perbaikan:** `scripts/peer_review_reanalysis.py:83–88` ternyata sudah menghitung "Coverage (common set)" untuk pertanyaan yang diselesaikan semua *run*, tetapi angkanya tidak dilaporkan di Bab 4. Ambil angka itu dari keluarannya dan laporkan dalam satu kalimat di bab4:114. Bila hasilnya tidak tersedia, tulis di abstrak "*tidak menaikkan rata-rata cakupan bukti (dihitung atas episode yang selesai)*".

**m9. "Pelatihan GRPO" vs. "bukan GRPO secara penuh"**

- **Lokasi:** bab1:24 (RQ1 "pelatihan GRPO"), bab3:28, bab4:32 (judul "Apakah GRPO Meningkatkan ..."), bab5:9, berlawanan dengan bab5:44 ("bukan GRPO secara penuh") dan bab3:172.
- **Observasi:** Abstrak dan bab5:7 sudah konsisten memakai "bergaya GRPO", tetapi RQ1 dan judul subbab belum.
- **Mengapa penting:** Penguji bisa bertanya mengapa RQ1 menyebut "GRPO" padahal Bab 5 menyatakan algoritmanya bukan GRPO penuh.
- **Perbaikan:** RQ1: "*Apakah pelatihan RL bergaya GRPO meningkatkan ...*". Sesuaikan juga bab3:28, bab4:32, dan bab5:9.

**m10. Bahasa Inggris (ABSTRACT) tidak ikut dalam *build* naskah Indonesia**

- **Lokasi:** `build_id_docx.py` ORDER (baris 26–35) hanya memuat "abstrak", dan *grep* tanpa membedakan huruf besar-kecil tidak menemukan "abstract" di skrip selain di *docstring*. Versi Inggris ada di `chapters/abstract.md`, tetapi tidak ada di `chapters-id/`.
- **Observasi:** Format TA Telkom University umumnya mensyaratkan ABSTRAK dan ABSTRACT. Isi `chapters/abstract.md` sejajar dengan abstrak.md dan akan perlu menerima perbaikan M1–M3 dan m3–m4 yang sama. Kata kuncinya juga berbeda: versi Inggris memuat GRPO dan HotpotQA, sedangkan versi Indonesia memuat RAG.
- **Mengapa penting:** Halaman depan yang tidak lengkap adalah masalah format yang langsung terlihat saat pemeriksaan administrasi.
- **Perbaikan:** Tambahkan `abstract.md` ke `chapters-id/` dan ke ORDER (setelah abstrak), samakan isi dengan abstrak yang sudah direvisi, dan samakan daftar kata kunci.

**m11. Catatan "[FIGURE UPDATE NEEDED]" masih ada, Gambar 4.5 belum sesuai dengan selang utama, dan nomor gambar tidak berurutan**

- **Lokasi:** bab4:86, 124, 176, 229. Gambar muncul dengan urutan 4.5 (bab4:82), 4.3, 4.1, 4.2, 4.7, 4.4, 4.6.
- **Mengapa penting:** Catatan internal akan ikut tercetak di docx. Gambar 4.5 menampilkan selang yang di teks disebut "hanya acuan" (bab4:67), sehingga gambar dan teks tidak konsisten. Nomor gambar seharusnya urut sesuai kemunculan.
- **Perbaikan:** Buat ulang gambar sesuai catatan tersebut, hapus catatannya, lalu beri nomor ulang gambar 4.1–4.7 sesuai urutan kemunculan.

**m12. R0 disebut sebagai varian yang "mengubah satu aspek VEGR"**

- **Lokasi:** bab3:115 ("Tiga varian mengubah satu aspek VEGR pada satu waktu"), diikuti R0 (bab3:117).
- **Observasi:** Ini bertentangan dengan bab3:9, yang menyatakan R0 berbeda dari VEGR pada enam komponen.
- **Mengapa penting:** Kalimat ini melemahkan kehati-hatian Subbab 3.1 bahwa R1 − R0 adalah perbandingan paket, bukan perbandingan satu komponen.
- **Perbaikan:** "*R0 adalah reward dasar (bukan varian satu aspek); R2 dan R3 masing-masing mengubah satu aspek VEGR.*"

**m13. Kesimpulan penutup abstrak berupa preskripsi umum**

- **Lokasi:** abstrak.md:3 "*Grounding* harus diberi *reward* dan diukur secara eksplisit; akurasi saja tidak mengungkapkannya."
- **Observasi:** "Diukur secara eksplisit" didukung oleh data (EM sama, P-gEM sangat berbeda). "Harus diberi *reward*" adalah generalisasi dari satu *dataset* dan satu model, dan R0 s7 tetap memberi sitasi tanpa *reward*.
- **Mengapa penting:** Kalimat terakhir abstrak paling sering dikutip. Isinya bertentangan dengan batasan di bab1:46 bahwa temuan tidak diklaim berlaku di luar pengaturan ini.
- **Perbaikan:** "*Dalam pengaturan ini, grounding baru terlihat ketika diukur secara eksplisit, dan baru stabil ketika diberi reward; akurasi saja tidak mengungkapkannya.*"

**m14. Perbaikan di abstrak akan menambah panjang abstrak yang sudah dibatasi satu halaman**

- **Lokasi:** abstrak.md:3 (sekitar 295 kata termasuk kata kunci; *commit* 05856b0 membatasinya menjadi satu halaman).
- **Observasi:** Perbaikan M2, M3, m3, m4, dan m13 menambah sekitar 50 kata.
- **Mengapa penting:** Abstrak bisa melewati satu halaman.
- **Perbaikan:** Imbangi dengan memangkas bagian lain. Kandidatnya: (a) gabungkan kalimat "Agen melakukan pencarian dengan BM25 ..." ke kalimat sebelumnya ("... pada satu GPU 48 GB, dengan pencarian BM25 pada 10 paragraf milik setiap pertanyaan"); (b) ringkas rincian komponen VEGR ("VEGR memberi *reward* pada jawaban yang benar, sitasi yang benar, dan pencarian yang berguna, serta memberi penalti ...") menjadi "VEGR menjumlahkan komponen jawaban, bukti, dan pencarian"; (c) hapus kalimat *dynamic sampling* karena hasilnya hanya dari satu *seed* dan sudah dibahas di Bab 5.

### NIT

- **n1. "SK" tidak didefinisikan di abstrak.** abstrak.md:3 memakai "selang kepercayaan *bootstrap*", lalu "SK 95%" tanpa definisi. Abstrak harus bisa dibaca tanpa bab lain. Perbaikan: "selang kepercayaan (SK) *bootstrap* ...".
- **n2. Istilah yang berganti-ganti** (dari *grep* `-n` di `chapters-id/`). Pilih satu istilah per konsep dan definisikan di Bab 1:
  - "*outcome-only*" (mis. bab1:18, bab3:117, bab4:39–41, 61, 163, 335–336; 28 baris di Bab 1–4) vs "*reward* berbasis jawaban saja" (abstrak.md:3, bab5:13, 15, 30). Bab 5 sama sekali tidak memakai "*outcome-only*", sedangkan tabel Bab 4 hanya memakai istilah itu.
  - "pembaruan" (mis. bab3:253, bab4:34, 184) vs "*update*" (bab5:16, 30, 42, 44, 57).
  - "grup" untuk grup *rollout* (bab3:159, bab4:157) vs "kelompok" dengan arti yang sama (bab2:46, bab5:7, bab5:30). Catatan: "tiga kelompok" di bab3:90 berarti kelompok komponen *reward*, jadi tidak perlu diubah.
  - "varians" (bab3:10, 159, 185, 226, 253; bab5:30) vs "variansi" (bab2:72, bab4:67, 102, 157).
  - "simpangan baku" (Bab 2–4) vs "standar deviasi" (bab1:26, bab5:21).
  - "penalti bergerbang" (bab1:26, bab1:58, bab5:21) vs "penalti bersyarat" (bab3:118) vs "penalti *gated*" (bab4:200, 233, 343).
  - "jalan pintas" vs "keruntuhan" vs "hilangnya sitasi" vs "berhenti memberi sitasi". bab4:108 secara eksplisit menyatakan bahwa perilaku R0 s7 "bukan jalan pintas", jadi definisi jalan pintas perlu konsisten.
- **n3. "perbaikan *optimizer*" (bab3:30) vs "perubahan *optimizer*" (bab1:26, bab5:19).** "Perbaikan" mengandaikan hasil yang lebih baik. Gunakan "perubahan".
- **n4. RQ1 di bab3:28 dan bab5:9 tidak menyebut "pada HotpotQA",** sedangkan bab1:24 menyebutnya. Samakan redaksinya.
- **n5. Sistematika di bab1:72 tidak menyebut subbab Kontribusi (5.2), dan urutannya (etika sebelum keterbatasan) berbeda dari Bab 5** (keterbatasan 5.3, etika 5.4).
- **n6. Kesimpulan tidak kembali ke G1–G4.** Bab 1 (bab1:17–20) dan Bab 2 (bab2:108–111) membingkai empat celah, tetapi Bab 5 hanya menjawab RQ. Satu kalimat di 5.2 yang memetakan kontribusi ke G1–G4 akan menutup argumen.
- **n7. Lisensi Qwen2.5-7B-Instruct (bab5:51).** Setahu reviewer, varian 7B dirilis di bawah Apache 2.0 (lisensi khusus Qwen berlaku untuk ukuran tertentu lain). Periksa *model card* dan tulis nama lisensinya secara eksplisit.
- **n8. bab3:151 "fungsi *reward* milik *run* tersebut (R0–R3)"** tidak mencakup R4 dan R0-dyn. Tulis "(R0–R4, R0-dyn)".
- **n9. bab4:118 "presisi 0,63–0,76"** untuk *run* dengan *reward* bukti sudah benar, tetapi R0 s7 (0,621) tidak disebut di kelompok mana pun. Tambahkan "R0 s7 berada pada 0,62".
- **n10. Judul vs isi.** Judul (build_id_docx.py:40) memakai "Agen RAG", sedangkan seluruh isi memakai "agen pencarian" pada kumpulan tertutup yang "lebih dekat dengan *reading comprehension*" (bab3:49). Tambahkan satu kalimat di bab1:42 yang menyatakan bahwa agen RAG pada judul adalah agen pencarian dalam pengertian terbatas ini.
- **n11. Bab 1 menyebut "anggaran proyek sekitar $45" (bab1:44) sedangkan bab5:5 menyebut "anggaran mahasiswa"** tanpa angka. Cukup sebutkan sekali secara konsisten.

## 5. Penilaian Bab 5 per RQ

| RQ (Bab 1) | Jawaban Bab 5 | Bukti Bab 4 | Kekuatan klaim |
|---|---|---|---|
| RQ1 (bab1:24) | "Ya, sebagian besar melalui pembelajaran protokol" (bab5:9) | Tabel 4.3 (semua SK > 0), Tabel 4.2 EM selesai, Tabel 4.6 | **Didukung.** Tambahkan bahwa *baseline* berbasis *prompt* tidak diuji (m5a) |
| RQ2 (bab1:25) | "Ya, dalam pengaturan penelitian ini" (bab5:11) | Tabel 4.3 baris R1−R0, Tabel 4.4, bab4:110 | **Terlalu kuat** di dua bagian: "tanpa kehilangan akurasi" (M1) dan "mencegah" (M4). Bagian *grounding* didukung, dengan catatan M2 |
| RQ3 (bab1:26) | "Eksploratif; tidak ada varian yang terbukti lebih baik" (bab5:19) | bab4:196–237 | **Didukung dan proporsional**, tetapi perilaku pencarian dan stabilitas tidak dijawab (m2) |

Keterbatasan (5.3) jujur dan konkret, dan butir 4–6 sangat baik, tetapi belum lengkap (m5). Saran (5.5) konkret dan terkait langsung dengan kelemahan desain (ablasi, celah, NLI, *prompt baseline*). Satu saran yang perlu ditambahkan: **menjalankan R0 dan R1 pada lebih banyak *seed*** untuk memperkirakan frekuensi keruntuhan, karena SK [0,09; 0,99] dan [0; 0,71] hampir tidak informatif. Saran 8 (bab5:64) hanya membahas R0-dyn.

## 6. Perkiraan Pertanyaan Sidang

1. **"Anda menyimpulkan VEGR tidak kehilangan akurasi. Berapa penurunan akurasi yang masih bisa disembunyikan oleh data Anda?"**
   Jawaban yang baik: menyebut batas bawah SK −3,6 poin (bab4:76), mengakui tidak ada margin non-inferioritas yang ditetapkan, dan menyatakan klaim yang tepat adalah "tidak dapat dibedakan", tidak lebih. Rujuk juga ke Keterbatasan 4 (bab5:41) tentang daya statistik.
2. **"Berapa bagian dari 24,4 poin itu yang merupakan kemampuan VEGR, dan berapa yang sekadar R0 berhenti menyitasi karena lingkungan mengizinkannya?"**
   Jawaban yang baik: per *seed*, R0 = 24,7 / 0 / 0 (Tabel 4.1). Terhadap R0 s7, selisihnya sekitar 8 poin. Celah lingkungan sengaja dibiarkan (bab3:66). Metrik dan *reward* memakai label yang sama (bab4:351). Usulan untuk membedakannya: menutup celah (saran 2) dan memakai pemeriksaan NLI (saran 4).
3. **"Mengapa R0 *seed* 7 tidak runtuh, padahal *reward*-nya sama? Apakah penjelasan mekanisme Anda masih berlaku?"**
   Jawaban yang baik: mengakui bahwa hal ini tidak dijelaskan (bab4:155). Tunjukkan bahwa nilai harapan menyitasi bergantung pada tingkat sitasi tidak valid awal, yang berbeda antar-*seed* (s8 ≈ 51%, s9 ≈ 23% pada blok pertama, Tabel 4.4). Jadi mekanisme ini membuat keruntuhan *mungkin*, bukan *pasti*. Sebutkan bahwa data *rollout* R0 s7 perlu diperiksa.
4. **"Kenaikan +20 poin sebagian besar adalah kepatuhan format. Apakah *prompt few-shot* tidak akan memberikan hasil yang sama dengan biaya nol?"**
   Jawaban yang baik: B0 gagal pada 130/300 episode, 85 di antaranya kegagalan bukti (bab4:28). EM selesai hanya naik 4–6 poin (bab4:92). Mengakui *baseline* ini tidak dijalankan (bab4:352) dan menjadi saran 5 (bab5:61). Tambahkan bahwa RL juga mengubah perilaku sitasi (presisi naik pada 136 pertanyaan bersama, bab4:114), yang belum tentu didapat dari *prompting*.
5. **"Apakah VEGR hanya memindahkan *reward hacking* ke bentuk lain, yaitu menyitasi satu kalimat 'aman' di awal paragraf?"**
   Jawaban yang baik: mengakui pola "presisi tetapi tidak lengkap" (bab4:186–194), yaitu 93% sitasi berada di kalimat 0–2 (Gambar 4.7) dan R1 s9 memiliki Sup-EM 0,7%. P-gEM justru memberi nilai pada perilaku ini, sehingga Sup-EM/Joint-EM dilaporkan bersamanya. R3 menunjukkan bahwa bobot yang berbeda menggeser *trade-off* tersebut. Saran 3 (*reward* F1 himpunan sitasi) adalah perbaikan yang tepat.

Pertanyaan cadangan: "Mengapa *holdout* dikatakan tidak dipakai untuk memilih pengaturan, padahal R3/R4/R0-dyn dirancang setelah melihatnya, dan kapan *seed* 9 diputuskan?" (M5). "Apakah perbandingan dengan B0 terganggu oleh perubahan versi PyTorch?" (bab4:355).

## 7. Batasan Review Ini

- Reviewer tidak menjalankan ulang skrip atau *bootstrap*. Semua angka "dihitung reviewer" adalah aritmetika sederhana dari tabel di naskah.
- Bab 2 dan Lampiran A tidak di-*review* secara isi. Pemeriksaannya hanya sebatas konsistensi celah G1–G4 (bab2:108–111, cocok dengan bab1:17–20).
- Ketentuan format resmi Telkom University (urutan halaman depan, batas kata abstrak) tidak diverifikasi dari dokumen pedoman. Abstrak saat ini sekitar 295 kata termasuk kata kunci.
- Tidak ada isi naskah yang dikirim ke layanan eksternal.
- Skrip pendukung dari *skill* `peer-review` (validasi *intake*, matriks klaim–bukti, *lint*) tidak dijalankan. Otorisasi dianggap jelas karena *review* ini diminta oleh penulis untuk naskahnya sendiri, dan matriks klaim–bukti disajikan langsung di Bagian 3.
