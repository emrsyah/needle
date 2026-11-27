# Review 2: Bab 3 (Perancangan Sistem) dan Lampiran A (Prompt)

> **Draf kerja dari reviewer berbantuan AI.** Penulis tetap harus memeriksa setiap lokasi, angka, dan klaim sebelum menindaklanjutinya. Review ini tidak memberi rekomendasi lulus/tidak lulus.

## 0. Ruang lingkup, bahan, dan batasan

- **Ruang lingkup:** `docs/thesis/chapters-id/bab3-perancangan-sistem.md` (selanjutnya `bab3`) dan `docs/thesis/chapters-id/lampiran-a-prompt.md` (`lampA`). Fokusnya adalah validitas rancangan eksperimen, kesesuaian teks dengan kode, reproduktibilitas, definisi metrik, dan rencana analisis statistik.
- **Kode yang diperiksa:**
  - `src/needle/rewards/__init__.py`, `src/needle/environment/search.py`, `src/needle/inference/runner.py`
  - `src/needle/training/{grpo,collector,local_policy,runtime}.py`, `src/needle/retrieval/bm25.py`, `src/needle/actions/__init__.py`
  - `scripts/{train_grpo,prepare_splits,evaluate_checkpoint,analyze_eval,peer_review_reanalysis}.py`
  - `configs/training/*.json` dan `configs/evaluation/*.json`
- **Data hasil yang dibaca ulang (read-only):**
  - `results/2026-10-10-batch/`, `results/2026-10-11-fixes/{runs,results}/`, `results/2026-10-10-grpo-smoke/b0_episodes.jsonl`
  - `logs/2026-10-10-*.md` dan `logs/2026-10-11-fixes-batch.md`
  - `docs/research/2026-10-11-results-diagnosis-and-improvements.md`
- **Batasan:**
  - Berkas sumber HotpotQA tidak ada di mesin lokal, sehingga validasi sitasi tidak bisa diputar ulang terhadap korpus. Temuan M1 karena itu memakai uji tidak langsung yang hanya memberi batas bawah.
  - Tidak ada pelatihan atau evaluasi yang dijalankan ulang.
  - Angka yang tidak saya hitung ulang ditandai "tidak diverifikasi".
- **Standar penilaian:** tugas akhir S1. Kebaruan bukan tujuan, dan kesederhanaan memang disengaja. Yang dinilai adalah apakah klaim sepadan dengan bukti dan apakah deskripsi metode jujur terhadap kode.

## 1. Peta singkat

- **Pertanyaan:** apakah *reward* bukti yang diperiksa aturan (VEGR, R1) mencegah jalan pintas *reward* berbasis jawaban saja (R0) tanpa kehilangan akurasi. Pertanyaan ini diajukan untuk agen pencarian Qwen2.5-7B + LoRA di HotpotQA *distractor*, dengan BM25 tertutup atas 10 paragraf.
- **Unit:** *run* pelatihan (3 *seed* per metode). Evaluasi dilakukan pada 300 pertanyaan *holdout* dengan *greedy decoding*.
- **Pembanding:** B0 (tanpa pelatihan), R0, R2, R3, R4, dan R0-dyn (1 *seed*).
- **Luaran:** EM atas 300 episode, *precision-grounded* EM, Sup-EM, Joint-EM/F1, tingkat sitasi, dan diagnostik pelatihan.

## 2. Kekuatan (layak dipertahankan saat sidang)

1. **Pengakuan keterbatasan sangat jujur.** Bab 3 sendiri sudah menyatakan bahwa R1 vs R0 mengubah seluruh paket *reward* (bab3:9), bahwa jumlah pembaruan berbeda (bab3:10), bahwa ρ≈1 sehingga metodenya efektif REINFORCE (bab3:172), bahwa sitasi kosong lolos (bab3:66), dan bahwa R3/R4/R0-dyn bersifat *post hoc* (bab3:257). Saya **tidak** menghitung hal-hal ini sebagai temuan Major baru, karena sudah diungkap. Yang perlu dicek hanyalah apakah pengungkapan itu terbawa ke kalimat kesimpulan (lihat M4).
2. **Deskripsi algoritma cocok dengan kode.**
   - *Advantage* std populasi + 1e-6 dan *skip* grup varians-nol (`grpo.py:26-42`, `grpo.py:160-166`, `train_grpo.py:180`).
   - Rasio tingkat sekuens dari rata-rata *log-prob* semua token aksi (`grpo.py:98-111`).
   - Satu `optimizer.step()` per pertanyaan (`train_grpo.py:186-208`).
   - Tanpa KL.
   - *Hyperparameter* (lr 2e-5, ε 0,2, *grad clip* 1,0, G=8, 1 *epoch*, `train_on_invalid`, LoRA r16/α32/*dropout* 0 dengan 7 modul, T=1,0, top-p 1,0, 128 token, top-k=3, anggaran 3) identik di 16 berkas (`configs/training/r*_*.json` + `grpo_run2.json`). Satu-satunya perbedaan di luar bobot *reward* dan `normalize_advantage_std` adalah `num_train_examples` = 512 pada R0-dyn.
   - Fraksi *clip* = 0 terkonfirmasi pada `r0_outcome_seed8` dan `r0_dynsample_seed8` (`metrics.jsonl`).
3. **Pembagian data reproducible dan cocok dengan teks.** `prepare_splits.py:145-151` (sha256 `"7:"+id`, 8 *byte*, < 0,95) sesuai dengan bab3:40. `splits.json` mencatat 85.573 / 4.452 / 300, dan *holdout* diverifikasi dengan *hash manifest* (`prepare_splits.py:159-185`).
4. **Rancangan berpasangan (*common random numbers*).** `select_train_ids` hanya bergantung pada *seed* (`train_grpo.py:45-53`), sehingga R0/R1/R2/R3/R4 pada *seed* yang sama melatih 256 pertanyaan yang sama dengan urutan yang sama. Ini kekuatan yang belum disebut eksplisit di bab3:180 dan layak ditambahkan.
5. **Contoh perhitungan Tabel 3.1 terverifikasi persis.** Total 4,933 vs 2,467 cocok dengan `results/2026-10-10-batch/{r1_full_seed8,r0_outcome_seed8}/episodes.jsonl`.
6. **Statistik sadar-*seed*.** Pemisahan analisis utama dan eksploratif, serta selang Clopper–Pearson untuk proporsi *seed*, sudah di atas rata-rata TA S1.

## 3. Temuan Major

### M1. Definisi "kegagalan bukti" di teks tidak sesuai kode: sitasi ke dokumen yang tidak pernah diambil tetap diterima

- **Lokasi teks:**
  - bab3:61 ("sitasi merujuk pada judul atau indeks kalimat yang tidak pernah muncul pada hasil pencarian mana pun")
  - bab3:88 ("agen tidak dapat menaikkan presisinya dengan menyitasi kalimat yang tidak pernah ia ambil")
  - bab3:121 ("sitasi ke judul atau kalimat yang tidak diambil membuat episode tersebut menjadi kegagalan bukti")
  - lampA:26 (aturan *prompt* "Use only the supplied search results for citations")
- **Lokasi kode:**
  - `src/needle/environment/search.py:169-181`. Validasi hanya memeriksa bahwa judul ada di **10 dokumen milik pertanyaan** (`self._example.documents`) dan bahwa indeks kalimat dalam jangkauan. Tidak ada pemeriksaan terhadap `self._history`.
  - `collector.py:229-234`. Kegagalan diklasifikasikan sebagai EVIDENCE bila pesan galat memuat "citation".
  - `actions/__init__.py:13` sendiri menyatakan bahwa validasi korpus adalah tanggung jawab `SearchEnvironment`, dan `SearchEnvironment` tidak memeriksa riwayat pencarian.
- **Bukti empiris (batas bawah):** jika semua dokumen yang disitasi pernah diambil, maka *coverage* ≤ *retrieval recall* (fakta emas yang disitasi ⊆ fakta emas yang terekspos). Episode selesai dengan *coverage* > *retrieval recall* karena itu pasti menyitasi kalimat emas dari dokumen yang tidak pernah diambil, dan episode itu dinilai valid.
  - `r2_gated_seed7`, `5a8db19d5542994ba4e3dd00`: Cov 1,0, RetR 0,5
  - `r4_drgrpo_seed7`, `5adc53f75542996e6852530a`: Cov 1,0, RetR 0,5
  - B0 (`b0_episodes.jsonl`): 2 episode (`5a722b8655429971e9dc9329`, `5adc53f75542996e6852530a`)

  Uji ini hanya menangkap sitasi **emas** dari dokumen yang tidak diambil. Sitasi non-emas ke dokumen pengecoh yang tidak diambil tidak terdeteksi tanpa korpus, sehingga jumlah sebenarnya bisa lebih besar.
- **Mengapa penting bagi penguji:**
  - Huruf "V" (*verifiable*) dan klaim "agen tidak dapat menaikkan presisi tanpa mengambil" adalah inti narasi VEGR.
  - Penjelasan mekanisme jalan pintas R0 (bab3:121-128) bergantung pada definisi ini. Kegagalan bukti sebenarnya adalah judul yang dikarang (di luar 10 dokumen) atau indeks di luar jangkauan, bukan "tidak diambil".
  - Penguji yang membaca `search.py` akan langsung menemukan ketidaksesuaian ini.
- **Perbaikan (pilih salah satu, tanpa *run* ulang):**
  - (a) Ubah teks menjadi "sitasi harus merujuk pada judul yang ada di 10 paragraf pertanyaan tersebut dan indeks kalimat yang valid". Lalu ubah bab3:88 dan bab3:121 menjadi "sitasi ke judul yang tidak ada di korpus pertanyaan (praktisnya, judul karangan) …".
  - (b) Tambahkan satu kalimat keterbatasan: "lingkungan tidak memeriksa apakah dokumen yang disitasi pernah diambil. Pada data *holdout*, setidaknya N episode menyitasi kalimat emas dari dokumen yang tidak diambil" (isi N dari pemindaian di atas).
  - (c) Jelaskan di lampA:26 bahwa aturan "Use only the supplied search results" hanya instruksi *prompt*, bukan aturan yang ditegakkan lingkungan.

### M2. Klaim "*holdout* tidak pernah digunakan untuk memilih pengaturan" bertentangan dengan dokumen sendiri

- **Lokasi:** bab3:41 ("*Holdout* tidak pernah digunakan untuk memilih pengaturan. Seluruh pengaturan dipilih pada data validasi") vs. bab3:228 dan bab3:257 (R3, R4, dan R0-dyn dirancang "setelah hasil *holdout* diketahui").
- **Bukti:**
  - Evaluasi validasi hanya muncul untuk B0 dan R1 s7 (`logs/2026-10-10-grpo-run2.md:11`).
  - *Batch* R0/R1/R2 hanya dievaluasi pada *holdout* (`logs/2026-10-10-r0-r1-r2-batch.md:5,22`).
  - Dokumen diagnosis yang memotivasi R3/R4/R0-dyn (`docs/research/2026-10-11-results-diagnosis-and-improvements.md`, tabel §1) memakai angka *holdout*.
  - Bobot *reward* memang nilai bawaan (bab3:96). Namun, pengaturan pelatihan **dipilih** dari uji *smoke* dan *run* 2: lr 1e-5 → 2e-5, G 4 → 8, 64 → 256 pertanyaan (`configs/training/grpo_smoke.json` vs `grpo_run2.json`, `logs/2026-10-10-runpod-grpo-smoke.md:57-60`). Keputusan "*run 2 settings … are adopted for the R0/R1/R2 comparison*" (`logs/2026-10-10-grpo-run2.md:55`) diambil dari log yang melaporkan **baik** *holdout* maupun validasi (`logs/2026-10-10-grpo-run2.md:27-54`). `train_on_invalid: true` diaktifkan setelah B0 menunjukkan sekitar 43% episode tidak valid (`logs/2026-10-10-runpod-grpo-smoke.md:20`).
- **Mengapa penting:** ini adalah kontradiksi internal dalam satu bab. Penguji yang teliti akan menanyakannya. Pengaruh praktisnya mungkin kecil, karena pilihannya kasar (lr, G, jumlah data) dan tidak membedakan antar-*reward*. Tetapi klaim "tidak pernah" tidak bisa dipertahankan secara harfiah, dan sama sekali tidak berlaku untuk R3/R4/R0-dyn.
- **Perbaikan:**
  - Ganti bab3:41 dengan "Bobot *reward* tidak di-*tuning*. Pengaturan pelatihan (lr, G, jumlah pertanyaan) ditetapkan setelah uji *smoke* dan *run* 2, yang dievaluasi pada validasi dan *holdout*. Pengaturan ini sama untuk semua varian *reward*, sehingga tidak menguntungkan satu varian tertentu. Varian R3, R4, dan R0-dyn dirancang setelah melihat hasil *holdout* R0–R2 (Subbab 3.8), sehingga angka *holdout* ketiganya bukan estimasi tak bias."
  - Hapus klaim "seluruh pengaturan dipilih pada data validasi".

### M3. R0-dyn dengan satu pertanyaan per langkah setara dengan "R0 s8 yang dilatih lebih lama", dan *seed* 8 tidak menentukan *run* secara deterministik

- **Lokasi:** bab3:185, bab3:249, bab3:253-255.
- **Observasi 1 (kesetaraan):** pada *run* standar, grup varians-nol sudah dilewati tanpa pembaruan (`train_grpo.py:180`). *Dynamic sampling* di kode hanya menambah panjang jadwal menjadi 512 pertanyaan dan berhenti setelah 256 pembaruan (`train_grpo.py:154-167`, `r0_dynsample_seed8.json` `num_train_examples: 512`). Karena `select_train_ids` mengurutkan berdasarkan *hash* lalu memotong (`train_grpo.py:50-53`), 256 pertanyaan pertama R0-dyn **identik dan berurutan sama** dengan R0 s8. Saya memverifikasinya: 256/256 `question_id` sama urutannya di `metrics.jsonl` kedua *run*. Dengan *batch* = 1 pertanyaan, DAPO *dynamic sampling* (yang di DAPO menjaga ukuran *batch* tetap) tidak punya efek lain selain "lanjutkan melatih pada 256 pertanyaan tambahan". Hal yang diuji R0-dyn adalah "lebih banyak data/pembaruan", bukan perubahan *optimizer*.
- **Observasi 2 (non-determinisme):** konfigurasi, versi *library* (`run_metadata.json`: torch 2.11.0+cu128, transformers 4.57.6, GPU L40S, revisi model sama), urutan pertanyaan, dan *seed rollout* (`train_grpo.py:172`) sama.
  - Langkah 1, sebelum pembaruan apa pun, identik 8/8 (status dan *reward* setiap *rollout* sama).
  - Mulai langkah 2, setelah pembaruan pertama, *reward_mean* sudah berbeda.
  - Kode berubah di antara kedua *batch* (R0 s8 dari `fd25d37`, R0-dyn dari `1ebe62c` menurut `logs/2026-10-11-fixes-batch.md:16`). Namun `git diff fd25d37 1ebe62c -- scripts/train_grpo.py src/needle/training/grpo.py` hanya menambah penghitung `updates`, opsi `normalize_std` (bawaan `True`), dan kondisi berhenti. Jalur *sampling* dan pembaruan untuk konfigurasi bawaan tidak berubah, sehingga perbedaannya paling mungkin berasal dari non-determinisme numerik GPU saat pembaruan. Dalam 256 langkah pertama, R0 s8 melewati 128 grup sedangkan R0-dyn melewati 107. Jadi `seed 8` **tidak** menentukan *run* secara bit-per-bit. Pernyataan bab3:254 bahwa "*seed* 8 adalah *seed* tempat pelatihan *outcome-only* runtuh" memperlakukan keruntuhan sebagai sifat *seed*, padahal pengulangan dengan *seed* yang sama pun bisa berbeda lintasannya.
- **Mengapa penting:** interpretasi bab3:249 ("jika *dynamic sampling* menghilangkan jalan pintas … maka *optimizer*-lah yang terimplikasi") tidak terdukung. Perbedaan R0-dyn vs R0 s8 terkacaukan oleh (i) dua kali lipat data dan (ii) variasi *run-to-run* pada *seed* yang sama.
- **Perbaikan (tanpa *run* baru):**
  - Ubah bab3:185 menjadi: "Karena setiap langkah memuat satu pertanyaan dan grup varians-nol sudah dilewati pada *run* standar, implementasi ini setara dengan melanjutkan pelatihan R0 s8 pada 256 pertanyaan tambahan sampai 256 pembaruan tercapai."
  - Tambahkan ke bab3:180 dan bab3:254 bahwa pelatihan tidak deterministik secara bit, dengan bukti langkah 2 dan 128 vs 107 *skip*.
  - Turunkan klaim R0-dyn menjadi "apakah lebih banyak pembaruan informatif mengubah perilaku sitasi", bukan "*optimizer* vs *reward*".

### M4. Rencana statistik: tidak ada metrik primer tunggal, klaim "tanpa kehilangan akurasi" tanpa margin, dan *bootstrap seed* dengan n=3 terlalu optimistis

- **Lokasi:** bab3:29 (RQ2 "… tanpa kehilangan akurasi"), bab3:225 (aturan signifikansi), bab3:228 dan bab3:244 (RQ1 "setiap metode dibandingkan dengan B0" sebagai utama).
- **Observasi:**
  1. **Metrik primer.** bab3:212 sudah menetapkan EM atas 300 episode sebagai "angka akurasi utama", dan itu baik. Namun, untuk sisi *grounding* RQ2 ("mencegah jalan pintas") tidak ada metrik primer yang ditetapkan. Skrip menguji Δ*grounded* EM dan ΔJoint-EM sekaligus, ditambah tingkat sitasi (`peer_review_reanalysis.py:118-124`), sehingga ada beberapa kesempatan untuk "signifikan". "Jalan pintas" juga belum didefinisikan secara operasional (mis. tingkat sitasi < x%).
  2. **Non-inferioritas.** "Tanpa kehilangan akurasi" adalah klaim non-inferioritas. Selang yang memuat 0 tidak membuktikan "tidak ada kerugian" tanpa margin yang ditetapkan sebelumnya (mis. −3 poin EM).
  3. ***Bootstrap seed* n=3.** `hboot` (`peer_review_reanalysis.py:94-112`) me-*resample* 3 *seed* dengan pengembalian dari 3, yang hanya menghasilkan 27 kombinasi berurutan (10 multiset berbeda). *Bootstrap* dengan n sekecil ini dikenal meremehkan varians antar-*seed* (faktor (n−1)/n = 2/3 pada varians, ditambah distribusi yang sangat diskret). Aturan "signifikan jika CI tidak memuat 0" karena itu cenderung *anti-conservative*. M3 menunjukkan bahwa variasi *run-to-run* bahkan ada pada *seed* yang sama.
  4. **RQ1.** RQ1 memasukkan R3/R4 sebagai perbandingan "utama" walaupun keduanya *post hoc* (bab3:257).
- **Mengapa penting:** RQ2 adalah klaim inti tesis. Penguji statistik akan menanyakan "apa *endpoint* primernya?" dan "bagaimana Anda tahu akurasinya tidak turun?".
- **Perbaikan:**
  - Nyatakan satu metrik primer per RQ (mis. RQ2-*grounding*: *precision-grounded* EM; RQ2-akurasi: EM atas 300).
  - Untuk "tanpa kehilangan akurasi", tetapkan margin dan nyatakan dengan jujur bahwa margin itu ditetapkan setelah data (karena memang begitu). Atau ganti kalimatnya menjadi "tidak terdeteksi penurunan akurasi; selang [a, b] tidak menyingkirkan penurunan hingga |a| poin".
  - Laporkan juga selisih per *seed* berpasangan (R1 s*k* − R0 s*k*, k = 7, 8, 9), apakah ketiganya bertanda sama, sebagai pelengkap yang transparan.
  - Tambahkan satu kalimat keterbatasan *bootstrap* n=3.
  - Pindahkan R3/R4 vs B0 ke kelompok eksploratif.

## 4. Temuan Minor

### m1. *Retrieval recall* tertimbang per *supporting fact*, bukan per dokumen emas

- **Lokasi:** bab3:84 ("proporsi dokumen emas yang muncul …") dan bab3:49 ("menemukan 83% dokumen emas"). Kode: `rewards/__init__.py:110-122`, yaitu |fakta emas yang dokumennya terambil| / |fakta emas|.
- **Masalah:** jika satu dokumen emas memuat 2 fakta dan dokumen lain 1 fakta, mengambil dokumen pertama saja memberi 0,67, bukan 0,5. Tabel 3.1 kebetulan konsisten karena setiap dokumen punya satu fakta.
- **Perbaikan:** definisikan ulang sebagai "proporsi *supporting facts* emas yang dokumennya muncul pada hasil pencarian", lalu ubah "83% dokumen emas" menjadi "83% *supporting facts* terekspos".

### m2. Bukti mekanisme R0 (bab3:125-128) memakai *proxy* sitasi yang berbeda dari kata-katanya

- **Lokasi:** bab3:125-126 ("176 *rollout* mencoba menyitasi … 42 *rollout* valid tanpa sitasi"). Kode: `peer_review_reanalysis.py:151-158`. Log *rollout* tidak menyimpan daftar sitasi, sehingga "*valid with citations*" didefinisikan sebagai `citation_precision > 0`, yaitu menyitasi ≥ 1 kalimat **emas**.
- **Akibat:** rollout valid yang menyitasi hanya kalimat non-emas masuk ke kelompok "tanpa sitasi". Angkanya sendiri terverifikasi dari `results/2026-10-10-batch/r0_outcome_seed8/metrics.jsonl` (32 langkah pertama, 256 *rollout*):
  - 128 valid dan 90 kegagalan *evidence*.
  - Di antara yang valid, 86 memiliki `citation_precision > 0` (rata-rata *reward* 0,50) dan 42 tidak (0,357).

  Hasil ini mengonfirmasi bahwa teks memakai *proxy* tersebut. Namun labelnya keliru, dan perhitungan −0,27 vs +0,36 bergantung pada label itu.
- **Perbaikan:** ganti label menjadi "valid dengan ≥ 1 sitasi emas" dan "valid tanpa sitasi emas (termasuk tanpa sitasi sama sekali)". Atau hitung ulang dari `rollouts.jsonl` bila berkas itu tersedia (ditulis di `train_grpo.py:235-237`).

### m3. R4 tidak secara sederhana "mengubah ukuran langkah efektif"

- **Lokasi:** bab3:184.
- **Bukti:** *grad-norm clipping* 1,0 aktif pada sebagian besar langkah. Norma pra-*clip* > 1 terjadi pada 98% langkah R1 s8 (median 2,08), 86% R1 s9, 97,7% R0 s8, dan 59–72% R4 s7–s9 (`results/2026-10-11-fixes/runs/*/metrics.jsonl`, kolom `grad_norm`). AdamW juga hampir invarian terhadap penskalaan konstan gradien.
- **Akibat:** pembagian std per grup sebagian besar "terserap". Yang benar-benar berubah pada R4 adalah bobot relatif antar-langkah (grup dengan sebaran *reward* besar berkontribusi lebih besar terhadap momen Adam) dan porsi langkah yang lolos tanpa *clip*.
- **Perbaikan:** ganti kalimat "ukuran langkah efektif juga berubah" dengan uraian di atas, dan sebutkan bahwa *gradient clipping* aktif pada mayoritas langkah di semua *run*. Fakta ini juga relevan untuk menafsirkan R1 vs R0, karena gradien yang efektif dinormalisasi berarti pembaruan per langkah mirip besarnya.

### m4. Normalisasi std memperbesar selisih *reward* yang sangat kecil di VEGR

- **Lokasi:** bab3:10 dan bab3:159.
- **Masalah:** pada VEGR, grup yang hanya berbeda sebesar 0,033 (satu pencarian, `search_cost_weight` 0,1 × 1/3) tetap menghasilkan *advantage* berskala satuan (`grpo.py:42`). Sebagian dari "sekitar 240 pembaruan" R1 bisa berasal dari sinyal biaya pencarian semata. Jumlah pembaruan per *run* yang saya hitung: R1 s8 248, R1 s9 238, R0 s8 128.
- **Perbaikan:** tambahkan satu kalimat bahwa jumlah pembaruan R1 juga mencakup grup dengan selisih *reward* yang sangat kecil. Bila ingin kuat, laporkan distribusi `reward_std` per langkah (sudah ada di `metrics.jsonl`).

### m5. Metrik kualitas dirata-ratakan hanya atas episode selesai, dengan penyebut yang sangat berbeda

- **Lokasi:** bab3:214.
- **Masalah:**
  - B0 hanya punya 170 episode selesai, sedangkan *run* terlatih 278–300 (dihitung dari `episodes.jsonl`). Perbandingan presisi dan cakupan antar-*run* karena itu terkena bias seleksi.
  - Presisi juga bernilai 0 untuk sitasi kosong (`rewards/__init__.py:105-106`), sehingga "presisi sitasi" mencampur "tidak menyitasi" dengan "menyitasi salah".
  - Skrip sudah punya analisis himpunan bersama dan "*precision when cited*" (`peer_review_reanalysis.py:62-63`, `peer_review_reanalysis.py:83-91`), tetapi Bab 3 tidak mendefinisikannya.
- **Perbaikan:** definisikan di 3.7 (a) presisi saat menyitasi dan (b) metrik pada himpunan pertanyaan yang diselesaikan semua *run*.

### m6. Joint-F1 adalah aproksimasi, bukan metrik resmi HotpotQA

- **Lokasi:** bab3:219 vs `peer_review_reanalysis.py:44-50`. Komentar kode menyatakan "*approximated with answer F1 for both sides*".
- **Masalah:** HotpotQA resmi memakai P = P_ans × P_sup dan R = R_ans × R_sup. Teks menyebutnya "metrik gabungan HotpotQA" tanpa kualifikasi.
- **Perbaikan:** tulis "aproksimasi Joint-F1 (F1 jawaban dipakai pada sisi presisi dan *recall*)", atau hitung versi resmi dari P/R token jawaban.

### m7. *Holdout* = 300 baris valid pertama berkas dev, bukan sampel acak

- **Lokasi:** bab3:41. `holdout_v1_ids.json`: `selection.method = "first_valid_rows"`.
- **Masalah:** urutan berkas dev bisa tidak acak terhadap jenis pertanyaan (*bridge*/*comparison*) atau tingkat kesulitan.
- **Perbaikan:** laporkan komposisi jenis/level pada 300 pertanyaan tersebut dibandingkan seluruh dev, dan sebut ini sebagai keterbatasan generalisasi.

### m8. Kategori kegagalan tidak lengkap dan tata bahasa `ANSWER` lebih longgar dari yang tertulis

- **Lokasi:** bab3:56-62 dan bab3:66.
- **Kode:**
  - `collector.py:213-234` juga punya kategori RETRIEVAL (galat pencarian non-anggaran) dan ANSWER (jawaban kosong).
  - Regex `actions/__init__.py:25` membuat `CITATIONS[...]` opsional, sehingga `ANSWER[x]` tanpa blok sitasi tetap sah.
  - Sitasi duplikat ditolak sebagai galat *parse* atau protokol (`actions/__init__.py:108-110`).
- **Perbaikan:** tambahkan dua kategori tersebut (atau nyatakan bahwa keduanya digabung ke "protokol" dalam pelaporan). Tulis tata bahasanya sebagai `ANSWER[a]( CITATIONS[...])?` dan sebut aturan duplikat.

### m9. Pencocokan judul sitasi bersifat toleran, sedangkan teks dan *prompt* menyebut "persis"

- **Lokasi:** bab3:139 ("judul disalin persis") dan lampA:24. Kode: `actions/__init__.py:60-86`.
- **Masalah:** jika tidak ada kecocokan persis, judul dicocokkan setelah menormalkan huruf besar/kecil, Unicode, tanda baca, dan spasi. Mekanisme ini dipanggil di `collector.py:50-59`, sehingga sebagian sitasi yang "tidak persis" tetap valid.
- **Perbaikan:** satu kalimat di 3.3 tentang kanonikalisasi judul. Kanonikalisasi ini juga relevan untuk analisis kegagalan '&' di Bab 4.

### m10. Lampiran A menyebut baris *prompt* sebagai penyebab celah sitasi kosong

- **Lokasi:** lampA:65.
- **Masalah:** celah tersebut ada di *parser* (`actions/__init__.py:10-11,25`) dan lingkungan (`search.py:151`, *default* `citations=()`), terlepas dari isi *prompt*. *Prompt* hanya membuatnya eksplisit.
- **Perbaikan:** "Baris ini mengizinkan secara eksplisit; *parser* dan lingkungan juga menerima daftar kosong."

### m11. Reproduktibilitas: *commit* kode tidak tercatat dan konfigurasi evaluasi berisi `null`

- **Lokasi:** bab3:283-288.
- **Status:** `run_metadata.json` mencatat revisi model (`a09a3545…`) dan versi *library*, tetapi tidak mencatat *commit* git (diakui di bab3:288). Selain itu:
  - `configs/evaluation/shared_environment.json` berisi `dataset_manifest_sha256: null` dan `dataset_manifest_status: "pending_first_official_source_run"`.
  - `local_baseline.json` berisi `revision_status: "pin_before_first_local_run"`.
- **Perbaikan:** isi nilai-nilai tersebut, atau tandai di teks bahwa *hash* yang berlaku adalah yang ada di `splits.json` (`manifest_sha256 a0c78cb1…`) dan `run_metadata.json`. Sebutkan juga *commit* per *batch* dari log.

## 5. Nit

- **n1.** Teks pertanyaan pada bab3:98 ("*Which company formed the boy group that debuted with 2014 S/S?*") adalah parafrase, sedangkan teks aslinya ada di lampA:27. Pakai teks asli atau beri tanda parafrase.
- **n2.** lampA:3 menyebut *prompt* "lengkap", padahal *chat template* Qwen2.5 menyisipkan pesan *system* bawaan bila tidak ada pesan *system*. `local_policy.py:42-45` hanya mengirim satu pesan *user*. Perlu dicek pada *tokenizer* lokal, lalu disebut dalam satu kalimat.
- **n3.** bab3:172 mengatribusikan selisih *log-prob* 0,03–0,09 ke derau bf16 (angka tidak diverifikasi untuk 16 *run*). Hipotesis yang belum diuji: selisih itu muncul dari perbedaan jalur komputasi, yaitu mode `eval()` saat koleksi (`local_policy.py:113,146`) vs `train()` + *gradient checkpointing* saat pembaruan, dengan bobot yang sama. Cukup sebutkan bahwa bobotnya identik, sehingga selisih itu murni numerik.
- **n4.** bab3:10 "sekitar 130 pembaruan" untuk R0: R0 s8 = 128 dan R0 s7 = 149 (107 grup dilewati, dihitung dari `results/2026-10-11-fixes/runs/r0_outcome_seed7/metrics.jsonl`). Sebut rentangnya.
- **n5.** bab3:178 "256 langkah" tidak berlaku untuk R0-dyn (512 langkah, 250 pembaruan). Tambahkan "kecuali R0-dyn".
- **n6.** Pencarian berisi hanya tanda baca (mis. `SEARCH[???]`) lolos *parser*, karena `actions/__init__.py:127-132` hanya menolak *payload* kosong, tetapi memicu `ValueError` di `bm25.py:64-65`. `collector.py:200-202` hanya menangkap `SearchEnvironmentError`, sehingga kasus ini bisa menjadi *crash*, bukan kegagalan protokol. Ini kasus tepi yang tidak memengaruhi hasil, tetapi layak dicatat bila ditanya.
- **n7.** Pseudocode bab3:198-202 menjumlahkan *loss* tanpa pembagi. Kode membagi dengan `group_size` = 8 (`train_grpo.py:201`). Tulis `loss /= 8`.

## 6. Ringkasan kesesuaian teks ↔ kode

| Klaim Bab 3 | Kode | Status |
|---|---|---|
| Sitasi tak terambil = kegagalan bukti (bab3:61,88,121) | `search.py:169-181`: hanya korpus 10 dokumen | **Tidak sesuai** (M1) |
| *Split* sha256 "7:"+id, 0,95 (bab3:40) | `prepare_splits.py:145-151` | Sesuai |
| *Holdout* tidak dipakai memilih pengaturan (bab3:41) | log dan dokumen diagnosis | **Kontradiktif** (M2) |
| *Dynamic sampling* (bab3:185) | `train_grpo.py:154-167` | Sesuai secara mekanis, tafsirannya keliru (M3) |
| RetR = proporsi dokumen emas (bab3:84) | `rewards/__init__.py:110-122` | **Tidak sesuai** (m1) |
| *Advantage*, *skip*, rasio sekuens, satu *update*/grup (bab3:153-176) | `grpo.py`, `train_grpo.py` | Sesuai |
| Bobot R0/R2/R3/R4 (bab3:115-119, 184) | `configs/training/*.json`, `rewards/__init__.py:192-193` | Sesuai |
| Joint-F1 "HotpotQA" (bab3:219) | `peer_review_reanalysis.py:44` | Aproksimasi (m6) |
| Judul disalin persis (bab3:139) | `actions/__init__.py:60-86` | Lebih longgar (m9) |
| Evaluasi *greedy*, tanpa *retry* (bab3:208) | `evaluate_checkpoint.py:5,123,148-149`; `collector.py` tanpa *retry* | Sesuai |

## 7. Pertanyaan sidang yang mungkin muncul

1. "Lingkungan Anda benar-benar memeriksa bahwa dokumen yang disitasi pernah diambil? Tunjukkan baris kodenya." (M1). Siapkan jawaban jujur beserta jumlah episode yang terdampak.
2. "R0-dyn melatih 256 pertanyaan yang sama dengan R0 s8 lalu 256 lagi. Apa bedanya dengan sekadar melatih R0 lebih lama? Dan mengapa dua *run* dengan *seed* 8 sudah berbeda di langkah kedua?" (M3)
3. "Apa *endpoint* primer untuk RQ2, dan atas dasar apa Anda menyatakan VEGR 'tanpa kehilangan akurasi'? Berapa margin yang Anda anggap tidak bermakna?" (M4)
4. "Dengan hanya tiga *seed*, seberapa dapat dipercaya selang *bootstrap* yang me-*resample seed*?" (M4)
5. "Karena R1 vs R0 mengubah enam komponen sekaligus, eksperimen minimal apa yang akan memisahkan efek 'memberi imbalan sitasi' dari efek 'sinyal lebih padat'?" Jawaban yang baik: satu varian R0 + Cov/Prec saja, sebagai saran Bab 5, tanpa *run* baru.

## 8. Prioritas tindakan (semuanya bisa dikerjakan tanpa GPU)

1. M1: perbaiki definisi kegagalan bukti di bab3:61, 88, 121 dan lampA:26/65, serta tambahkan kalimat keterbatasan beserta hitungan batas bawahnya.
2. M2: tulis ulang bab3:41.
3. M3: tulis ulang bab3:185, 249, dan 254, dan tambahkan catatan non-determinisme di bab3:180.
4. M4: tetapkan metrik primer, ubah kalimat non-inferioritas, tambahkan selisih per *seed* dan catatan *bootstrap* n=3.
5. Minor m1–m11 dan nit sesuai waktu.
