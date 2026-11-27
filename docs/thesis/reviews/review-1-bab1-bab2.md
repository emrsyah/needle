# Review 1: Bab 1, Bab 2, dan Daftar Pustaka

> **Status: draf kerja.** Review ini dibuat dengan bantuan AI. Penulis TA harus memeriksa sendiri setiap lokasi, kutipan, dan klaim sebelum dipakai. Review ini tidak memberi vonis lulus atau tidak lulus. Isinya hanya temuan dan saran perbaikan.

**Cakupan:** `bab1-pendahuluan.md`, `bab2-kajian-pustaka.md`, `daftar-pustaka.md`. Bab 3–5 hanya dibaca sekilas untuk memeriksa keselarasan. Temuan yang menyangkut bab lain diberi label **[lintas-bab]**, tetapi perbaikannya tetap diarahkan ke baris di Bab 1/Bab 2.

**Standar penilaian:** tugas akhir S1. Kebaruan bukan tujuan, dan kesederhanaan memang disengaja. Kritik di sini menyangkut **akurasi klaim, konsistensi, dan kejujuran framing**, bukan kurangnya kebaruan atau skala.

**Format lokasi:** `file:baris`, mengacu pada berkas di `docs/thesis/chapters-id/`.

## Paper yang diperiksa langsung

Paper berikut dibaca dari teksnya melalui alphaXiv atau web:

| Ref | Yang diperiksa | Hasil |
|---|---|---|
| [2] Search-R1 | Reward, algoritma, hardware, seed | Reward hanya EM (§3.4: "*we do not incorporate format rewards*"). Aksi yang tidak valid dibalas pesan "*My action is not correct. Let me rethink.*" (Algorithm 1), bukan −1. PPO adalah algoritma default, dan GRPO dibandingkan di §5.1. Kalimat "*a single node with 8 H100 GPUs*" (Lampiran B.2) terkonfirmasi. Tidak ada laporan seed atau variansi pada halaman yang terbaca (hlm. 8 dan 10–14 tidak ikut terbaca). |
| [4] DeepSeek-R1 | Data bibliografi | *Nature* 645(8081):633–638, 2025, doi:10.1038/s41586-025-09422-z. Judul: "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning". |
| [7] CaRR | Arti *shortcut*, judge | *Shortcut* berarti agen hanya memakai beberapa hop terakhir, mengabaikan constraint lain, dan memanggil tool lebih sedikit. Judge LLM (DeepSeek-v3.2) dipakai untuk reward outcome maupun rubrik. Format error mendapat reward **0**. Evaluasi diulang 3 kali. |
| [19] AutoRefine | Hardware, seed, reward | "*8 NVIDIA A100-80GB GPUs*" (Lampiran B.1) dan pengecekan 3 seed di Lampiran C.2 terkonfirmasi. Reward retrieval = jawaban emas muncul di blok `<refine>`. Tabel 8: reward atas *retrieved documents* hanya memberi kenaikan kecil (avg 0,384 vs 0,371 untuk *answer-only*). |
| [24] StepSearch | Reward, kontrol | Reward langkah = *information gain* (cosine TF-IDF terhadap dokumen emas) dikurangi penalti redundansi, ditambah reward kata kunci. Ada baseline PPO/GRPO dengan reward F1 jawaban saja pada setup yang sama (Tabel 2). Pelatihan memakai 16 GPU H800. Tidak ada informasi seed pada halaman yang terbaca. |
| [26] Huang dkk. 2024 | Jenis reward, seed | Reward sitasi dihitung dengan **model NLI** (TRUE T5-11B; attrscore-flan-t5-xl saat RL), jadi bukan aturan. Lampiran C: "*We repeat each RL experiment 3 times, each with a different seed*". Pembanding: *holistic reward* (§2.3 tidak dibaca, jadi jangan disamakan dengan *outcome-only*). |
| [27] PROGRESS | Judge, venue | Pencocokan query untuk reward cakupan dilakukan oleh **LLM judge** (Qwen2.5-72B-Instruct). Pelatihan memakai 8 GPU A100. Di arXiv tercatat sebagai cs.AI (2 Agu 2026). Format naskahnya bergaya Interspeech, tetapi status diterima atau belum **tidak terkonfirmasi** dari halaman yang terbaca. |

**Belum diperiksa:** [1], [3], [5], [6], [8]–[18], [20]–[23], [25], [28]–[33]. Untuk referensi klasik ([1], [3], [5], [6], [9]–[18], [25], [33]) karakterisasinya tampak wajar. Paper 2026 ([8], [28]–[32]) sebaiknya dicek ulang oleh penulis sendiri.

## Ringkasan

Bab 1 dan Bab 2 sudah **jujur dan tertata**. Hal-hal yang sudah baik:

- Batasan eksplisit tentang korpus tertutup (bab1:42).
- Daftar "hal yang tidak diklaim" (bab1:46).
- RQ3 diberi label eksploratif (bab1:28).
- Komponen VEGR diakui bukan hal baru (bab1:55, bab2:94).
- Klaim hardware dikutip langsung dari lampiran paper.
- Penomoran sitasi rapi (lihat bagian akhir).

Masalah utamanya ada empat:

1. **Baseline R0 disebut setara Search-R1, padahal tidak.** Perbedaannya justru mekanisme yang menyebabkan hilangnya sitasi.
2. **RQ2 memakai istilah "jalan pintas" dan "tanpa kehilangan akurasi".** "Jalan pintas" tidak didefinisikan, dan "tanpa kehilangan akurasi" tidak didukung statistik tesis sendiri.
3. **Beberapa paper di Tabel 2.1 (G2) dikarakterisasi kurang tepat.**
4. **Klaim tentang penggunaan *holdout* saling bertentangan antar-bab.**

Semuanya bisa diperbaiki dengan mengubah kata-kata. Tidak perlu eksperimen baru.

---

## MAJOR

### M1. R0 disebut mengikuti reward Search-R1, padahal berbeda tepat pada mekanisme yang menjelaskan hasil

- **Lokasi:**
  - bab2:61: "*Baseline outcome-only* pada tugas akhir ini (R0) mengikuti rancangan *reward* yang sama."
  - bab1:7: "Search-R1 menggunakan **GRPO** [3]".
  - [lintas-bab] bab3:117: "Varian ini mencerminkan agen pencarian dengan *outcome reward* seperti Search-R1 [2]."
- **Masalah:**
  - Reward Search-R1 adalah `r = EM(a_pred, a_gold)` tanpa format reward (Search-R1 §3.4). Aksi yang tidak valid tidak diberi −1. Model hanya menerima pesan "rethink" (Algorithm 1). Algoritma defaultnya PPO, dan GRPO dipakai sebagai pembanding.
  - R0 di tesis ini = EM **ditambah penalti −1 untuk episode tidak valid, termasuk sitasi yang gagal validasi**.
  - Tesis sendiri menjelaskan bahwa hilangnya sitasi disebabkan oleh penalti −1 itu, ditambah tidak adanya imbalan untuk sitasi benar (bab3:121–128 [lintas-bab]; bab2:98).
  - Jadi temuan inti (2 dari 3 seed berhenti menyitasi) berlaku untuk "*EM + penalti validitas −1 + environment menerima jawaban tanpa sitasi*". Temuan ini tidak berlaku untuk "*reward outcome-only ala Search-R1*" secara umum.
- **Kenapa penting di sidang:** Penguji yang pernah membaca Search-R1 akan bertanya, "Search-R1 tidak memberi −1, jadi apakah keruntuhan ini artefak desain baseline Anda?" Jika Bab 2 menyebut R0 "sama", penulis terlihat salah mengutip. Jika perbedaannya diakui sejak awal, ini justru menjadi poin analisis yang kuat.
- **Perbaikan:**
  - bab2:61, ganti kalimat terakhir dengan: "*Baseline outcome-only* pada tugas akhir ini (R0) mengadopsi gagasan *reward* berbasis EM dari Search-R1, dengan satu perbedaan penting: episode yang tidak valid, termasuk sitasi yang merujuk kalimat yang tidak pernah diambil, diberi *reward* −1, sedangkan Search-R1 tidak memakai *reward* format dan hanya meminta model 'berpikir ulang' ketika aksinya tidak valid [2]. Perbedaan ini relevan untuk Bab 4, karena penalti inilah yang mendorong R0 berhenti menyitasi."
  - bab1:7, ganti "Search-R1 menggunakan **GRPO** [3], yaitu algoritma RL…" dengan: "Search-R1 dapat dilatih dengan PPO maupun **GRPO** [3]; GRPO adalah algoritma RL…"
  - Tambahkan satu butir di Batasan bab1:38–47: "**Baseline R0.** R0 bukan replikasi Search-R1; R0 memakai EM ditambah penalti −1 untuk episode tidak valid (Subbab 3.4), sehingga temuan tentang hilangnya sitasi berlaku untuk rancangan *reward* tersebut."

### M2. RQ2 memakai istilah yang tidak didefinisikan dan klaim akurasi yang tidak didukung statistik tesis sendiri

- **Lokasi:** bab1:25 (RQ2): "Apakah *verifiable evidence-grounded reward* (VEGR) **mencegah perilaku jalan pintas** dari *reward* berbasis jawaban saja, **tanpa kehilangan akurasi**?" Juga bab1:56: "Paket VEGR lengkap mencapai **akurasi yang sama** (+1,0 poin EM, SK 95% … [−3,6; +5,4])".
- **Masalah 1, "jalan pintas" tidak didefinisikan di Bab 1 dan maknanya dilunakkan di Bab 2:**
  - Bab 4 memakai "jalan pintas" untuk perilaku berhenti menyitasi ("jalan pintas tanpa sitasi", bab4:223 [lintas-bab]), tetapi Bab 1 tidak pernah mendefinisikannya.
  - Bab 2 sendiri menyatakan perilaku R0 "lebih tepat digambarkan sebagai konsekuensi rancangan *reward* yang dapat diprediksi daripada sebagai eksploitasi yang tidak terduga" (bab2:98).
  - Bab 4 menyebut pencarian berulang R0 s7 "bukan jalan pintas" (bab4:108 [lintas-bab]), jadi pembaca perlu tahu perilaku mana yang dimaksud.
  - Istilah "jalan pintas" juga diambil dari CaRR (bab2:82, bab1:17). Di CaRR artinya mengabaikan hop atau constraint dan memanggil tool lebih sedikit, bukan berhenti menyitasi. Lihat juga m3.
  - Selain itu, VEGR memberi reward langsung pada sitasi yang benar. Jadi "VEGR mencegah agen berhenti menyitasi" sebagian besar benar *by construction*. Penguji bisa menyebut ini pertanyaan yang jawabannya sudah pasti.
- **Masalah 2, "tanpa kehilangan akurasi" adalah klaim non-inferioritas tanpa margin:**
  - CI R1−R0 = [−3,6; +5,4]. Data ini masih konsisten dengan penurunan hingga 3,6 poin EM.
  - Bab 4 sendiri menulis bahwa hasil tidak signifikan "bukan bukti bahwa kedua metode setara" ([lintas-bab] bab4:198) dan bahwa akurasinya "secara statistik tidak dapat dibedakan" (bab4:335).
  - Namun bab1:56, bab4:100 ("Akurasinya sama"), bab4:126 ("menyamai"), dan bab5:13 ("akurasi yang sama") memakai bahasa ekuivalensi.
- **Kenapa penting:** RQ adalah hal pertama yang diuji di sidang. Jika istilah di RQ tidak terdefinisi dan klaim akurasinya bertentangan dengan Bab 4, penguji akan menganggap rumusan masalahnya belum matang.
- **Perbaikan:**
  - bab1:25, ganti RQ2 dengan: "**RQ2.** Dibandingkan dengan *reward* berbasis jawaban saja (R0), apakah VEGR menghasilkan agen yang lebih ter-*grounding* (sitasi yang benar terhadap *supporting facts* emas), dan apakah terdapat perbedaan akurasi yang dapat dideteksi di antara keduanya?"
  - Jika istilah "jalan pintas" tetap dipakai, definisikan secara operasional di bab1:17, misalnya "yaitu menjawab tanpa menyertakan sitasi bukti (tingkat sitasi ≈ 0)", dan pakai definisi itu secara konsisten.
  - bab1:56, ganti "mencapai akurasi yang sama" dengan "tidak menunjukkan perbedaan akurasi yang dapat dideteksi (+1,0 poin EM, SK 95% [−3,6; +5,4]; data masih konsisten dengan selisih −3,6 hingga +5,4 poin)". Lakukan perubahan yang sama di abstrak, bab4:100, bab4:126, dan bab5:13 [lintas-bab].

### M3. Karakterisasi beberapa paper di Tabel 2.1 (G2) kurang tepat

- **Lokasi:**
  - bab2:109 (baris G2 di Tabel 2.1): "*reward* berbasis aturan (Huang dkk. [26]; StepSearch; HERALD) dilaporkan tanpa kontrol semacam itu".
  - bab2:85–87: PROGRESS [27] ditempatkan di paragraf "*Reward* sitasi berbasis aturan dan berbasis model tanpa LLM *judge*".
  - bab1:18 (G2): "Alternatif berbasis aturan sudah ada (Bab 2), tetapi jarang dibandingkan…"
- **Masalah:**
  - **Huang dkk. [26]:** reward sitasinya dihitung dengan model NLI (TRUE T5-11B, lalu attrscore-flan-t5-xl saat RL), jadi bukan **berbasis aturan**. Paper itu juga menulis "*We repeat each RL experiment 3 times, each with a different seed*" dan membandingkan dengan *holistic reward* (isi *holistic reward* ini belum diperiksa karena §2.3 tidak dibaca). Jadi "tanpa kontrol semacam itu" tidak tepat: ada multi-seed dan ada pembanding, meskipun setting-nya bukan agen pencarian *multi-turn*.
  - **StepSearch [24]:** reward langkahnya memang berbasis aturan (TF-IDF terhadap dokumen emas). Namun StepSearch **punya** kontrol PPO/GRPO dengan reward F1 jawaban saja pada setup yang sama (Tabel 2) dan ablasi "w/o steps reward". Yang tidak terlihat dari halaman yang terbaca hanyalah pengulangan seed.
  - **HERALD [31]:** bab2:92 menyebutnya "evaluasi, bukan metode pelatihan", tetapi bab2:109 mendaftarkannya sebagai *reward* yang "dilaporkan tanpa kontrol". Ini kontradiksi internal.
  - **PROGRESS [27]:** pencocokan query dilakukan oleh LLM judge (Qwen2.5-72B), jadi tidak cocok berada di bawah judul "tanpa LLM *judge*". Selain itu bab2:85 menyebut kelompok ini "*preprint* tahun 2026", sementara daftar pustaka [27] menulis "dalam *Interspeech*, 2026" dan [30] menulis "dalam … EMNLP, 2026" (daftar-pustaka:55, :61).
- **Kenapa penting:** Tabel 2.1 adalah dasar klaim celah penelitian. Penguji yang mengenal satu saja dari paper ini, terutama Huang dkk. 2024 yang cukup dikenal, akan melihat bahwa celahnya dirumuskan lebih lebar daripada yang didukung literatur. Ini lebih merusak kredibilitas daripada mengakui bahwa celahnya sempit.
- **Perbaikan:**
  - Persempit G2 di bab1:18 dan bab2:109 menjadi celah yang memang tidak terbantah: "*Reward* bukti **berbasis aturan pada tingkat kalimat** untuk agen pencarian ***multi-turn***, dibandingkan dengan kontrol *outcome-only* **pada beberapa *seed*** dalam pengaturan yang selebihnya identik, jarang dilaporkan."
  - Revisi isi sel "Bukti dari literatur" G2, misalnya: "CaRR, STAMP, dan PROGRESS memakai LLM *judge*/verifikator; Huang dkk. [26] memakai model NLI dan 3 *seed*, tetapi bukan untuk agen pencarian *multi-turn*; StepSearch berbasis aturan dan memiliki kontrol *outcome-only*, tetapi tidak melaporkan pengulangan *seed*." Penulis perlu memverifikasi sendiri dua hal sebelum memakai kalimat ini: jenis pembanding *holistic reward* pada [26] (§2.3), dan tidak adanya laporan *seed* di StepSearch (review ini hanya membaca sebagian halaman).
  - Hapus HERALD dari sel tersebut, atau tulis "HERALD mengaudit *reward* semacam ini".
  - bab2:85: ubah judul paragraf menjadi "*Reward* bukti dan *retrieval* tanpa LLM *judge* atau dengan model kecil", lalu pindahkan PROGRESS ke kelompok berbasis LLM *judge* (bersama CaRR/STAMP). Cara lain: tambahkan keterangan "(memakai LLM *judge* untuk mencocokkan *query*)".

### M4. [lintas-bab] Klaim tentang penggunaan *holdout* saling bertentangan, dan rumusan Bab 1 kurang spesifik

- **Lokasi:**
  - bab1:28: "RQ3 bersifat eksploratif, karena varian-variannya sebagian dirancang setelah **hasil pertama** diperoleh".
  - bab3:41: "*Holdout* tidak pernah digunakan untuk memilih pengaturan."
  - bab3:228, bab4:198, bab4:358: R3, R4, dan R0-dyn "dirancang setelah hasil ***holdout*** diketahui". Bandingkan juga bab3:257: ditambahkan "setelah hasil R0/R1/R2 dianalisis".
- **Masalah:** Bab 3:41 menyatakan *holdout* tidak pernah dipakai untuk memilih pengaturan, padahal tiga varian dirancang sebagai respons terhadap hasil *holdout* R0/R1/R2. Bab 1 hanya menyebut "hasil pertama", sehingga pembaca tidak tahu bahwa yang dimaksud adalah hasil *holdout*.
- **Kenapa penting:** Kontaminasi *holdout* adalah hal yang biasa dicari penguji. Pengungkapannya sudah ada di Bab 3–4, jadi ini bukan masalah integritas. Namun kalimat absolut di bab3:41 membuat tesis terlihat kontradiktif.
- **Perbaikan:**
  - bab1:28, ganti dengan: "RQ3 bersifat eksploratif, karena R3, R4, dan R0-dyn dirancang setelah hasil *holdout* R0/R1/R2 diketahui (Subbab 3.8), sehingga *holdout* tidak lagi sepenuhnya 'buta' untuk perbandingan RQ3."
  - bab3:41: ganti "tidak pernah digunakan untuk memilih pengaturan" dengan "tidak digunakan untuk memilih *hyperparameter* atau *checkpoint*; namun, hasil *holdout* R0/R1/R2 memotivasi rancangan varian eksploratif R3, R4, dan R0-dyn (Subbab 3.8)."

---

## MINOR

### m0. Pemetaan G3/G4 ke rancangan hanya ada di Tabel 2.1, dan Tujuan 3 tidak sama dengan RQ3

- **Lokasi:**
  - bab1:15–20 (empat celah G1–G4), bab1:24–26 (tiga RQ), bab1:36 (Tujuan 3: "mengubah *grounding*, cakupan bukti, dan stabilitas pelatihan").
  - bab1:19 (G3): "Belum jelas apakah **efek yang sama** dapat dikaji pada satu GPU."
  - bab2:110–111 (Tabel 2.1, baris G3 dan G4).
- **Masalah:**
  - Ada empat celah tetapi hanya tiga RQ. Tabel 2.1 (bab2:110–111) sudah menunjukkan bahwa G3 dan G4 dijawab lewat **desain** (satu GPU; 3 *seed*), tetapi §1.2 tidak menyatakannya, sehingga pembaca Bab 1 melihat dua celah tanpa RQ.
  - Rumusan G3, "efek yang sama dapat dikaji pada satu GPU", menyiratkan bahwa efek dari setting multi-GPU (Wikipedia penuh, E5, PPO) direproduksi. Padahal tesis membuat tugasnya **jauh lebih kecil**: 10 paragraf tertutup, 256 pertanyaan, dan LoRA (bab1:42–44). Jadi satu GPU cukup karena tugasnya dikecilkan, bukan karena efek yang sama terbukti bisa dikaji dengan sumber daya lebih kecil.
  - Tujuan 3 menyebut "stabilitas pelatihan", tetapi RQ3 (bab1:26) menyebut "*grounding* dan perilaku pencarian". Jawaban RQ3 di §4.6 [lintas-bab] juga hampir tidak membahas perilaku pencarian untuk R2–R4.
- **Kenapa penting:** Penguji biasanya memeriksa rantai celah → RQ → tujuan → jawaban dari Bab 1 saja. Celah yang tidak punya RQ di Bab 1 akan dipertanyakan, walaupun jawabannya ada di Bab 2.
- **Perbaikan:**
  - Setelah bab1:20, tambahkan (mengikuti Tabel 2.1): "G1 dan G2 dijawab oleh RQ1–RQ3. G3 dan G4 bukan pertanyaan penelitian tersendiri, melainkan dijawab melalui rancangan eksperimen: seluruh *run* dilatih pada satu GPU (G3), dan setiap metode utama dilatih dengan tiga *seed* serta dilaporkan dengan SK yang memperhitungkan *seed* (G4)."
  - bab1:19, ubah G3 menjadi kendala praktis yang jujur: "**G3, hambatan komputasi.** … Penelitian ini tidak mereproduksi setting tersebut, tetapi menyederhanakan tugas (korpus tertutup, LoRA, 256 pertanyaan) agar perbandingan *reward* dapat dilakukan dengan anggaran mahasiswa pada satu GPU."
  - Selaraskan Tujuan 3 (bab1:36) dengan RQ3 (bab1:26): pilih "*grounding*, cakupan bukti, perilaku pencarian, dan variansi antar-*seed*", lalu pakai daftar yang sama di keduanya.

### m1. Kontribusi 2 menonjolkan metrik yang paling longgar dan tidak menyebut hasil yang tidak signifikan

- **Lokasi:** bab1:56 ("**24,4 poin lebih tinggi pada *precision-grounded* EM**").
- **Masalah:**
  - *Precision-grounded* EM longgar terhadap sitasi yang hilang (bab3:216 [lintas-bab]).
  - R0 s8 dan s9 otomatis bernilai 0 pada metrik ini karena tidak menyitasi sama sekali.
  - Metrik standar HotpotQA (Joint-EM) memberi R1−R0 = +3,9 [−4,0; +11,4], tidak signifikan (bab4:102).
  - VEGR juga tidak menaikkan cakupan dibanding B0 (bab4:345).
  - Semua ini sudah diakui di Bab 4–5, tetapi tidak muncul di Bab 1, padahal penguji sering hanya membaca Bab 1 dan abstrak dengan teliti.
- **Kenapa penting:** Jika penguji menemukan Joint-EM yang tidak signifikan di Bab 4 setelah membaca klaim "+24,4" di Bab 1, kesannya penulis memilih metrik yang paling menguntungkan.
- **Perbaikan:** Tambahkan di akhir butir 2 (bab1:56): "Pada metrik gabungan HotpotQA yang lebih ketat (Joint-EM), selisihnya (+3,9 poin, SK [−4,0; +11,4]) tidak signifikan, dan VEGR tidak menaikkan cakupan bukti dibandingkan dengan model tanpa pelatihan."

### m2. Batasan Bab 1 belum memuat dua keterbatasan penting yang sudah ada di Bab 4/5

- **Lokasi:** bab1:38–49. Bandingkan dengan [lintas-bab] bab4:351, bab4:352, dan bab5:43.
- **Masalah:**
  - Tumpang-tindih *reward* dan metrik: VEGR dilatih pada *supporting facts* emas yang sama dengan yang dipakai metrik *grounding* (Goodhart).
  - Tidak ada *baseline* berbasis *prompt few-shot*, padahal RQ1 "sebagian besar pembelajaran protokol".
  - Keduanya sudah diungkap di Bab 4 dan Bab 5. Bab 1:49 juga sudah merujuk Bab 5, jadi ini cukup ditampilkan lebih awal.
- **Kenapa penting:** Kedua hal ini adalah kritik paling jelas terhadap RQ1 dan RQ2. Lebih baik penguji membacanya dari penulis di Bab 1 daripada menemukannya sendiri.
- **Perbaikan:** Tambahkan dua butir singkat di bab1:47:
  - "**Metrik dan *reward*.** *Grounding* diukur terhadap *supporting facts* emas yang sama dengan yang diberi *reward* oleh VEGR."
  - "**Baseline *prompt*.** Tidak ada *baseline prompt few-shot*; B0 adalah model yang sama dengan *prompt* yang sama tanpa pelatihan."

### m3. CaRR dan G1: "jalan pintas" CaRR berbeda dengan berhenti menyitasi

- **Lokasi:** bab2:82: "Temuan ini **sejalan** dengan perilaku jalan pintas yang dilaporkan pada Bab 4". bab1:17: "Penelitian terbaru melaporkan perilaku jalan pintas (*shortcut*) **seperti ini**".
- **Masalah:**
  - Di CaRR, *shortcut* berarti agen menjawab dari beberapa hop terakhir tanpa memverifikasi constraint lain, dengan pemanggilan tool lebih sedikit dan sitasi halaman lebih sedikit.
  - Di tesis ini, agen tidak berhenti mencari (R0 s9 tetap rata-rata 2,27 pencarian), tetapi berhenti **menyitasi**.
  - Bab 4:327 [lintas-bab] sudah memakai kata yang lebih tepat: "pola yang **berkaitan**".
- **Kenapa penting:** CaRR adalah satu-satunya bukti literatur untuk G1. Jika penguji mengenal CaRR, menyamakan dua perilaku yang berbeda akan terlihat sebagai salah baca.
- **Perbaikan:** Di bab2:82, ganti "sejalan dengan" menjadi "berkaitan dengan, tetapi tidak identik dengan,". Di bab1:17, ganti "seperti ini" menjadi "yang serupa, yaitu jawaban benar yang tidak didukung bukti lengkap [7]".

### m4. Search-R1 "melatih setiap konfigurasi hanya satu kali" lebih kuat daripada buktinya

- **Lokasi:** bab1:20 dan bab2:111 (Tabel 2.1, G4): "Search-R1 dan CaRR melatih setiap konfigurasi (hanya) satu kali". Bandingkan dengan bab2:72: "Search-R1 tidak melaporkan *seed* maupun variansi".
- **Masalah:** "Tidak melaporkan" tidak sama dengan "hanya melatih satu kali". bab2:72 sudah memakai rumusan yang benar, tetapi bab1:20 dan Tabel 2.1 menguatkannya.
- **Kenapa penting:** Klaim tentang apa yang *tidak* dilakukan paper lain mudah dibantah. Satu kalimat yang terlalu kuat membuat penguji meragukan karakterisasi literatur lainnya.
- **Perbaikan:** Samakan bab1:20 dan bab2:111 dengan bab2:72. Untuk bab1:20 → "Search-R1 tidak melaporkan *seed* maupun variansi, dan CaRR melatih setiap konfigurasi satu kali (Bab 2)…". Penulis perlu memverifikasi sendiri klaim tentang CaRR. Halaman yang terbaca hanya menunjukkan bahwa evaluasinya diulang 3 kali.

### m5. AutoRefine hanya mendukung RetR secara lemah

- **Lokasi:** bab2:63: "Hal ini mendukung gagasan yang juga digunakan pada VEGR, yaitu bahwa memberi *reward* pada kualitas pencarian antara membantu…"
- **Masalah:** Reward retrieval AutoRefine diberikan pada blok `<refine>`, yaitu ringkasan model, saat jawaban emas muncul di sana. Tabel 8 AutoRefine menunjukkan reward pada *retrieved documents* hanya memberi kenaikan kecil (avg 0,384 vs 0,371 untuk *answer-only*). Komponen RetR pada VEGR adalah reward pada dokumen yang diambil. Varian ini masih sedikit lebih baik daripada *answer-only*, tetapi kenaikannya jauh lebih kecil daripada reward pada `<refine>`.
- **Kenapa penting:** Mengutip paper sebagai dukungan, padahal tabelnya sendiri menunjukkan efek yang kecil untuk bentuk yang Anda pakai, mudah dipatahkan oleh penguji yang membaca AutoRefine.
- **Perbaikan:** bab2:63, ganti kalimat terakhir dengan: "AutoRefine menunjukkan bahwa *reward* antara dapat membantu, tetapi juga melaporkan bahwa *reward* langsung pada dokumen yang diambil hanya memberi kenaikan kecil dibandingkan dengan *reward* pada langkah *refine* [19, Lampiran C.3]. VEGR memakai bentuk yang pertama (*retrieval recall* atas dokumen yang diambil)."

### m6. DeepSeek-R1: deskripsi reward dan entri pustaka [4]

- **Lokasi:** bab2:44 ("*Reward* yang digunakan memeriksa akurasi jawaban dan format keluaran, tanpa *reward model* yang dipelajari") dan daftar-pustaka:9.
- **Masalah:**
  - Deskripsi reward itu tepat untuk tahap R1-Zero dan RL penalaran. Pipeline lengkap DeepSeek-R1 juga memiliki tahap untuk tugas umum. Penulis perlu memverifikasi sendiri detailnya di paper Nature.
  - Entri [4] tidak lengkap (judul hanya "DeepSeek-R1", tanpa volume, halaman, atau DOI).
- **Kenapa penting:** Entri pustaka yang tidak lengkap biasanya ditandai pembimbing atau penguji sebagai kesalahan format. Deskripsi reward yang terlalu umum juga bisa dikoreksi oleh penguji yang membaca paper aslinya.
- **Perbaikan:**
  - bab2:44: "…**DeepSeek-R1-Zero** dilatih dengan GRPO dan *reward* berbasis aturan yang memeriksa akurasi jawaban dan format…"
  - daftar-pustaka:9 → `[4] DeepSeek-AI, "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning," *Nature*, vol. 645, no. 8081, hlm. 633–638, 2025. doi:10.1038/s41586-025-09422-z.` Data ini sudah diverifikasi via PubMed dan Nature. Jika format kampus meminta nama penulis pertama, gunakan "D. Guo dkk." sesuai PubMed.

### m7. "83% dokumen emas" dihitung hanya atas episode yang selesai

- **Lokasi:** bab1:42: "Model tanpa pelatihan sudah mengambil 83% dokumen emas (*retrieval recall* 0,83)". Juga bab5:38.
- **Masalah:** Menurut konvensi Bab 4 (bab4:30, Tabel 4.1), *retrieval recall* B0 = 0,829 dirata-ratakan hanya atas **170 dari 300** episode yang selesai. Kalimat di Bab 1 terbaca seolah berlaku untuk semua pertanyaan.
- **Kenapa penting:** Angka ini dipakai untuk membuktikan bahwa tugasnya mudah. Jika penyebutnya tidak jelas, penguji bisa menanyakan berapa *recall* atas 300 pertanyaan.
- **Perbaikan:** "Pada episode yang selesai, model tanpa pelatihan sudah mengambil rata-rata 83% dokumen emas (*retrieval recall* 0,83)…"

### m8. Referensi yang belum ada untuk konsep yang dipakai

- **Lokasi:** bab1:5 dan bab2:13 (RAG tanpa sitasi); bab1:35, bab1:56, bab2:100 (bootstrap dan Clopper–Pearson tanpa sitasi metode); [lintas-bab] bab3:172 (algoritma yang benar-benar dijalankan adalah "REINFORCE dengan *baseline* rata-rata grup", tetapi Bab 2 hanya membahas PPO dan GRPO).
- **Masalah:** Konsep inti dipakai tanpa rujukan primer. Bab 2 juga tidak menyiapkan pembaca untuk fakta bahwa *clipping* PPO tidak pernah aktif.
- **Kenapa penting:** Kajian pustaka dinilai dari apakah ia menyiapkan semua konsep yang dipakai di Bab 3–4. Metode statistik dan algoritma yang benar-benar dijalankan tanpa rujukan akan ditanyakan.
- **Perbaikan:** Tambahkan rujukan berikut **setelah diverifikasi sendiri**:
  - RAG: Lewis dkk., 2020, NeurIPS.
  - REINFORCE: Williams, 1992, *Machine Learning*.
  - Opsional, RLOO: Ahmadian dkk., 2024.
  - Bootstrap: Efron & Tibshirani, 1993.
  - Clopper & Pearson, 1934, *Biometrika*.

  Di bab2:38, tambahkan satu kalimat: "Bentuk paling sederhana dari *policy gradient* adalah REINFORCE [Williams 1992], yang mengurangi variansi dengan mengurangkan sebuah *baseline* dari *reward*; GRPO dapat dipandang sebagai REINFORCE dengan *baseline* rata-rata grup ditambah *clipping* PPO."

### m9. Logika determinisme BM25 di bab2:15

- **Lokasi:** bab2:15: "Sifat deterministik inilah yang membuat *reward* bukti … dapat diverifikasi secara tepat, karena selalu diketahui *query* mana yang memunculkan dokumen mana."
- **Masalah:** Reward bisa diverifikasi karena environment **mencatat** dokumen yang dikembalikan di setiap episode. Determinisme membantu **reproduktibilitas**, bukan verifikasi.
- **Kenapa penting:** Ini kesalahan logika kecil, tetapi berada di dasar argumen "judge-free", sehingga mudah disorot.
- **Perbaikan:** "Sifat deterministik ini membuat setiap episode dapat direproduksi persis. Verifikasi *reward* bukti sendiri dimungkinkan karena *environment* mencatat dokumen yang dikembalikan oleh setiap *query* dalam episode (Bab 3)."

### m10. Kontribusi 3: bahasa peringkat dan kausal untuk hasil eksploratif

- **Lokasi:** bab1:59 ("R3 … **menempati peringkat terbaik** pada Joint-EM") dan bab1:60 ("*dynamic sampling* **menaikkan** akurasi *outcome-only*").
- **Masalah:**
  - Untuk R3, yang diuji hanya R3−R1 (+6,3 [+0,2; +13,1]). Hasil ini eksploratif, tidak dikoreksi untuk perbandingan berganda, dan dirancang setelah hasil *holdout* diketahui. R3 tidak diuji terhadap R2.
  - Untuk R0-dyn, hasilnya berasal dari satu *run* dengan dua kali jumlah pertanyaan, jadi efeknya tercampur (bab4:219 [lintas-bab] sudah mengakuinya).
- **Kenapa penting:** Bab 1 menyebut hasil eksploratif dengan bahasa yang lebih kuat daripada Bab 4. Inkonsistensi seperti ini mudah ditemukan penguji.
- **Perbaikan:** "R3 memperoleh Joint-EM rata-rata tertinggi (selisih terhadap R1 +6,3, SK [+0,2; +13,1], eksploratif)". Untuk R0-dyn: "*run* dengan *dynamic sampling* memperoleh EM 8,0 poin lebih tinggi, tetapi juga melihat dua kali lebih banyak pertanyaan, sehingga efeknya tidak dapat dipisahkan".

### m11. Latar belakang memotivasi dengan RAG terbuka, tetapi setting-nya *reading comprehension* tertutup

- **Lokasi:** bab1:5 (pengetahuan LLM yang tetap dan perlunya *retrieval*) dan bab1:11 (HotpotQA "dengan biaya rendah").
- **Masalah:** Motivasi di bab1:5 adalah pengetahuan di luar model, tetapi setting penelitian ini selalu memuat paragraf emas di 10 kandidat. Bab 3:49 [lintas-bab] sudah menyebut tugas ini "lebih dekat dengan pemahaman bacaan". Bab 1 baru menyebutnya di Batasan.
- **Kenapa penting:** Penguji bisa bertanya, "Motivasinya pengetahuan di luar model, tetapi buktinya selalu ada di input. Apa hubungannya?" Satu kalimat jembatan mencegah pertanyaan itu.
- **Perbaikan:** Tambahkan di akhir bab1:11: "Setting ini sengaja dibuat seperti laboratorium: kumpulan dokumennya tertutup dan selalu memuat bukti emas, sehingga yang dikaji adalah pengaruh *reward* terhadap perilaku menyitasi, bukan kemampuan menemukan dokumen di korpus besar."

### m12. RQ1: "performa" terlalu umum

- **Lokasi:** bab1:24.
- **Masalah:** Jawaban RQ1 di Bab 4–5 adalah "ya, tetapi sebagian besar melalui pembelajaran protokol". Rumusan "meningkatkan performa" membuat RQ1 hampir pasti terjawab "ya", karena B0 gagal pada 130 dari 300 episode.
- **Kenapa penting:** RQ yang jawabannya hampir pasti "ya" terlihat lemah. Rumusan yang memisahkan protokol dari kualitas jawaban justru menonjolkan analisis Bab 4 yang sudah bagus.
- **Perbaikan:** "**RQ1.** Apakah pelatihan GRPO meningkatkan akurasi agen pencarian 7B pada HotpotQA dibandingkan dengan model tanpa pelatihan, dan seberapa besar kenaikan tersebut berasal dari kepatuhan protokol dibandingkan dengan kualitas jawaban?"

### m13. Daftar pustaka: konsistensi format

- **Lokasi:** daftar-pustaka:41–65.
- **Masalah:**
  - [20]–[24] dan [26]–[32] tidak mencantumkan inisial ("Song dkk.", "Huang dkk."), sedangkan entri lain mencantumkannya ("B. Jin dkk.").
  - [27] menulis "dalam *Interspeech*, 2026" dan [30] menulis "dalam … EMNLP, 2026", tetapi teks bab2:85 menyebut keduanya preprint. Status diterima [27] tidak terkonfirmasi dari arXiv, dan [30] belum diperiksa.
  - Daftar referensi PROGRESS mengutip DeepResearcher [22] sebagai makalah EMNLP 2025, hlm. 414–431. Penulis perlu memeriksa versi terbit tersebut sendiri.
- **Kenapa penting:** Format pustaka yang tidak seragam adalah koreksi standar dari pembimbing atau penguji. Status venue yang tidak konsisten antara teks dan daftar pustaka juga mengundang pertanyaan.
- **Perbaikan:**
  - Inisial yang sudah terverifikasi dari paper sumber: [20] H. Song, [21] M. Chen, [22] Y. Zheng, [23] H. Sun, [24] Z. Wang, [26] C. Huang, [27] S. Paul.
  - Untuk [28]–[32], periksa sendiri di halaman arXiv masing-masing.
  - Untuk [27], pilih salah satu: tulis "*arXiv preprint* arXiv:2608.00969" jika status Interspeech belum dapat dibuktikan, atau sertakan bukti penerimaannya.

---

## NIT

Semua Nit penting karena alasan yang sama: penguji membaca dengan teliti, dan kesalahan kecil yang menumpuk memberi kesan naskah belum dirapikan. Alasan spesifik ditulis bila ada.

1. **Singkatan "SK" dipakai untuk beberapa hal.** bab1:35 mendefinisikan SK sebagai CI *bootstrap* "yang memperhitungkan *seed*". Namun bab1:56 memakai "SK 95% Clopper–Pearson", dan bab2:100 serta bab5:7 mendefinisikan ulang "selang kepercayaan (*confidence interval*, CI)". Definisikan SK sekali sebagai "selang kepercayaan" (umum), lalu tambahkan keterangan "(dengan *seed*)" atau "(Clopper–Pearson)" bila perlu.
2. **Rumusan RQ berbeda antara Bab 1 dan Bab 3.** bab3:28–30 [lintas-bab] menghilangkan "pada HotpotQA" dari RQ1 dan menulis "perbaikan *optimizer*", sedangkan bab1:26 menulis "perubahan *optimizer*". Salin rumusan RQ dari Bab 1 apa adanya.
3. **Klaim tanpa sitasi di bab1:7.** Kalimat "*Prompting* saja tidak secara andal menghasilkan perilaku pencarian yang baik" belum disitasi. Tambahkan [2], seperti di bab2:32.
4. **Sitasi langsung di bab1:19.** Kalimat "Search-R1 dilatih pada 8 GPU H100 dan AutoRefine pada 8 GPU A100 (Bab 2)" sebaiknya langsung menyitasi [2], [19].
5. **bab2:57:** "tidak muat pada satu GPU" → "tidak muat pada satu GPU 48 GB".
6. **bab2:100:** Agarwal dkk. "…dan tidak pernah sebagai *run* tunggal" adalah parafrase yang terlalu keras. Lebih aman: "…dan tidak hanya sebagai estimasi titik dari sedikit *run*".
7. **bab2:26:** Untuk ALCE, "pemisahan yang **sama**" → "pemisahan yang **analog**". *Citation recall* di ALCE adalah entailment per pernyataan, sedangkan cakupan di VEGR adalah *recall* terhadap himpunan kalimat emas.
8. **bab2:38, bab2:42, bab2:83:** Keterangan "dipublikasikan sebagai *preprint*" hanya muncul untuk PPO, DeepSeekMath, dan STAMP, padahal [20], [22]–[24], [28]–[29], [31]–[32] juga preprint. Konsistenkan, atau hapus semua keterangan itu karena daftar pustaka sudah memuat informasinya.
9. **bab1:61:** Kalimat "…tersedia di repositori; *adapter* LoRA disimpan untuk 15 dari 16 *run*" bisa terbaca seolah adapter ada di repositori, padahal bab3:287 menyatakan adapter disimpan lokal dan tidak masuk git. Tambahkan "(secara lokal, di luar repositori)".
10. **bab2:98:** Catatan tentang dua judul makalah [5] lebih cocok dijadikan catatan kaki.

---

## Pemeriksaan sitasi: nomor di teks dan daftar pustaka

- Semua nomor [1]–[33] di daftar pustaka **disitasi** setidaknya sekali di Bab 1–5. Tidak ada referensi yatim.
- Tidak ada nomor di teks yang **tidak ada** di daftar pustaka.
- Urutan nomor **sesuai urutan kemunculan pertama** (gaya IEEE): Bab 1 memperkenalkan [1]–[10] secara berurutan, lalu Bab 2 memperkenalkan [11]–[33] secara berurutan (bab2:7 → [11] sampai bab2:98 → [33]). Bab 3–5 hanya memakai ulang nomor yang sudah ada.
- Sitasi berlokasi ("[2, Lampiran B.2]" di bab2:69, "[19, Lampiran B.1]" di bab2:69) sudah diverifikasi cocok dengan isi paper.

---

## Kemungkinan pertanyaan penguji di sidang (cakupan Bab 1–2)

1. **"R0 Anda memberi −1 untuk sitasi tidak valid, sedangkan Search-R1 tidak. Bukankah keruntuhan sitasi R0 adalah artefak dari baseline yang Anda rancang sendiri? Apa yang terjadi jika sitasi tidak valid diberi 0?"**
   Jawaban yang disarankan: akui perbedaannya (M1). Jelaskan bahwa tujuan R0 adalah mengisolasi efek *tidak adanya imbalan* untuk sitasi benar ketika penalti validitas yang sama diterapkan pada kedua *reward*. Sebutkan juga bahwa varian "invalid = 0" tidak diuji dan merupakan saran yang wajar.
2. **"VEGR memberi *reward* pada sitasi yang benar, dan metriknya juga mengukur sitasi yang benar terhadap label yang sama. Wajar dong VEGR menang. Apa yang sebenarnya dipelajari dari RQ2?"**
   Jawaban yang disarankan: akui tumpang-tindih tersebut (bab4:351). Tekankan bahwa temuan utamanya adalah **mekanisme** keruntuhan R0 dan kenyataan bahwa akurasi saja tidak memperlihatkannya, bukan bahwa VEGR lebih ter-*grounding*. Rujuk juga Joint-EM yang tidak signifikan.
3. **"Anda menyebut celah G2 sebagai *reward* tanpa *judge* yang jarang diuji multi-*seed*. Huang dkk. 2024 sudah melakukan 3 *seed* dengan *reward* sitasi terperinci. Apa bedanya dengan Anda?"**
   Jawaban yang disarankan: Huang dkk. memakai model NLI, bukan aturan; setting-nya *long-form* QA, bukan agen pencarian *multi-turn*; dan pembandingnya *holistic reward* (periksa sendiri isinya di §2.3 paper tersebut sebelum sidang). Persempit celahnya sesuai M3.
4. **"Kalau model tanpa pelatihan sudah menemukan 83% dokumen emas dan semua pertanyaan pasti punya bukti di 10 paragraf, apakah ini masih 'agen pencarian'? Kenapa tidak memakai HotpotQA *fullwiki*?"**
   Jawaban yang disarankan: setting tertutup dipilih agar bukti dapat diperiksa dengan tepat dan biaya tetap rendah. Istilah "agen pencarian" sudah dibatasi di bab1:42. Sebutkan juga bahwa angka 83% dihitung atas episode yang selesai (m7).
5. **"RQ2 menanyakan 'tanpa kehilangan akurasi', tetapi CI Anda [−3,6; +5,4]. Berapa penurunan akurasi yang masih Anda anggap 'tidak kehilangan'?"**
   Jawaban yang disarankan: tidak ada margin non-inferioritas yang ditetapkan sebelumnya. Kesimpulan yang sah hanyalah "tidak terdeteksi perbedaan, dan penurunan hingga sekitar 3,6 poin belum dapat disingkirkan" (M2).
