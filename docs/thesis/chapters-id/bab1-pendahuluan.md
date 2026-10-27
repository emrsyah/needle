# BAB 1 — PENDAHULUAN

## 1.1 Latar Belakang

*Large Language Model* (LLM) dapat menjawab berbagai macam pertanyaan, tetapi pengetahuannya tetap sejak pelatihan berakhir, dan LLM terkadang menyampaikan informasi yang salah dengan penuh keyakinan. Salah satu solusi yang umum digunakan adalah *Retrieval-Augmented Generation* (RAG), yaitu sistem yang terlebih dahulu mengambil dokumen yang relevan sebelum menjawab, lalu menyusun jawabannya berdasarkan dokumen tersebut. Namun, banyak pertanyaan yang tidak dapat dijawab hanya dengan satu kali proses *retrieval*. Pertanyaan *multi-hop* seperti "Perusahaan mana yang membentuk *boy group* yang debut dengan album *2014 S/S*?" mengharuskan sistem menemukan satu fakta terlebih dahulu (grup tersebut), lalu menggunakan fakta itu untuk menemukan fakta berikutnya (perusahaannya).

Hal ini mendorong munculnya **agen pencarian**, yaitu LLM yang memutuskan sendiri kapan harus mencari, apa yang dicari, dan kapan harus berhenti lalu menjawab [1]. *Prompting* saja tidak secara andal menghasilkan perilaku pencarian yang baik, sehingga penelitian terbaru melatih agen pencarian dengan *reinforcement learning* (RL). Secara khusus, **Search-R1** [2] melatih model dengan *outcome reward*, artinya model hanya diberi *reward* ketika jawaban akhirnya benar. Search-R1 menggunakan **GRPO** [3], yaitu algoritma RL yang membandingkan beberapa percobaan untuk pertanyaan yang sama alih-alih melatih model nilai (*value model*) yang terpisah. GRPO dengan *reward* yang diperiksa berbasis aturan juga menjadi inti dari model penalaran terbaru [4].

Pemberian *reward* hanya pada jawaban akhir menimbulkan sebuah masalah. Jawaban yang benar tidak menunjukkan bahwa agen telah menemukan, menggunakan, atau dapat menunjukkan bukti (*evidence*) di balik jawaban tersebut. Dalam istilah RL, *reward* jawaban merupakan **proksi** dari hal yang sebenarnya diinginkan, yaitu jawaban benar yang *didukung oleh bukti*. Optimasi terhadap sebuah proksi dapat menghasilkan perilaku yang memperoleh skor tinggi pada proksi tersebut, tetapi tidak pada tujuan yang sebenarnya [5].

HotpotQA [6] menyediakan cara untuk mengkaji masalah ini dengan biaya rendah. Pertanyaan pada HotpotQA dianotasi dengan kalimat-kalimat persis yang mendukung jawabannya. Apabila agen memberikan sitasi berupa kalimat, sitasi tersebut dapat diperiksa dengan aturan sederhana terhadap label emas ini, tanpa memerlukan *judge*.

## 1.2 Perumusan Masalah

Tugas akhir ini membahas empat celah penelitian yang saling berkaitan, yaitu sebagai berikut.

1. **G1, benar tidak berarti ter-*grounding*.** Agen yang dilatih dengan *reward* berbasis jawaban saja dapat belajar menjawab tanpa menunjukkan bukti, selama jawaban akhirnya benar. Penelitian terbaru melaporkan perilaku jalan pintas (*shortcut*) seperti ini pada agen pencarian yang dilatih berbasis *outcome* [7].
2. **G2, *reward* bukti tanpa *judge* jarang diuji terhadap kontrol *outcome-only*.** Beberapa *reward* bukti menggunakan LLM sebagai *judge* [7], [8], yang menambah biaya dan berpotensi menimbulkan bias. Alternatif berbasis aturan sudah ada (Bab 2), tetapi jarang dibandingkan dengan *reward outcome-only* pada beberapa *seed* dalam pengaturan yang selebihnya identik.
3. **G3, hambatan komputasi.** RL untuk agen pencarian dilaporkan pada skala multi-GPU: Search-R1 dilatih pada 8 GPU H100 dan AutoRefine pada 8 GPU A100 (Bab 2). Belum jelas apakah efek yang sama dapat dikaji pada satu GPU.
4. **G4, lemahnya bukti tentang faktor yang berpengaruh.** Hasil RL bervariasi cukup besar antar-*seed* acak [9]. Search-R1 dan CaRR melatih setiap konfigurasi hanya satu kali (Bab 2), dan satu *run* tidak dapat memisahkan efek suatu metode dari *noise* akibat *seed* [10].

Berdasarkan celah tersebut, rumusan masalah pada penelitian ini adalah sebagai berikut.

1. **RQ1.** Apakah pelatihan GRPO meningkatkan performa agen pencarian 7B pada HotpotQA dibandingkan dengan model tanpa pelatihan?
2. **RQ2.** Apakah *verifiable evidence-grounded reward* (VEGR) mencegah perilaku jalan pintas dari *reward* berbasis jawaban saja, tanpa kehilangan akurasi?
3. **RQ3.** Bagaimana pengaruh varian *reward* (penalti bergerbang/*gated penalties* dan bobot cakupan yang lebih besar) serta perubahan *optimizer* (*advantage* tanpa normalisasi standar deviasi dan *dynamic sampling*) terhadap *grounding* dan perilaku pencarian?

RQ1 dan RQ2 merupakan pertanyaan utama. RQ3 bersifat eksploratif, karena varian-variannya sebagian dirancang setelah hasil pertama diperoleh (Subbab 3.8).

## 1.3 Tujuan

Setiap tujuan menjawab satu rumusan masalah. Tujuan pada penelitian ini adalah sebagai berikut.

1. **Untuk RQ1:** melatih agen pencarian dengan RL bergaya GRPO dan LoRA pada satu GPU, lalu mengukur akurasi dan kepatuhan protokolnya dibandingkan dengan model tanpa pelatihan pada *holdout* yang dibekukan.
2. **Untuk RQ2:** merancang *reward* tanpa *judge*, yaitu VEGR, yang memberi *reward* pada jawaban yang benar, sitasi yang benar, dan pencarian yang berguna, lalu membandingkannya dengan *reward* berbasis jawaban saja pada tiga *seed*, baik dari sisi akurasi maupun *grounding*, menggunakan selang kepercayaan (*confidence interval*, CI) *bootstrap* yang memperhitungkan *seed* (selanjutnya disingkat SK).
3. **Untuk RQ3:** menguji apakah varian *reward* dan perubahan *optimizer* mengubah *grounding*, cakupan bukti, dan stabilitas pelatihan.

## 1.4 Batasan Masalah

Tugas akhir ini sengaja dibatasi pada ruang lingkup yang sempit. Batasan masalah pada penelitian ini adalah sebagai berikut.

1. **Data dan *retrieval*.** Penelitian ini menggunakan HotpotQA pada *distractor setting*. Agen melakukan pencarian dengan BM25 pada **kumpulan tertutup berisi 10 paragraf milik setiap pertanyaan**, dan kumpulan ini selalu memuat kedua paragraf emas. Model tanpa pelatihan sudah mengambil 83% dokumen emas (*retrieval recall* 0,83), sehingga tugas ini jauh lebih mudah dibandingkan dengan pencarian pada web terbuka atau korpus terbuka. Istilah "agen pencarian" pada tugas akhir ini berarti agen yang melakukan pencarian pada kumpulan tertutup tersebut.
2. **Model.** Model yang digunakan adalah Qwen2.5-7B-Instruct yang diadaptasi dengan LoRA.
3. **Skala.** Setiap *run* pelatihan menggunakan 256 pertanyaan selama satu *epoch*, dan evaluasi menggunakan 300 pertanyaan *holdout* yang dibekukan. Anggaran proyek ini sekitar $45, sedangkan biaya komputasi yang benar-benar terpakai sekitar $49 (Tabel 3.2).
4. **Label.** VEGR membutuhkan bukti emas pada tingkat kalimat, yang hanya disediakan oleh sedikit *dataset*. Metode ini tidak dapat diterapkan apa adanya pada data yang tidak memiliki label tersebut.
5. **Hal yang tidak diklaim.** Tugas akhir ini tidak mengusulkan algoritma RL baru maupun jenis *reward* baru, tidak mengklaim akurasi *state-of-the-art*, dan tidak mengklaim bahwa temuannya berlaku pula untuk pencarian web terbuka, *dataset* lain, atau model yang lebih besar.
6. **Celah (*loophole*) yang diketahui.** Lingkungan (*environment*) menerima jawaban tanpa sitasi. Hal ini sengaja dipertahankan agar seluruh *run* dapat dibandingkan, dan konsekuensinya dilaporkan.

Keterbatasan lainnya dibahas pada Bab 5.

## 1.5 Manfaat Penelitian

**Kontribusi.** Kontribusi penelitian ini adalah sebagai berikut.

1. **VEGR**, yaitu *reward* tanpa *judge* pada tingkat kalimat untuk melatih agen pencarian. Setiap komponennya diperiksa berbasis aturan terhadap *supporting facts* emas. VEGR juga menegakkan protokol aksi: sitasi yang dikarang, aksi yang tidak sesuai format, dan kehabisan anggaran langkah masing-masing diberi penalti −1. Komponen-komponen *reward* tersebut secara individual bukanlah hal baru; kontribusinya terletak pada kombinasi komponen tersebut dan evaluasinya yang terkontrol.
2. **Bukti tentang kapan *reward* berbasis jawaban saja kehilangan sitasi, beserta penyebabnya.** Dua dari tiga *seed outcome-only* berhenti memberikan sitasi sama sekali (SK 95% Clopper–Pearson untuk proporsi tersebut [0,09; 0,99]). Log pelatihan menunjukkan penyebabnya: dengan *reward* berbasis jawaban saja, sitasi yang tidak valid bernilai −1, sedangkan sitasi yang benar tidak memperoleh apa pun. Paket VEGR lengkap mencapai akurasi yang sama (+1,0 poin EM, SK 95% yang memperhitungkan *seed* [−3,6; +5,4]), tetapi **24,4 poin lebih tinggi pada *precision-grounded* EM** (SK [+8,8; +37,3]).
3. **Perbandingan multi-*seed* yang terkontrol** dengan SK *bootstrap* yang memperhitungkan *seed*. Hasil eksploratif menunjukkan bahwa:
   - peningkatan +6 poin EM pada satu *seed* (penalti bergerbang, R2) tidak tereplikasi pada *seed* lain;
   - VEGR membuat sitasi menjadi benar tetapi belum lengkap, dan bobot cakupan yang lebih besar (R3) menukar presisi dengan kelengkapan sehingga menempati peringkat terbaik pada Joint-EM HotpotQA;
   - pada satu *seed* yang diuji, *dynamic sampling* menaikkan akurasi *outcome-only* (bersamaan dengan jumlah pertanyaan pelatihan yang dua kali lipat), tetapi tidak mencegah hilangnya sitasi.
4. ***Pipeline* satu GPU yang terdokumentasi.** Satu *run* pelatihan beserta evaluasinya membutuhkan biaya sekitar $2,25 pada satu GPU 48 GB. Pembagian data, konfigurasi, log pelatihan, dan hasil per episode tersedia di repositori; *adapter* LoRA disimpan untuk 15 dari 16 *run*.

**Manfaat.** Bagi peneliti, tugas akhir ini menunjukkan bagaimana sebuah *reward* bukti dapat dievaluasi dengan biaya rendah dan dengan ketidakpastian yang dilaporkan secara jujur, serta memberikan penjelasan konkret tentang bagaimana *reward* berbasis jawaban saja dapat kehilangan sitasi. Bagi praktisi yang membangun sistem *retrieval-augmented*, tugas akhir ini menunjukkan bahwa akurasi jawaban saja tidak mengungkapkan apakah jawaban tersebut ter-*grounding*, sehingga *grounding* perlu diberi *reward* dan diukur secara eksplisit.

## 1.6 Sistematika Penulisan

Sistematika penulisan tugas akhir ini adalah sebagai berikut.

- **Bab 2** mengkaji landasan teori, yaitu LLM, *retrieval*, *multi-hop question answering* dan evaluasi sitasi, agen LLM, RL untuk LLM (PPO, GRPO beserta variannya, dan LoRA), agen pencarian yang dilatih dengan RL, *reward* bukti, serta *reward hacking* dan evaluasi RL. Bab ini diakhiri dengan celah penelitian.
- **Bab 3** menjelaskan perancangan sistem, yaitu pembagian data, lingkungan pencarian, *reward* VEGR beserta variannya, model *policy*, algoritma pelatihan, protokol evaluasi, dan matriks eksperimen.
- **Bab 4** melaporkan hasil penelitian berdasarkan rumusan masalah, mendiagnosis dinamika pelatihan, menyajikan contoh *trace*, dan membahas ancaman terhadap validitas.
- **Bab 5** memuat kesimpulan, jawaban atas rumusan masalah, serta pembahasan etika, keterbatasan, dan saran untuk penelitian selanjutnya.
