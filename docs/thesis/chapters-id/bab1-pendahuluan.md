# BAB 1 PENDAHULUAN

## 1.1 Latar Belakang

*Large Language Model* (LLM) dapat menjawab berbagai macam pertanyaan, tetapi pengetahuannya tetap sejak pelatihan berakhir, dan LLM terkadang menyampaikan informasi yang salah dengan penuh keyakinan. Salah satu solusi yang umum digunakan adalah *Retrieval-Augmented Generation* (RAG), yaitu sistem yang terlebih dahulu mengambil dokumen yang relevan sebelum menjawab, lalu menyusun jawabannya berdasarkan dokumen tersebut. Namun, banyak pertanyaan yang tidak dapat dijawab hanya dengan satu kali proses *retrieval*. Pertanyaan *multi-hop* seperti "Perusahaan mana yang membentuk *boy group* yang debut dengan album *2014 S/S*?" mengharuskan sistem menemukan satu fakta terlebih dahulu (grup tersebut), lalu menggunakan fakta itu untuk menemukan fakta berikutnya (perusahaannya).

Hal ini mendorong munculnya **agen pencarian**, yaitu LLM yang memutuskan sendiri kapan harus mencari, apa yang dicari, dan kapan harus berhenti lalu menjawab [1]. *Prompting* saja tidak secara andal menghasilkan perilaku pencarian yang baik [2], sehingga penelitian terbaru melatih agen pencarian dengan *reinforcement learning* (RL). Secara khusus, **Search-R1** [2] melatih model dengan *outcome reward*, artinya model hanya diberi *reward* ketika jawaban akhirnya benar. Search-R1 dapat dilatih dengan PPO maupun **GRPO** [3]. GRPO adalah algoritma RL yang membandingkan beberapa percobaan untuk pertanyaan yang sama alih-alih melatih model nilai (*value model*) yang terpisah. GRPO dengan *reward* yang diperiksa berbasis aturan juga menjadi inti dari model penalaran terbaru [4].

Pemberian *reward* hanya pada jawaban akhir menimbulkan sebuah masalah. Jawaban yang benar tidak menunjukkan bahwa agen telah menemukan, menggunakan, atau dapat menunjukkan bukti (*evidence*) di balik jawaban tersebut. Dalam istilah RL, *reward* jawaban merupakan **proksi** dari hal yang sebenarnya diinginkan, yaitu jawaban benar yang *didukung oleh bukti*. Optimasi terhadap sebuah proksi dapat menghasilkan perilaku yang memperoleh skor tinggi pada proksi tersebut, tetapi tidak pada tujuan yang sebenarnya [5].

HotpotQA [6] menyediakan cara untuk mengkaji masalah ini tanpa penilai manusia maupun model. Pertanyaan pada HotpotQA dianotasi dengan kalimat-kalimat persis yang mendukung jawabannya. Apabila agen memberikan sitasi berupa kalimat, sitasi tersebut dapat diperiksa dengan aturan sederhana terhadap label emas ini, tanpa memerlukan *judge*. Pengaturan ini sengaja dibuat seperti laboratorium: kumpulan dokumen untuk setiap pertanyaan bersifat tertutup dan selalu memuat bukti emas, sehingga yang dikaji adalah pengaruh *reward* terhadap perilaku menyitasi, bukan kemampuan menemukan dokumen di dalam korpus yang besar.

## 1.2 Perumusan Masalah

Tugas akhir ini membahas empat celah penelitian yang saling berkaitan, yaitu sebagai berikut.

1. **G1, benar tidak berarti ter-*grounding*.** Agen yang dilatih dengan *reward* berbasis jawaban saja (*outcome-only*) dapat belajar menjawab tanpa menunjukkan bukti, selama jawaban akhirnya benar. Pada tugas akhir ini, perilaku tersebut disebut **jalan pintas** (*shortcut*), yaitu agen menjawab tanpa menyertakan sitasi bukti sehingga tingkat sitasinya mendekati nol. CaRR [7] melaporkan gejala yang serupa pada agen pencarian yang dilatih berbasis *outcome*, yaitu jawaban benar yang tidak didukung bukti secara lengkap. Istilah *shortcut* pada CaRR lebih luas, karena mencakup pula pengabaian sebagian *hop* dan pemanggilan *tool* yang lebih sedikit.
2. **G2, *reward* bukti berbasis aturan jarang diuji terhadap kontrol *outcome-only* pada agen pencarian.** Beberapa *reward* bukti menggunakan LLM sebagai *judge* [7], [8], yang menambah model tambahan dalam *loop* pelatihan dan berpotensi menimbulkan bias. *Reward* bukti tanpa LLM *judge* sudah ada (Bab 2), tetapi *reward* berbasis aturan pada tingkat kalimat untuk agen pencarian *multi-turn* yang dibandingkan dengan kontrol *outcome-only* pada beberapa *seed*, dalam pengaturan yang selebihnya identik, jarang dilaporkan.
3. **G3, hambatan komputasi.** RL untuk agen pencarian umumnya dilaporkan pada skala multi-GPU: Search-R1 dilatih pada 8 GPU H100 [2], dan AutoRefine pada 8 GPU A100 (Bab 2). Tugas akhir ini tidak mereproduksi pengaturan tersebut, tetapi menyederhanakan tugasnya (kumpulan dokumen tertutup, LoRA, dan 256 pertanyaan pelatihan per *run*) agar perbandingan *reward* dapat dilakukan pada satu GPU.
4. **G4, lemahnya bukti tentang faktor yang berpengaruh.** Hasil RL bervariasi cukup besar antar-*seed* acak [9], sehingga satu *run* tidak dapat memisahkan efek suatu metode dari *noise* akibat *seed* [10]. Search-R1 tidak melaporkan *seed* maupun variansi, dan CaRR melaporkan pengulangan evaluasi, bukan pengulangan pelatihan dengan *seed* yang berbeda (Bab 2).

Berdasarkan celah tersebut, rumusan masalah pada penelitian ini adalah sebagai berikut.

1. **RQ1.** Apakah pelatihan GRPO meningkatkan performa agen pencarian 7B pada HotpotQA dibandingkan dengan model tanpa pelatihan?
2. **RQ2.** Apakah *verifiable evidence-grounded reward* (VEGR) mencegah perilaku jalan pintas dari *reward* berbasis jawaban saja, tanpa kehilangan akurasi?
3. **RQ3.** Bagaimana pengaruh varian *reward* (penalti bergerbang/*gated penalties* dan bobot cakupan yang lebih besar) serta perubahan *optimizer* (*advantage* tanpa normalisasi standar deviasi dan *dynamic sampling*) terhadap *grounding* dan perilaku pencarian?

G1 dan G2 dijawab oleh RQ1–RQ3. G3 dan G4 tidak dirumuskan sebagai pertanyaan penelitian tersendiri, melainkan dijawab melalui rancangan eksperimen: seluruh *run* dilatih pada satu GPU (G3), dan setiap metode utama dilatih dengan tiga *seed* serta dilaporkan dengan selang kepercayaan yang memperhitungkan *seed* (G4).

RQ1 dan RQ2 merupakan pertanyaan utama. Pada RQ1, pelatihan yang dimaksud adalah RL bergaya GRPO (Subbab 3.6), dan performa diukur dengan *exact match* (EM) atas seluruh 300 pertanyaan *holdout*. Kenaikannya kemudian dipisahkan menjadi bagian yang berasal dari kepatuhan terhadap protokol aksi dan bagian yang berasal dari akurasi jawaban pada episode yang selesai. Pada RQ2, VEGR dinilai mencegah jalan pintas apabila agen tetap menyitasi bukti yang benar pada setiap *seed*. Frasa "tanpa kehilangan akurasi" diperiksa melalui selisih EM terhadap *reward outcome-only* beserta selang kepercayaannya. Karena tidak ada margin non-inferioritas yang ditetapkan sejak awal, kesimpulan tentang akurasi dinyatakan sebagai ada atau tidaknya perbedaan yang terdeteksi, disertai batas-batas selangnya. RQ3 bersifat eksploratif, karena R3, R4, dan R0-dyn dirancang setelah hasil *holdout* R0, R1, dan R2 diketahui (Subbab 3.7). Penalti bergerbang pada RQ3 adalah varian R2, yang pada bab-bab berikutnya disebut penalti bersyarat (*gated*).

## 1.3 Tujuan

Setiap tujuan menjawab satu rumusan masalah. Tujuan pada penelitian ini adalah sebagai berikut.

1. **Untuk RQ1:** melatih agen pencarian dengan RL bergaya GRPO dan LoRA pada satu GPU, lalu mengukur akurasi dan kepatuhan protokolnya dibandingkan dengan model tanpa pelatihan pada *holdout* yang dibekukan.
2. **Untuk RQ2:** merancang *reward* tanpa *judge*, yaitu VEGR, yang memberi *reward* pada jawaban yang benar, sitasi yang benar, dan pencarian yang berguna, lalu membandingkannya dengan *reward outcome-only* pada tiga *seed*, baik dari sisi akurasi maupun *grounding*, menggunakan selang kepercayaan (SK, *confidence interval*) *bootstrap* yang memperhitungkan *seed*.
3. **Untuk RQ3:** menguji apakah varian *reward* (penalti bersyarat dan bobot cakupan ×2) serta perubahan *optimizer* mengubah *grounding*, cakupan bukti, perilaku pencarian, dan variansi antar-*seed*.

## 1.4 Batasan Masalah

Tugas akhir ini sengaja dibatasi pada ruang lingkup yang sempit. Batasan masalah pada penelitian ini adalah sebagai berikut.

1. **Data dan *retrieval*.** Penelitian ini menggunakan HotpotQA pada *distractor setting*. Agen melakukan pencarian dengan BM25 pada **kumpulan tertutup berisi 10 paragraf milik setiap pertanyaan**, dan kumpulan ini selalu memuat kedua paragraf emas. Pada episode yang selesai, model tanpa pelatihan sudah memperoleh *retrieval recall* 0,83, artinya 83% *supporting facts* emas berada pada paragraf yang terambil oleh pencariannya. Oleh karena itu, tugas ini jauh lebih mudah dibandingkan dengan pencarian pada web terbuka atau korpus terbuka. Istilah "agen pencarian" pada tugas akhir ini, termasuk "agen RAG" pada judul, berarti agen yang melakukan pencarian pada kumpulan tertutup tersebut.
2. **Model.** Model yang digunakan adalah Qwen2.5-7B-Instruct yang diadaptasi dengan LoRA.
3. **Skala.** Setiap *run* pelatihan menggunakan 256 pertanyaan selama satu *epoch*, dan evaluasi menggunakan 300 pertanyaan *holdout* yang dibekukan. Seluruh pelatihan dijalankan pada satu GPU 48 GB, dengan total waktu pelatihan sekitar 33 jam GPU untuk 16 *run* (Tabel 3.4).
4. **Label.** VEGR membutuhkan bukti emas pada tingkat kalimat, yang hanya disediakan oleh sedikit *dataset*. Metode ini tidak dapat diterapkan apa adanya pada data yang tidak memiliki label tersebut.
5. ***Baseline outcome-only*.** *Reward* R0 berupa EM ditambah penalti −1 untuk episode yang tidak valid (Subbab 3.5). Search-R1 tidak memakai *reward* format dan tidak memberi penalti pada aksi yang tidak valid [2]. Dengan demikian, R0 bukan replikasi Search-R1, dan temuan tentang hilangnya sitasi berlaku untuk rancangan *reward* R0 tersebut.
6. **Metrik dan *reward*.** *Grounding* diukur terhadap *supporting facts* emas yang sama dengan yang diberi *reward* oleh VEGR. Tidak ada pemeriksaan atribusi yang independen dari label tersebut.
7. ***Baseline prompt*.** Tidak ada *baseline* berbasis *prompt few-shot*. *Baseline* B0 adalah model yang sama dengan *prompt* yang sama tanpa pelatihan.
8. **Hal yang tidak diklaim.** Tugas akhir ini tidak mengusulkan algoritma RL baru maupun jenis *reward* baru, tidak mengklaim akurasi *state-of-the-art*, dan tidak mengklaim bahwa temuannya berlaku pula untuk pencarian web terbuka, *dataset* lain, atau model yang lebih besar.
9. **Celah (*loophole*) yang diketahui.** Lingkungan (*environment*) menerima jawaban tanpa sitasi, dan tidak memeriksa apakah paragraf yang disitasi pernah muncul pada hasil pencarian. Hal ini sengaja dipertahankan agar seluruh *run* dapat dibandingkan pada lingkungan yang sama, dan konsekuensinya dilaporkan.

Keterbatasan lainnya dibahas pada Bab 5.

## 1.5 Manfaat Penelitian

**Kontribusi.** Kontribusi penelitian ini adalah sebagai berikut.

1. **VEGR, yaitu *reward* tanpa *judge* pada tingkat kalimat untuk melatih agen pencarian. Setiap komponennya diperiksa berbasis aturan terhadap *supporting facts* emas, dan aksi yang tidak valid (termasuk sitasi ke judul atau kalimat yang tidak ada) diberi penalti −1. Komponen-komponennya secara individual bukanlah hal baru; kontribusinya terletak pada kombinasi komponen tersebut dan evaluasinya yang terkontrol.**
2. **Bukti bahwa VEGR mempertahankan akurasi yang setara dengan _reward outcome-only_ sambil meningkatkan _grounding_ bukti. Tidak ada perbedaan akurasi yang terdeteksi (+1,0 poin EM, SK [−3,6; +5,4]), sedangkan _precision-grounded_ EM naik 24,4 poin (SK [+8,8; +37,3]). Dua dari tiga _seed outcome-only_ berhenti memberikan sitasi sama sekali, sedangkan tidak satu pun dari tiga _seed_ VEGR mengalaminya.**
3. **Perbandingan multi-*seed* yang terkontrol dengan SK *bootstrap* yang memperhitungkan *seed*. Perbandingan ini menunjukkan bahwa kenaikan pada satu *seed* dapat tidak tereplikasi pada *seed* lain (R2), dan bahwa bobot cakupan yang lebih besar (R3) menukar presisi sitasi dengan kelengkapan bukti.**
4. ***Pipeline* satu GPU yang terdokumentasi. Pembagian data, konfigurasi, log pelatihan, hasil per episode, dan skrip analisis tersedia di repositori, dan satu *run* pelatihan beserta evaluasinya membutuhkan waktu sekitar 2 jam pada satu GPU 48 GB.**

Rincian angka setiap kontribusi beserta batasannya dibahas pada Bab 4 dan Bab 5.

**Manfaat.** Bagi peneliti, tugas akhir ini menunjukkan bagaimana sebuah *reward* bukti dapat dievaluasi pada satu GPU dan dengan ketidakpastian yang dilaporkan secara jujur, serta memberikan contoh konkret tentang bagaimana *reward outcome-only* dapat kehilangan sitasi. Bagi praktisi yang membangun sistem *retrieval-augmented*, tugas akhir ini menunjukkan bahwa akurasi jawaban saja tidak mengungkapkan apakah jawaban tersebut ter-*grounding*, sehingga *grounding* perlu diukur secara eksplisit.

## 1.6 Sistematika Penulisan

Sistematika penulisan tugas akhir ini adalah sebagai berikut.

- **Bab 2** mengkaji landasan teori, yaitu LLM, *retrieval*, *multi-hop question answering* dan evaluasi sitasi, agen LLM, RL untuk LLM (PPO, GRPO beserta variannya, dan LoRA), agen pencarian yang dilatih dengan RL, *reward* bukti, serta *reward hacking* dan evaluasi RL. Bab ini diakhiri dengan celah penelitian.
- **Bab 3** menjelaskan perancangan sistem sesuai urutan diagram alur penelitian, yaitu agen pencarian, *dataset* dan pembagian data, lingkungan pencarian, *reward* VEGR beserta variannya, algoritma pelatihan GRPO, dan skenario eksperimen, diikuti protokol evaluasi serta implementasi dan komputasi.
- **Bab 4** melaporkan hasil penelitian berdasarkan rumusan masalah, mendiagnosis dinamika pelatihan, menyajikan contoh *trace*, dan membahas variasi hasil antar-*seed*.
- **Bab 5** memuat kesimpulan beserta jawaban atas rumusan masalah, kontribusi, keterbatasan, etika dan dampak, serta saran untuk penelitian selanjutnya.
- **Lampiran A** memuat teks *prompt* agen untuk satu pertanyaan, **Lampiran B** memuat dinamika pelatihan seluruh *run*, dan **Lampiran C** memuat posisi kalimat yang disitasi oleh salah satu *run* R1.
