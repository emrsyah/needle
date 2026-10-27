# BAB 1 PENDAHULUAN

## 1.1 Latar Belakang

*Large Language Model* (LLM) dapat menjawab berbagai macam pertanyaan, tetapi pengetahuannya tetap sejak pelatihan berakhir, dan LLM terkadang menyampaikan informasi yang salah dengan penuh keyakinan. Salah satu solusi yang umum digunakan adalah *Retrieval-Augmented Generation* (RAG), yaitu sistem yang terlebih dahulu mengambil dokumen yang relevan sebelum menjawab, lalu menyusun jawabannya berdasarkan dokumen tersebut. Namun, banyak pertanyaan yang tidak dapat dijawab hanya dengan satu kali proses *retrieval*. Pertanyaan *multi-hop* seperti "Perusahaan mana yang membentuk *boy group* yang debut dengan album *2014 S/S*?" mengharuskan sistem menemukan satu fakta terlebih dahulu (grup tersebut), lalu menggunakan fakta itu untuk menemukan fakta berikutnya (perusahaannya).

Hal ini mendorong munculnya **agen pencarian**, yaitu LLM yang memutuskan sendiri kapan harus mencari, apa yang dicari, dan kapan harus berhenti lalu menjawab [1]. *Prompting* saja tidak secara andal menghasilkan perilaku pencarian yang baik, sehingga penelitian terbaru melatih agen pencarian dengan *reinforcement learning* (RL). Secara khusus, **Search-R1** [2] melatih model dengan *outcome reward*, artinya model hanya diberi *reward* ketika jawaban akhirnya benar. Search-R1 menggunakan **GRPO** [3], yaitu algoritma RL yang membandingkan beberapa percobaan untuk pertanyaan yang sama alih-alih melatih model nilai (*value model*) yang terpisah. GRPO dengan *reward* yang diperiksa berbasis aturan juga menjadi inti dari model penalaran terbaru [4].

Pemberian *reward* hanya pada jawaban akhir menimbulkan sebuah masalah. Jawaban yang benar tidak menunjukkan bahwa agen telah menemukan, menggunakan, atau dapat menunjukkan bukti (*evidence*) di balik jawaban tersebut. Dalam istilah RL, *reward* jawaban merupakan **proksi** dari hal yang sebenarnya diinginkan, yaitu jawaban benar yang *didukung oleh bukti*. Optimasi terhadap sebuah proksi dapat menghasilkan perilaku yang memperoleh skor tinggi pada proksi tersebut, tetapi tidak pada tujuan yang sebenarnya [5]. Penelitian terbaru juga melaporkan perilaku jalan pintas (*shortcut*) pada agen pencarian yang dilatih berbasis *outcome* [7]. Beberapa penelitian telah mengusulkan *reward* bukti untuk mengatasi hal ini, tetapi sebagian di antaranya menggunakan LLM sebagai *judge* [7], [8], yang menambah biaya dan berpotensi menimbulkan bias.

HotpotQA [6] menyediakan cara untuk mengkaji masalah ini dengan biaya rendah. Pertanyaan pada HotpotQA dianotasi dengan kalimat-kalimat persis yang mendukung jawabannya (*supporting facts*). Apabila agen memberikan sitasi berupa kalimat, sitasi tersebut dapat diperiksa dengan aturan sederhana terhadap label emas ini, tanpa memerlukan *judge*.

Selain itu, terdapat dua hambatan praktis dalam mengkaji *reward* untuk agen pencarian. Pertama, RL untuk agen pencarian umumnya dilaporkan pada skala multi-GPU; Search-R1, misalnya, dilatih pada 8 GPU H100 (Bab 2). Kedua, hasil RL bervariasi cukup besar antar-*seed* acak [9], sehingga hasil dari satu *run* tidak dapat memisahkan efek suatu metode dari *noise* akibat *seed* [10].

Oleh karena itu, penelitian ini mengusulkan **VEGR** (*Verifiable Evidence-Grounded Reward*), yaitu *reward* tanpa *judge* pada tingkat kalimat yang seluruh komponennya diperiksa secara berbasis aturan terhadap *supporting facts* emas HotpotQA. VEGR akan memberi *reward* pada jawaban yang benar, sitasi yang benar, dan pencarian yang berguna, serta memberi penalti pada aksi yang tidak valid. Agen pencarian Qwen2.5-7B-Instruct [12] dengan LoRA akan dilatih menggunakan *policy gradient* bergaya GRPO pada satu GPU, dan VEGR akan dibandingkan secara terkontrol dengan *reward* berbasis jawaban saja pada beberapa *seed*. Komponen-komponen *reward* tersebut secara individual bukanlah hal baru; fokus penelitian ini terletak pada kombinasi komponen tersebut untuk agen pencarian *multi-turn* dan pada evaluasinya yang terkontrol dengan ketidakpastian yang dilaporkan secara jujur.

## 1.2 Perumusan Masalah

Berdasarkan latar belakang di atas, permasalahan yang diangkat pada penelitian ini adalah sebagai berikut.

1. **Benar tidak berarti ter-*grounding*.** Agen yang dilatih dengan *reward* berbasis jawaban saja dapat belajar menjawab tanpa menunjukkan bukti, selama jawaban akhirnya benar. Perilaku jalan pintas seperti ini telah dilaporkan pada agen pencarian yang dilatih berbasis *outcome* [7].
2. ***Reward* bukti tanpa *judge* jarang diuji terhadap kontrol *outcome-only*.** Beberapa *reward* bukti menggunakan LLM sebagai *judge* [7], [8], yang menambah biaya dan berpotensi menimbulkan bias. Alternatif berbasis aturan sudah ada (Bab 2), tetapi jarang dibandingkan dengan *reward outcome-only* pada beberapa *seed* dalam pengaturan yang selebihnya identik.
3. **Hambatan komputasi.** RL untuk agen pencarian dilaporkan pada skala multi-GPU: Search-R1 dilatih pada 8 GPU H100 dan AutoRefine pada 8 GPU A100 (Bab 2). Belum jelas apakah efek yang sama dapat dikaji pada satu GPU.
4. **Lemahnya bukti tentang faktor yang berpengaruh.** Hasil RL bervariasi cukup besar antar-*seed* acak [9]. Search-R1 dan CaRR melatih setiap konfigurasi hanya satu kali (Bab 2), dan satu *run* tidak dapat memisahkan efek suatu metode dari *noise* akibat *seed* [10].

Berdasarkan permasalahan tersebut, pertanyaan penelitian (*research question*, RQ) yang akan dijawab pada penelitian ini adalah sebagai berikut.

1. **RQ1.** Apakah pelatihan GRPO meningkatkan performa agen pencarian 7B pada HotpotQA dibandingkan dengan model tanpa pelatihan?
2. **RQ2.** Apakah *verifiable evidence-grounded reward* (VEGR) mencegah perilaku jalan pintas dari *reward* berbasis jawaban saja, tanpa kehilangan akurasi?
3. **RQ3.** Bagaimana pengaruh varian *reward* (penalti bergerbang/*gated penalties* dan bobot cakupan yang lebih besar) serta perubahan *optimizer* (*advantage* tanpa normalisasi standar deviasi) terhadap *grounding* dan perilaku pencarian?

RQ1 dan RQ2 merupakan pertanyaan utama. RQ3 bersifat eksploratif, sehingga perbandingannya tidak akan dikoreksi untuk perbandingan berganda dan hasil yang tidak signifikan akan dilaporkan sebagai tidak konklusif.

## 1.3 Tujuan

Setiap tujuan menjawab satu pertanyaan penelitian. Tujuan pada penelitian ini adalah sebagai berikut.

1. **Untuk RQ1:** melatih agen pencarian dengan RL bergaya GRPO dan LoRA pada satu GPU, lalu mengukur akurasi dan kepatuhan protokolnya dibandingkan dengan model tanpa pelatihan pada *holdout* yang dibekukan.
2. **Untuk RQ2:** merancang *reward* tanpa *judge*, yaitu VEGR, yang memberi *reward* pada jawaban yang benar, sitasi yang benar, dan pencarian yang berguna, lalu membandingkannya dengan *reward* berbasis jawaban saja pada tiga *seed*, baik dari sisi akurasi maupun *grounding*, menggunakan selang kepercayaan (*confidence interval*) *bootstrap* yang memperhitungkan *seed*.
3. **Untuk RQ3:** menguji apakah varian *reward* (penalti bergerbang dan bobot cakupan ×2) serta perubahan *optimizer* (*advantage* tanpa normalisasi standar deviasi) mengubah *grounding*, cakupan bukti, dan stabilitas pelatihan.

## 1.4 Batasan Masalah

Penelitian ini sengaja dibatasi pada ruang lingkup yang sempit. Batasan masalah pada penelitian ini adalah sebagai berikut.

1. **Data dan *retrieval*.** Penelitian ini menggunakan HotpotQA pada *distractor setting*. Agen akan melakukan pencarian dengan BM25 pada **kumpulan tertutup berisi 10 paragraf milik setiap pertanyaan**, dan kumpulan ini selalu memuat kedua paragraf emas. Oleh karena itu, tugas ini diperkirakan jauh lebih mudah dibandingkan dengan pencarian pada web terbuka atau korpus terbuka. Istilah "agen pencarian" pada penelitian ini berarti agen yang melakukan pencarian pada kumpulan tertutup tersebut.
2. **Model.** Model yang digunakan adalah Qwen2.5-7B-Instruct yang diadaptasi dengan LoRA.
3. **Skala.** Setiap *run* pelatihan akan menggunakan 256 pertanyaan selama satu *epoch*, dan evaluasi akan menggunakan 300 pertanyaan *holdout* yang dibekukan. Anggaran komputasi penelitian ini direncanakan sekitar $45.
4. **Label.** VEGR membutuhkan bukti emas pada tingkat kalimat, yang hanya disediakan oleh sedikit *dataset*. Metode ini tidak dapat diterapkan apa adanya pada data yang tidak memiliki label tersebut.
5. **Hal yang tidak diklaim.** Penelitian ini tidak mengusulkan algoritma RL baru maupun jenis *reward* baru, tidak menargetkan akurasi *state-of-the-art*, dan tidak mengklaim bahwa temuannya akan berlaku pula untuk pencarian web terbuka, *dataset* lain, atau model yang lebih besar.
6. **Celah (*loophole*) yang diketahui.** Lingkungan (*environment*) menerima jawaban tanpa sitasi. Hal ini sengaja dipertahankan agar seluruh *run* dapat dibandingkan pada lingkungan yang sama, dan konsekuensinya akan dilaporkan.

## 1.5 Rencana Kegiatan

Rencana kegiatan pada penelitian ini adalah sebagai berikut.

1. **Kajian Pustaka RL untuk Agen Pencarian dan *Reward* Berbasis Bukti.**

    > Mengkaji metode pelatihan agen pencarian dengan RL (Search-R1, GRPO beserta variannya), *reward* bukti yang telah diusulkan (baik berbasis aturan maupun berbasis *LLM judge*), evaluasi sitasi, serta praktik evaluasi RL multi-*seed* untuk mengidentifikasi celah penelitian dan menentukan *baseline* eksperimen.

2. **Persiapan *Dataset* HotpotQA dan Lingkungan Pencarian.**

    > Menyiapkan HotpotQA (*distractor setting*) beserta pembagian data latih, validasi, dan *holdout* beku berisi 300 pertanyaan yang dapat direproduksi. Penulis juga membangun lingkungan pencarian berbasis BM25 pada 10 paragraf milik setiap pertanyaan, lengkap dengan tata bahasa aksi `SEARCH`/`ANSWER`, anggaran 3 pencarian, dan validasi sitasi.

3. **Perancangan *Reward* VEGR dan Skenario Eksperimen.**

    > Merancang fungsi *reward* VEGR (komponen jawaban, bukti, dan pencarian, serta penalti −1 untuk episode tidak valid) beserta *reward outcome-only* sebagai pembanding dan varian-variannya. Penulis juga merancang skenario eksperimen, meliputi *baseline* tanpa pelatihan, metode usulan, dan variasi *reward* maupun *optimizer*, masing-masing pada tiga *seed*.

4. **Implementasi *Pipeline* Pelatihan GRPO dan LoRA.**

    > Mengimplementasikan *pipeline* pelatihan *policy gradient* bergaya GRPO dengan *adapter* LoRA pada Qwen2.5-7B-Instruct agar dapat dijalankan pada satu GPU 48 GB, termasuk pencatatan konfigurasi, metrik pelatihan, dan episode evaluasi untuk menjaga reproduktibilitas.

5. **Pelatihan, Pengujian, dan Analisis.**

    > Menjalankan seluruh skenario eksperimen dan mengevaluasinya pada *holdout* beku menggunakan metrik akurasi jawaban dan metrik *grounding*. Penulis juga melakukan analisis statistik dengan *paired bootstrap* yang memperhitungkan *seed*, serta menganalisis kategori kegagalan dan dinamika pelatihan.

6. **Penyusunan Laporan Tugas Akhir.**

    > Menyusun laporan lengkap yang mencakup latar belakang, kajian pustaka, perancangan sistem, hasil eksperimen, analisis, dan kesimpulan.
