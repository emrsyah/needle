# BAB 2 KAJIAN PUSTAKA

Bab ini membahas landasan teori dan penelitian terkait yang dibutuhkan dalam tugas akhir ini. Subbab 2.1 hingga 2.4 membahas komponen dasar, yaitu model bahasa, *retrieval*, *multi-hop question answering* dengan bukti, dan agen LLM. Subbab 2.5 membahas *reinforcement learning* (RL) untuk LLM, mulai dari *policy gradient* dan PPO hingga GRPO beserta perbaikan-perbaikannya yang terbaru. Subbab 2.6 dan 2.7 mengkaji agen pencarian yang dilatih dengan RL serta *reward* berbasis bukti, yaitu bidang yang menjadi tempat kontribusi tugas akhir ini. Subbab 2.8 membahas *reward hacking* dan cara mengevaluasi hasil RL. Terakhir, Subbab 2.9 merangkum celah penelitian yang dijawab oleh tugas akhir ini.

## 2.1 *Large Language Model* dan *Instruction Tuning*

*Large language model* (LLM) pertama-tama dilatih awal (*pretraining*) dengan tugas memprediksi token berikutnya pada korpus teks berukuran besar, kemudian diadaptasi agar dapat mengikuti instruksi melalui *supervised fine-tuning* dan pelatihan berbasis preferensi. Ouyang dkk. [11] menunjukkan bahwa *fine-tuning* dengan umpan balik manusia membuat model lebih mampu mengikuti maksud pengguna, bahkan ketika model tersebut jauh lebih kecil dibandingkan dengan model aslinya.

Model *open-weight* yang telah melalui *instruction tuning*, seperti keluarga **Qwen2.5** [12], tersedia dalam berbagai ukuran. Model berukuran sekitar 7B parameter cukup kecil untuk di-*fine-tune* pada satu GPU, selama pelatihannya bersifat hemat parameter (Subbab 2.5.4). Model yang sudah mengikuti instruksi juga dapat langsung diminta mengikuti format keluaran tertentu, sehingga dapat menjadi titik awal bagi *reinforcement learning*.

## 2.2 *Retrieval-Augmented Generation* dan BM25

Pengetahuan sebuah LLM bersifat tetap sejak waktu pelatihan dan dapat keliru. ***Retrieval-augmented generation* (RAG)** [34] mengatasi hal ini dengan mengambil dokumen yang relevan lalu menyusun jawaban dengan berlandaskan dokumen tersebut. Pada RAG dasar, *retrieval* hanya dilakukan satu kali, yaitu sebelum proses pembangkitan jawaban. Pertanyaan yang membutuhkan beberapa potong informasi dapat memerlukan beberapa kali *retrieval*, di mana setiap *retrieval* bergantung pada hasil yang diperoleh sebelumnya. Hal inilah yang mendasari pendekatan berbasis agen pada Subbab 2.4.

**BM25** [13] adalah fungsi pemeringkatan leksikal yang bersifat *sparse*, yaitu memberi skor pada sebuah dokumen berdasarkan seberapa sering istilah pada *query* muncul di dalamnya, dibobot oleh kelangkaan setiap istilah, dan dinormalisasi terhadap panjang dokumen. BM25 tidak memerlukan pelatihan dan bersifat deterministik, sehingga *query* yang sama selalu menghasilkan dokumen yang sama. Sifat ini membuatnya banyak dipakai sebagai *retriever* dasar ketika hasil eksperimen harus dapat direproduksi.

## 2.3 *Multi-Hop QA*, HotpotQA, dan Evaluasi Sitasi

**HotpotQA** [6] adalah *dataset multi-hop question answering* yang dibangun dari Wikipedia, di mana untuk menjawab sebuah pertanyaan diperlukan penggabungan informasi dari dua paragraf. HotpotQA menyediakan dua pengaturan evaluasi: pengaturan *distractor*, ketika paragraf yang memuat bukti dicampur dengan paragraf yang tampak relevan tetapi tidak memuat bukti, dan pengaturan *full wiki*, ketika sistem harus mencari sendiri di seluruh Wikipedia.

Selain jawaban, anotator HotpotQA juga menandai ***supporting facts***, yaitu kalimat-kalimat yang dibutuhkan untuk menjawab setiap pertanyaan. *Dataset* ini mengevaluasi jawaban dengan *exact match* (EM) dan F1 tingkat token. Prediksi *supporting facts* dievaluasi secara terpisah sebagai himpunan kalimat, dengan presisi dan *recall* terhadap himpunan kalimat rujukan, **Sup-EM** (himpunan yang diprediksi sama persis dengan himpunan rujukan), dan Sup-F1. **Metrik gabungan** (*joint metrics*), yaitu Joint-EM dan Joint-F1, mensyaratkan jawaban dan *supporting facts* benar secara bersamaan [6].

Label pada tingkat kalimat ini memungkinkan pemeriksaan kebenaran bukti (*evidence*) yang diberikan model dengan aturan sederhana, tanpa memerlukan manusia ataupun model sebagai penilai. Bab 3 menjelaskan bagaimana label ini digunakan dalam tugas akhir ini.

**Evaluasi sitasi.** Gao dkk. [14] memperkenalkan **ALCE**, yaitu *benchmark* untuk LLM yang menghasilkan teks disertai sitasi. ALCE memperlakukan kebenaran dan kualitas sitasi sebagai dimensi yang terpisah, serta memisahkan *citation recall* (apakah setiap pernyataan didukung oleh bagian teks yang disitasi?) dari *citation precision* (apakah setiap sitasi relevan?). Para penulisnya melaporkan bahwa bahkan model terbaik pada saat itu sering kali belum memiliki dukungan sitasi yang lengkap. Pada ALCE, *citation recall* dinilai per pernyataan oleh model *natural language inference* (NLI). Pada *dataset* yang memiliki label bukti tingkat kalimat seperti HotpotQA, pemisahan yang sama dapat dihitung dengan mencocokkan sitasi secara langsung terhadap label tersebut.

## 2.4 Agen LLM dan Penggunaan *Tool*

**Agen LLM** menyelingi penalaran dengan aksi di dalam suatu lingkungan. **ReAct** [1] memberi *prompt* kepada model agar bergantian antara penalaran dan pemanggilan *tool* (misalnya pencarian), serta mendasarkan setiap langkah pada hasil observasi dari langkah-langkah sebelumnya.
*Prompting* saja tidak menjamin bahwa model menggunakan *tool* dengan baik. Jin dkk. [2] berpendapat bahwa LLM yang hanya diberi *prompt* mungkin tidak mengetahui cara berinteraksi dengan mesin pencari secara optimal. Hal ini mendasari pelatihan penggunaan *tool* oleh agen dengan *reinforcement learning*.

## 2.5 *Reinforcement Learning* untuk LLM

### 2.5.1 *Policy Gradient* dan PPO

Pada *fine-tuning* dengan RL, LLM berperan sebagai ***policy***, yaitu diberikan suatu konteks, model menghasilkan aksi (token), dan setiap episode dinilai dengan sebuah *reward*. Metode *policy gradient* menaikkan probabilitas aksi yang menghasilkan *reward* di atas rata-rata. Bentuk paling sederhananya adalah REINFORCE [35], yang mengalikan gradien *log-probability* sebuah episode dengan *reward*-nya, dan biasanya mengurangkan sebuah *baseline* dari *reward* untuk menekan variansi. ***Proximal Policy Optimization*** (PPO) [15] menstabilkan proses ini dengan memotong (*clipping*) rasio antara probabilitas *policy* baru dan *policy* lama, sehingga satu kali pembaruan tidak dapat menggeser *policy* terlalu jauh. *Clipping* ini hanya berpengaruh ketika data yang sama dipakai untuk beberapa kali pembaruan, sehingga *policy* yang diperbarui sempat menjauh dari *policy* yang mengumpulkan data. PPO standar untuk LLM juga melatih jaringan nilai (*critic*) untuk mengestimasi *reward* yang diharapkan, sehingga memori yang dibutuhkan kira-kira menjadi dua kali lipat.

### 2.5.2 GRPO

***Group Relative Policy Optimization*** (GRPO) diperkenalkan pada DeepSeekMath [3]. GRPO menghilangkan *critic*. Sebagai gantinya, GRPO mengambil sampel satu grup keluaran untuk *prompt* yang sama dan menggunakan *reward* setiap keluaran relatif terhadap grupnya, yang dinormalisasi dengan rata-rata dan simpangan baku grup tersebut, sebagai *advantage*. Dengan kata lain, GRPO dapat dipandang sebagai *policy gradient* dengan *baseline* berupa rata-rata grup, ditambah *clipping* bergaya PPO.

**DeepSeek-R1** [4] menunjukkan bahwa GRPO dengan *reward* berbasis aturan yang dapat diverifikasi mampu melatih kemampuan penalaran yang kuat. Pada tahap **DeepSeek-R1-Zero** dan tahap RL penalarannya, *reward* yang digunakan memeriksa akurasi jawaban dan format keluaran dengan aturan, tanpa *reward model* yang dipelajari. VEGR mengikuti prinsip yang sama, yaitu setiap komponen *reward* dapat diperiksa dengan aturan.

Normalisasi grup memiliki efek samping yang penting dalam tugas akhir ini. Ketika seluruh keluaran dalam satu grup menerima *reward* yang sama, setiap *advantage* bernilai nol, sehingga grup tersebut tidak memberikan sinyal pembelajaran.

### 2.5.3 Perbaikan GRPO: Dr. GRPO dan DAPO

Terdapat dua arah penelitian terbaru yang memodifikasi GRPO, yaitu sebagai berikut.

1. **Dr. GRPO** [16] mengidentifikasi bias optimasi pada GRPO dan menghilangkan dua suku normalisasi, yaitu normalisasi panjang respons dan pembagian *advantage* dengan simpangan baku grup.
2. **DAPO** [17] mengusulkan beberapa perubahan untuk RL skala besar. Salah satunya adalah ***dynamic sampling***, yaitu grup yang seluruh keluarannya mendapat *reward* yang sama dibuang, dan grup tambahan diambil hingga *batch* yang berisi banyak grup memiliki sinyal pembelajaran. Perubahan lainnya meliputi *clip-higher* dan *loss* tingkat token.

Subbab 3.6 menjelaskan bagian mana dari kedua perbaikan tersebut yang diuji dalam tugas akhir ini.

### 2.5.4 *Fine-Tuning* Hemat Parameter: LoRA

*Fine-tuning* seluruh 7 miliar parameter dengan RL tidak muat pada satu GPU 48 GB. **LoRA** [18] membekukan bobot hasil *pretraining* dan melatih matriks-matriks kecil berperingkat rendah (*low-rank*) yang ditambahkan pada lapisan-lapisan tertentu. Cara ini memangkas kebutuhan memori menjadi sebagian kecil dari *fine-tuning* penuh, sehingga model berukuran 7B dapat dilatih dengan RL pada satu GPU 48 GB.

## 2.6 *Reinforcement Learning* untuk Agen Pencarian

**Search-R1** [2] merupakan acuan utama tugas akhir ini. Search-R1 melatih LLM untuk mengirimkan *query* pencarian selama penalaran langkah demi langkah, menggunakan RL dengan ***reward* sederhana berbasis hasil akhir** (*outcome-based*), yaitu EM antara jawaban prediksi dan jawaban emas. Search-R1 tidak memakai *reward* format. Ketika aksi model tidak valid, model hanya menerima pesan agar berpikir ulang, bukan penalti. Pelatihan mencakup trajektori *multi-turn*, token hasil *retrieval* di-*mask* dari perhitungan *loss*, dan model dapat dilatih dengan PPO maupun GRPO. Search-R1 melaporkan peningkatan yang besar dibandingkan dengan *baseline* RAG pada tujuh *dataset* QA dengan model Qwen2.5. Gagasan *reward* berbasis EM inilah yang menjadi acuan *baseline outcome-only* pada tugas akhir ini (Subbab 3.5).

**AutoRefine** [19] menambahkan langkah penyempurnaan (*refine*) pengetahuan hasil *retrieval* di antara pencarian. AutoRefine dilatih dengan GRPO menggunakan *reward* kebenaran jawaban sekaligus ***reward* khusus *retrieval***, yang diberikan ketika jawaban emas muncul pada blok *refine*, dan melaporkan peningkatan terbesarnya pada pertanyaan *multi-hop*. AutoRefine juga melaporkan bahwa *reward* yang diberikan langsung atas dokumen yang diambil hanya menghasilkan kenaikan kecil dibandingkan dengan *reward* jawaban saja (rata-rata 0,384 berbanding 0,371) [19, Tabel 8]. Komponen *retrieval recall* pada VEGR termasuk bentuk yang kedua, yaitu *reward* atas dokumen yang diambil.

Beberapa sistem lain pada tahun 2025 juga melatih agen pencarian dengan RL, sebagian besar dengan *reward* berbasis hasil akhir. **R1-Searcher** [20] dan **ReSearch** [21] menggunakan RL untuk mengajarkan LLM kapan dan bagaimana memanggil *tool* pencarian selama penalaran. **DeepResearcher** [22] memperluas pelatihan ini ke lingkungan web dunia nyata. **ZeroSearch** [23] menghindari biaya mesin pencari sungguhan selama pelatihan dengan cara menyimulasikan pencarian. **StepSearch** [24] menambahkan *reward* per langkah pada PPO, yaitu memberi *reward* pada setiap langkah pencarian berdasarkan *information gain* terhadap dokumen emas, bukan hanya pada jawaban akhir.

Sistem-sistem tersebut memiliki dua sifat yang sama, dan tugas akhir ini berangkat dari sifat yang berbeda, yaitu sebagai berikut.

1. Sistem-sistem tersebut dilatih pada skala *multi-GPU*. Search-R1 dilatih "*on a single node with 8 H100 GPUs*" [2, Lampiran B.2], dan AutoRefine "*on 8 NVIDIA A100-80GB GPUs*" [19, Lampiran B.1].
2. *Reward* utamanya adalah kebenaran jawaban. Jawaban yang benar tidak menunjukkan bahwa model menggunakan, atau dapat menunjukkan, bukti di balik jawaban tersebut.

Hasil utama sistem-sistem tersebut juga umumnya dilaporkan tanpa variansi antar-*run* pelatihan. Search-R1 tidak melaporkan *seed* maupun variansi. Tabel utama AutoRefine berasal dari satu *run*, dengan pemeriksaan tiga *seed* yang hanya terdapat pada lampiran makalahnya [19, Lampiran C.2]. CaRR (Subbab 2.7) melaporkan pengulangan evaluasi, bukan pengulangan pelatihan dengan *seed* yang berbeda.

## 2.7 *Grounding* dan *Reward* Berbasis Bukti

Beberapa penelitian telah melangkah lebih jauh dari *reward* tunggal yang bersifat holistik.

**Fine-Grained RLHF** [25] melakukan pelatihan dengan beberapa *reward model*, di mana masing-masing menyasar jenis kesalahan yang berbeda (misalnya ketidakbenaran faktual, ketidakrelevanan, atau ketidaklengkapan), alih-alih satu skor preferensi holistik. Para penulisnya menunjukkan bahwa perubahan cara menggabungkan *reward* mengubah perilaku model. Hal ini mendukung rancangan VEGR yang dibangun per aspek serta eksperimen pembobotan ulang R3.

**_Reward_ bukti dengan LLM *judge* atau verifikator.** Terdapat dua makalah tahun 2026 yang secara langsung membahas bukti dalam RL untuk agen pencarian, yaitu sebagai berikut.

1. **CaRR / C-GRPO** [7] memecah pertanyaan *multi-hop* menjadi rubrik-rubrik *single-hop*. Sebuah LLM penilai (*judge*) memeriksa apakah setiap rubrik didukung oleh konten yang disitasi, dan *reward* rubrik hanya dihitung ketika jawabannya benar. Para penulisnya melaporkan bahwa di bawah GRPO *outcome-only*, agen mempelajari **jawaban jalan pintas** (*shortcut answers*), yaitu menjawab dari sebagian *hop* tanpa memverifikasi batasan lain, dengan pemanggilan *tool* yang lebih sedikit. Temuan ini berkaitan dengan, tetapi tidak identik dengan, hilangnya sitasi yang dilaporkan pada Bab 4.
2. **STAMP** [8] memberi kredit pada setiap langkah pencarian yang pertama kali memunculkan dokumen pendukung yang disitasi, dan menggunakan LLM verifikator untuk memeriksa sitasi. Makalah ini mencatat bahwa *reward* yang memperhatikan sitasi dan *reward* cakupan bukti kini sudah umum dalam arah penelitian ini.

**_Reward_ bukti tanpa LLM *judge*.** Huang dkk. [26] melatih model untuk menghasilkan teks disertai sitasi menggunakan *reward* yang terperinci (*fine-grained*) untuk kebenaran dan kualitas sitasi. *Reward* sitasinya dihitung dengan model *natural language inference* (NLI), bukan dengan aturan, dan setiap eksperimen RL-nya diulang dengan tiga *seed*. Namun, pengaturannya adalah *question answering* berbentuk teks panjang, bukan agen pencarian *multi-turn*. StepSearch (Subbab 2.6) memberi *reward* berbasis aturan pada langkah-langkah pencarian terhadap dokumen emas dan memiliki pembanding PPO/GRPO dengan *reward* jawaban saja pada pengaturan yang sama, tetapi tidak melaporkan pengulangan *seed*.

Beberapa makalah tahun 2026 lainnya juga menyasar bukti dan *retrieval* di dalam RL untuk agen pencarian, yaitu sebagai berikut.

1. PROGRESS [27] memberi *reward* atas cakupan terhadap *query* pencarian esensial yang diusulkan oleh model-model *teacher* yang dibekukan. Pencocokan *query* tersebut dilakukan oleh LLM *judge*.
2. Huang dkk. [28] memberikan kredit pada langkah-langkah *retrieval* antara.
3. BiCAA [29] menambahkan *reward* tingkat langkah untuk langkah yang membawa bukti baru.
4. Zhu dkk. [30] menurunkan *reward* langkah yang padat (*dense*) dari estimasi kegunaan fakta.

**Evaluasi *reward* bukti.** Dua makalah lain merupakan evaluasi, bukan metode pelatihan. HERALD [31] mengaudit *reward* "*proof-of-retrieval*" berbasis aturan. Makalah ini menemukan serangan *citation laundering* dan memperbaikinya dengan memberi penalti pada sitasi terhadap bagian teks yang tidak pernah diambil. Khan dkk. [32] menemukan bahwa agen yang dilatih dengan RLVR berbasis hasil akhir masih melakukan rasionalisasi pasca-hoc (*post-rationalise*) terhadap sitasinya dengan tingkat yang sama seperti model dasarnya.

**Posisi tugas akhir ini.** CaRR, STAMP, dan PROGRESS bergantung pada **LLM *judge* atau verifikator**, yang menambah biaya dan berpotensi menimbulkan bias. Alternatif tanpa LLM *judge* juga sudah ada, baik berbasis model NLI [26] maupun berbasis aturan [24]. Komponen-komponen VEGR bukanlah hal baru (Subbab 3.5), dan VEGR bukan jenis *reward* yang baru. Kontribusi tugas akhir ini adalah sebuah **studi terkontrol**, yaitu *reward* bukti berbasis aturan pada tingkat kalimat untuk agen pencarian *multi-turn*, yang diuji pada tiga *seed* terhadap kontrol *outcome-only* dalam pengaturan yang selebihnya identik, pada *setting* kecil dengan satu GPU di mana setiap komponen bukti dapat dihitung secara tepat dari *supporting facts* emas HotpotQA.

## 2.8 *Reward Hacking* dan Evaluasi RL

***Reward hacking.*** Skalse dkk. [5] mendefinisikan *reward hacking* secara formal, yaitu kondisi ketika optimasi terhadap *reward* proksi yang tidak sempurna menurunkan kinerja pada tujuan yang sebenarnya.^[Judul versi prosiding makalah ini adalah "*Defining and Characterizing Reward Gaming*", sedangkan judul versi arXiv-nya adalah "…*Reward Hacking*".] Para penulisnya berpendapat bahwa proksi yang tidak dapat di-*hack* jarang ditemukan. Gao, Schulman, dan Hilton [33] mengukur efek yang sama pada *reward model* yang dipelajari, yaitu bahwa optimasi proksi yang terlalu keras pada akhirnya merugikan skor yang sebenarnya (skor "emas"). Pada tugas akhir ini, proksinya adalah kebenaran jawaban, sedangkan tujuan sebenarnya adalah jawaban yang benar *dan* ter-*grounding*. Bab 4 melaporkan *policy outcome-only* yang memenuhi proksi tanpa menyitasi bukti. Perilaku tersebut konsisten dengan definisi *reward*-nya (Subbab 3.5), yang tidak pernah menambah *reward* untuk sitasi yang benar, sehingga dalam tugas akhir ini perilaku tersebut dipandang sebagai konsekuensi rancangan *reward* yang dapat diantisipasi, bukan sebagai eksploitasi yang tidak terduga.

**Evaluasi RL dengan sedikit *run*.** Hasil RL sangat bervariasi antar-*seed* acak. Henderson dkk. [9] menunjukkan bahwa perbedaan antar-*seed* saja dapat tampak seperti perbedaan yang signifikan antar-metode. Agarwal dkk. [10] berpendapat bahwa ketika hanya tersedia sedikit *run*, hasil harus dilaporkan sebagai estimasi interval, misalnya selang kepercayaan *bootstrap*, dan tidak hanya sebagai estimasi titik. Oleh karena itu, tugas akhir ini melatih setiap metode utama dengan 3 *seed* dan membandingkan model menggunakan SK *bootstrap* berpasangan yang memperhitungkan *seed* (Subbab 3.8).

## 2.9 Celah Penelitian dan Posisi Penelitian

**Tabel 2.1.** Celah penelitian, bukti dari literatur, dan cara tugas akhir ini menjawabnya.

| Celah | Bukti dari literatur | Cara tugas akhir ini menjawabnya |
|---|---|---|
| **G1** Benar ≠ ter-*grounding* | *Reward outcome-only* (Search-R1) tidak memeriksa bukti; CaRR melaporkan jawaban jalan pintas | Mengukur *grounding* secara langsung dengan bukti emas tingkat kalimat; membandingkan R0 dengan VEGR |
| **G2** *Reward* bukti berbasis aturan pada tingkat kalimat untuk agen pencarian *multi-turn* jarang diuji pada beberapa *seed* terhadap kontrol *outcome-only* | CaRR, STAMP, dan PROGRESS menggunakan LLM *judge* atau verifikator; Huang dkk. [26] menggunakan model NLI dan tiga *seed*, tetapi bukan untuk agen pencarian *multi-turn*; StepSearch berbasis aturan dan memiliki kontrol *outcome-only*, tetapi tidak melaporkan pengulangan *seed* | VEGR yang diperiksa dengan aturan dibandingkan dengan *reward outcome-only* pada tiga *seed* |
| **G3** RL untuk agen pencarian dilaporkan pada skala *multi-GPU* | Search-R1 (8 × H100) dan AutoRefine (8 × A100) | Tugas yang disederhanakan pada satu GPU 48 GB dengan LoRA, sekitar 2 jam per *run* |
| **G4** Hasil satu *run* mengira *noise* antar-*seed* sebagai efek metode | Henderson [9]; Agarwal [10]; Search-R1 tidak melaporkan *seed* maupun variansi | 3 *seed* per metode; SK *bootstrap* yang memperhitungkan *seed* |

Secara ringkas, tugas akhir ini tidak mengusulkan algoritma RL baru dan tidak mengklaim akurasi *state-of-the-art*. Tugas akhir ini mengajukan pertanyaan yang lebih sempit dalam *setting* yang sepenuhnya terkontrol: **ketika bukti dapat diperiksa secara tepat, apakah pemberian *reward* atas bukti tersebut mengubah perilaku agen pencarian, dibandingkan dengan pemberian *reward* atas jawaban saja?**
