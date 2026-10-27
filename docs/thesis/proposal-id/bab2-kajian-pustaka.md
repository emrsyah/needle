# BAB 2 KAJIAN PUSTAKA

Bab ini membahas landasan teori dan penelitian terkait yang dibutuhkan dalam tugas akhir ini. Subbab 2.1 hingga 2.4 membahas komponen dasar, yaitu model bahasa, *retrieval*, *multi-hop question answering* dengan bukti, dan agen LLM. Subbab 2.5 membahas *reinforcement learning* (RL) untuk LLM, mulai dari PPO hingga GRPO beserta perbaikan-perbaikannya yang terbaru. Subbab 2.6 dan 2.7 mengkaji agen pencarian yang dilatih dengan RL serta *reward* berbasis bukti, yaitu bidang yang menjadi tempat kontribusi penelitian ini. Subbab 2.8 membahas *reward hacking* dan cara mengevaluasi hasil RL. Terakhir, Subbab 2.9 merangkum celah penelitian dan bagaimana penelitian ini akan menjawabnya.

## 2.1 *Large Language Model* dan *Instruction Tuning*

*Large language model* (LLM) pertama-tama dilatih awal (*pretraining*) dengan tugas memprediksi token berikutnya pada korpus teks berukuran besar, kemudian diadaptasi agar dapat mengikuti instruksi melalui *supervised fine-tuning* dan pelatihan berbasis preferensi. Ouyang dkk. [11] menunjukkan bahwa *fine-tuning* dengan umpan balik manusia membuat model lebih mampu mengikuti maksud pengguna, bahkan ketika model tersebut jauh lebih kecil dibandingkan dengan model aslinya.

Penelitian ini akan menggunakan **Qwen2.5-7B-Instruct** [12], yaitu model *open-weight* yang telah melalui *instruction tuning* dan dipublikasikan dalam bentuk laporan teknis. Model berukuran 7B cukup kecil untuk dilakukan *fine-tuning* pada satu GPU 48 GB, selama pelatihannya bersifat hemat parameter (Subbab 2.5.4). *Instruction tuning* juga berpengaruh terhadap eksperimen, karena model tanpa pelatihan diharapkan sudah dapat mencoba mengikuti format aksi sehingga *reinforcement learning* memiliki titik awal.

## 2.2 *Retrieval-Augmented Generation* dan BM25

Pengetahuan sebuah LLM bersifat tetap sejak waktu pelatihan dan dapat keliru. ***Retrieval-augmented generation* (RAG)** mengatasi hal ini dengan mengambil dokumen yang relevan lalu menyusun jawaban dengan berlandaskan dokumen tersebut. Pada RAG dasar, *retrieval* hanya dilakukan satu kali, yaitu sebelum proses pembangkitan jawaban. Pertanyaan yang membutuhkan beberapa potong informasi dapat memerlukan beberapa kali *retrieval*, di mana setiap *retrieval* bergantung pada hasil yang diperoleh sebelumnya. Hal inilah yang mendasari pendekatan berbasis agen pada Subbab 2.4.

Penelitian ini akan menggunakan **BM25** sebagai *retriever* [13]. BM25 adalah fungsi pemeringkatan leksikal yang bersifat *sparse*, yaitu memberi skor pada sebuah dokumen berdasarkan seberapa sering istilah pada *query* muncul di dalamnya, dibobot oleh kelangkaan setiap istilah, dan dinormalisasi terhadap panjang dokumen. BM25 tidak memerlukan pelatihan dan bersifat deterministik, sehingga *query* yang sama selalu menghasilkan dokumen yang sama. Sifat deterministik inilah yang membuat *reward* bukti pada Bab 3 dapat diverifikasi secara tepat, karena selalu diketahui *query* mana yang memunculkan dokumen mana.

## 2.3 *Multi-Hop QA*, HotpotQA, dan Evaluasi Sitasi

**HotpotQA** [6] adalah *dataset multi-hop question answering* yang dibangun dari Wikipedia, di mana untuk menjawab sebuah pertanyaan diperlukan penggabungan informasi dari dua paragraf. Karakteristik HotpotQA yang akan digunakan dalam penelitian ini adalah sebagai berikut.

1. ***Distractor setting.*** Setiap pertanyaan disertai 10 paragraf, yaitu 2 paragraf emas (*gold*) ditambah 8 paragraf pengecoh (*distractor*) yang diambil agar tampak relevan. Sistem harus menemukan paragraf emas di antara paragraf-paragraf pengecoh tersebut.
2. ***Supporting facts.*** Anotator menandai kalimat-kalimat yang tepat dibutuhkan untuk menjawab setiap pertanyaan, dalam bentuk pasangan (judul paragraf, indeks kalimat). *Dataset* ini mengevaluasi jawaban dengan *exact match* (EM) dan F1 tingkat token, dan mengevaluasi prediksi *supporting facts* secara terpisah dengan **Sup-EM** (himpunan yang diprediksi sama dengan himpunan emas) dan Sup-F1. **Metrik gabungan** (*joint metrics*), yaitu Joint-EM dan Joint-F1, mensyaratkan jawaban dan *supporting facts* benar secara bersamaan. Penelitian ini akan melaporkan metrik-metrik tersebut berdampingan dengan ukuran *grounding* yang dirancang dalam penelitian ini (Subbab 3.7).

Label pada tingkat kalimat ini memungkinkan pemeriksaan kebenaran bukti (*evidence*) yang diberikan model dengan aturan sederhana, tanpa memerlukan manusia ataupun model sebagai penilai.

**Evaluasi sitasi.** Gao dkk. [14] memperkenalkan **ALCE**, yaitu *benchmark* untuk LLM yang menghasilkan teks disertai sitasi. ALCE memperlakukan kebenaran dan kualitas sitasi sebagai dimensi yang terpisah, serta memisahkan *citation recall* (apakah setiap pernyataan didukung oleh bagian teks yang disitasi?) dari *citation precision* (apakah setiap sitasi relevan?). Para penulisnya melaporkan bahwa bahkan model terbaik pada saat itu sering kali belum memiliki dukungan sitasi yang lengkap. Penelitian ini akan menggunakan pemisahan yang sama antara cakupan bukti (*evidence coverage*, sebuah bentuk *recall*) dan presisi sitasi (Bab 3). Perbedaan utamanya terletak pada *setting*-nya, yaitu pada penelitian ini sitasi diperiksa terhadap kalimat emas HotpotQA, bukan dinilai oleh model *natural language inference*.

## 2.4 Agen LLM dan Penggunaan *Tool*

**Agen LLM** menyelingi penalaran dengan aksi di dalam suatu lingkungan. **ReAct** [1] memberi *prompt* kepada model agar bergantian antara penalaran dan pemanggilan *tool* (misalnya pencarian), serta mendasarkan setiap langkah pada hasil observasi dari langkah-langkah sebelumnya. Agen pada penelitian ini akan mengikuti *loop* bergaya ReAct yang disederhanakan dengan tata bahasa aksi yang ketat, yaitu pada setiap giliran agen melakukan pencarian atau memberikan jawaban akhir disertai sitasi (Bab 3).

*Prompting* saja tidak menjamin bahwa model menggunakan *tool* dengan baik. Jin dkk. [2] berpendapat bahwa LLM yang hanya diberi *prompt* mungkin tidak mengetahui cara berinteraksi dengan mesin pencari secara optimal. Hal ini mendasari pelatihan penggunaan *tool* oleh agen dengan *reinforcement learning*.

## 2.5 *Reinforcement Learning* untuk LLM

### 2.5.1 *Policy Gradient* dan PPO

Pada *fine-tuning* dengan RL, LLM berperan sebagai ***policy***, yaitu diberikan suatu konteks, model menghasilkan aksi (token), dan setiap episode dinilai dengan sebuah *reward*. Metode *policy gradient* menaikkan probabilitas aksi yang menghasilkan *reward* di atas rata-rata. ***Proximal Policy Optimization*** (PPO) [15], yang dipublikasikan sebagai *preprint*, menstabilkan proses ini dengan memotong (*clipping*) rasio antara probabilitas *policy* baru dan *policy* lama, sehingga satu kali pembaruan tidak dapat menggeser *policy* terlalu jauh. PPO standar untuk LLM juga melatih jaringan nilai (*critic*) untuk mengestimasi *reward* yang diharapkan, sehingga memori yang dibutuhkan kira-kira menjadi dua kali lipat.

### 2.5.2 GRPO

***Group Relative Policy Optimization*** (GRPO) diperkenalkan pada DeepSeekMath [3], yang juga dipublikasikan sebagai *preprint*. GRPO menghilangkan *critic*. Sebagai gantinya, GRPO mengambil sampel sekelompok keluaran untuk *prompt* yang sama dan menggunakan *reward* setiap keluaran relatif terhadap kelompoknya, yang dinormalisasi dengan rata-rata dan simpangan baku kelompok tersebut, sebagai *advantage*.

**DeepSeek-R1** [4] menunjukkan bahwa GRPO dengan *reward* berbasis aturan yang dapat diverifikasi mampu melatih kemampuan penalaran yang kuat. *Reward* yang digunakan memeriksa akurasi jawaban dan format keluaran, tanpa *reward model* yang dipelajari. VEGR yang diusulkan dalam penelitian ini mengikuti prinsip yang sama, yaitu setiap komponen *reward* dapat diperiksa dengan aturan.

Normalisasi kelompok memiliki efek samping yang penting bagi penelitian ini. Ketika seluruh keluaran dalam satu kelompok menerima *reward* yang sama, setiap *advantage* bernilai nol, sehingga kelompok tersebut tidak memberikan sinyal pembelajaran.

### 2.5.3 Perbaikan GRPO: Dr. GRPO dan DAPO

Terdapat dua arah penelitian terbaru yang memodifikasi GRPO dan relevan dengan penelitian ini, yaitu sebagai berikut.

1. **Dr. GRPO** [16] mengidentifikasi bias optimasi pada GRPO dan menghilangkan suku normalisasi panjang dan simpangan baku. Penelitian ini hanya akan menguji perubahan yang kedua, yaitu menghitung *advantage* tanpa membagi dengan simpangan baku (*run* R4, "GRPO tanpa normalisasi std"). Dengan demikian, R4 merupakan ablasi parsial, bukan Dr. GRPO.
2. **DAPO** [17] mengusulkan beberapa perubahan untuk RL skala besar. Salah satunya adalah ***dynamic sampling***, yaitu kelompok yang seluruh keluarannya mendapat *reward* yang sama dibuang, dan kelompok tambahan diambil hingga *batch* memiliki sinyal pembelajaran. Gagasan ini berkaitan langsung dengan masalah kelompok tanpa sinyal pembelajaran yang diuraikan pada Subbab 2.5.2.

### 2.5.4 *Fine-Tuning* Hemat Parameter: LoRA

*Fine-tuning* seluruh 7 miliar parameter dengan RL tidak muat pada satu GPU. **LoRA** [18] membekukan bobot hasil *pretraining* dan melatih matriks-matriks kecil berperingkat rendah (*low-rank*) yang ditambahkan pada lapisan-lapisan tertentu. Cara ini memangkas kebutuhan memori menjadi sebagian kecil dari *fine-tuning* penuh, dan hal inilah yang memungkinkan penelitian ini direncanakan pada satu GPU 48 GB.

## 2.6 *Reinforcement Learning* untuk Agen Pencarian

**Search-R1** [2] merupakan acuan utama penelitian ini. Search-R1 melatih LLM untuk mengirimkan *query* pencarian selama penalaran langkah demi langkah, menggunakan RL dengan ***reward* sederhana berbasis hasil akhir** (*outcome-based*), yaitu apakah jawaban akhirnya benar. Pelatihan mencakup trajektori *multi-turn*, dan token hasil *retrieval* di-*mask* dari perhitungan *loss*. Search-R1 melaporkan peningkatan yang besar dibandingkan dengan *baseline* RAG pada tujuh *dataset* QA dengan model Qwen2.5. *Baseline outcome-only* pada penelitian ini (R0) akan mengikuti rancangan *reward* yang sama.

**AutoRefine** [19], yang dipublikasikan di NeurIPS, menambahkan langkah penyempurnaan pengetahuan hasil *retrieval* di antara pencarian. AutoRefine dilatih dengan GRPO menggunakan *reward* kebenaran jawaban sekaligus ***reward* khusus *retrieval***, dan melaporkan peningkatan terbesarnya pada pertanyaan *multi-hop*. Hal ini mendukung gagasan yang juga digunakan pada VEGR, yaitu bahwa memberi *reward* pada kualitas pencarian antara dapat membantu lebih dari sekadar jawaban akhir.

Beberapa sistem lain pada tahun 2025 juga melatih agen pencarian dengan RL, sebagian besar dengan *reward* berbasis hasil akhir. **R1-Searcher** [20] dan **ReSearch** [21] menggunakan RL untuk mengajarkan LLM kapan dan bagaimana memanggil *tool* pencarian selama penalaran. **DeepResearcher** [22] memperluas pelatihan ini ke lingkungan web dunia nyata. **ZeroSearch** [23] menghindari biaya mesin pencari sungguhan selama pelatihan dengan cara menyimulasikan pencarian. **StepSearch** [24] menambahkan *reward* per langkah pada PPO, yaitu memberi *reward* pada setiap langkah pencarian terhadap bukti emas, bukan hanya pada jawaban akhir.

Sistem-sistem tersebut memiliki dua sifat yang sama, dan penelitian ini berangkat dari sifat yang berbeda, yaitu sebagai berikut.

1. Sistem-sistem tersebut dilatih pada skala *multi-GPU*. Search-R1 dilatih "*on a single node with 8 H100 GPUs*" [2, Lampiran B.2], dan AutoRefine "*on 8 NVIDIA A100-80GB GPUs*" [19, Lampiran B.1].
2. *Reward* utamanya adalah kebenaran jawaban. Jawaban yang benar tidak menunjukkan bahwa model menggunakan, atau dapat menunjukkan, bukti di balik jawaban tersebut.

Sistem-sistem tersebut juga melaporkan hasil utamanya dari satu kali *run* pelatihan. Search-R1 tidak melaporkan *seed* maupun variansi. Tabel utama AutoRefine berasal dari satu *run*, dengan pemeriksaan tiga *seed* yang hanya terdapat pada lampiran (Lampiran C.2). CaRR (Subbab 2.7) melatih setiap konfigurasi satu kali dan hanya mengulang evaluasinya.

## 2.7 *Grounding* dan *Reward* Berbasis Bukti

Beberapa penelitian telah melangkah lebih jauh dari *reward* tunggal yang bersifat holistik.

**Fine-Grained RLHF** [25] melakukan pelatihan dengan beberapa *reward model*, di mana masing-masing menyasar jenis kesalahan yang berbeda (misalnya ketidakbenaran faktual, ketidakrelevanan, atau ketidaklengkapan), alih-alih satu skor preferensi holistik. Para penulisnya menunjukkan bahwa perubahan cara menggabungkan *reward* mengubah perilaku model. Hal ini mendukung rancangan VEGR yang dibangun per aspek serta rencana eksperimen pembobotan ulang R3.

Terdapat dua makalah tahun 2026 yang secara langsung membahas bukti dalam RL untuk agen pencarian, yaitu sebagai berikut.

1. **CaRR / C-GRPO** [7] memecah pertanyaan *multi-hop* menjadi rubrik-rubrik *single-hop*. Sebuah LLM penilai (*judge*) memeriksa apakah setiap rubrik didukung oleh konten yang disitasi, dan *reward* rubrik hanya dihitung ketika jawabannya benar. Para penulisnya melaporkan bahwa di bawah GRPO *outcome-only*, agen mempelajari **jawaban jalan pintas** (*shortcut answers*) dan melakukan lebih sedikit pemanggilan *tool*. Penelitian ini akan memeriksa apakah perilaku jalan pintas serupa juga muncul pada *baseline outcome-only* (R0).
2. **STAMP** [8], yang dipublikasikan sebagai *preprint*, memberi kredit pada setiap langkah pencarian yang pertama kali memunculkan dokumen pendukung yang disitasi, dan menggunakan LLM verifikator untuk memeriksa sitasi. Makalah ini mencatat bahwa *reward* yang memperhatikan sitasi dan *reward* cakupan bukti kini sudah umum dalam arah penelitian ini.

***Reward* sitasi berbasis aturan dan berbasis model tanpa LLM *judge*.** *Reward* bukti tanpa penilai (*judge-free*) sebenarnya sudah ada. Huang dkk. [26] melatih model untuk menghasilkan teks disertai sitasi menggunakan *reward* yang terperinci (*fine-grained*) untuk kebenaran dan kualitas sitasi. StepSearch (Subbab 2.6) memberi *reward* pada langkah-langkah pencarian terhadap bukti emas. Beberapa *preprint* tahun 2026 juga menyasar bukti dan *retrieval* di dalam RL untuk agen pencarian, yaitu sebagai berikut.

1. PROGRESS [27] memberi *reward* atas cakupan terhadap *query* pencarian esensial yang diusulkan oleh model-model *teacher* yang dibekukan.
2. Huang dkk. [28] memberikan kredit pada langkah-langkah *retrieval* antara.
3. BiCAA [29] menambahkan *reward* tingkat langkah untuk langkah yang membawa bukti baru.
4. Zhu dkk. [30] menurunkan *reward* langkah yang padat (*dense*) dari estimasi kegunaan fakta.

Selain itu, terdapat dua makalah lain yang merupakan evaluasi, bukan metode pelatihan. HERALD [31] mengaudit *reward* "*proof-of-retrieval*" berbasis aturan. Makalah ini menemukan serangan *citation laundering* dan memperbaikinya dengan memberi penalti pada sitasi terhadap bagian teks yang tidak pernah diambil, yaitu aturan yang sama dengan yang akan ditegakkan oleh kegagalan bukti (*evidence failure*) pada VEGR. Khan dkk. [32] menemukan bahwa agen yang dilatih dengan RLVR berbasis hasil akhir masih melakukan rasionalisasi pasca-hoc (*post-rationalise*) terhadap sitasinya dengan tingkat yang sama seperti model dasarnya.

**Posisi penelitian ini.** CaRR dan STAMP bergantung pada **LLM *judge* atau verifikator**, yang menambah biaya dan berpotensi menimbulkan bias, tetapi alternatif berbasis aturan juga sudah ada sebagaimana diuraikan di atas. Komponen-komponen VEGR bukanlah hal baru (Subbab 3.3), dan VEGR bukan jenis *reward* yang baru. Kontribusi yang diusulkan dalam penelitian ini adalah sebuah **studi terkontrol**, yaitu *reward* bukti tanpa *judge* yang diperiksa dengan aturan, yang akan diuji pada tiga *seed* terhadap kontrol *outcome-only*, dalam *setting* kecil dengan satu GPU di mana setiap komponen bukti dapat dihitung secara tepat dari *supporting facts* emas HotpotQA.

## 2.8 *Reward Hacking* dan Evaluasi RL

***Reward hacking.*** Skalse dkk. [5] mendefinisikan *reward hacking* secara formal, yaitu kondisi ketika optimasi terhadap *reward* proksi yang tidak sempurna menurunkan kinerja pada tujuan yang sebenarnya. Judul versi prosiding makalah ini adalah "*Defining and Characterizing Reward Gaming*", sedangkan judul versi arXiv-nya adalah "…*Reward Hacking*". Para penulisnya berpendapat bahwa proksi yang tidak dapat di-*hack* jarang ditemukan. Gao, Schulman, dan Hilton [33] mengukur efek yang sama pada *reward model* yang dipelajari, yaitu bahwa optimasi proksi yang terlalu keras pada akhirnya merugikan skor yang sebenarnya (skor "emas"). Pada penelitian ini, proksinya adalah kebenaran jawaban, sedangkan tujuan sebenarnya adalah jawaban yang benar *dan* ter-*grounding*. Diduga *policy outcome-only* dapat memenuhi proksi tersebut tanpa menyitasi bukti. Apabila hal ini terjadi, perilaku tersebut mengikuti secara langsung dari definisi *reward* (Subbab 3.3), sehingga lebih tepat digambarkan sebagai konsekuensi rancangan *reward* yang dapat diprediksi daripada sebagai eksploitasi yang tidak terduga.

**Evaluasi RL dengan sedikit *run*.** Hasil RL sangat bervariasi antar-*seed* acak. Henderson dkk. [9] menunjukkan bahwa perbedaan antar-*seed* saja dapat tampak seperti perbedaan yang signifikan antar-metode. Agarwal dkk. [10] berpendapat bahwa ketika hanya tersedia sedikit *run*, hasil harus dilaporkan sebagai estimasi interval (misalnya selang kepercayaan (*confidence interval*, CI) *bootstrap*), dan tidak pernah sebagai *run* tunggal. Oleh karena itu, penelitian ini akan melatih setiap metode utama dengan 3 *seed* dan membandingkan model menggunakan SK *bootstrap* berpasangan yang memperhitungkan *seed* (Subbab 3.7).

## 2.9 Celah Penelitian dan Rencana Penelitian

**Tabel 2.1.** Celah penelitian, bukti dari literatur, dan rencana penelitian ini untuk menjawabnya.

| Celah | Bukti dari literatur | Rencana penelitian ini untuk menjawabnya |
|---|---|---|
| **G1** Benar ≠ ter-*grounding* | *Reward outcome-only* (Search-R1) tidak memeriksa bukti; CaRR melaporkan jawaban jalan pintas | Mengukur *grounding* secara langsung dengan bukti emas tingkat kalimat; membandingkan R0 dengan VEGR |
| **G2** *Reward* bukti tanpa *judge* sudah ada, tetapi jarang diuji pada beberapa *seed* terhadap kontrol *outcome-only* | CaRR dan STAMP menggunakan LLM *judge* atau verifikator; *reward* berbasis aturan (Huang dkk. [26]; StepSearch; HERALD) dilaporkan tanpa kontrol semacam itu | Membandingkan VEGR yang diperiksa dengan aturan dengan *reward outcome-only* pada tiga *seed* |
| **G3** RL untuk agen pencarian dilaporkan pada skala *multi-GPU* | Search-R1 (8 × H100) dan AutoRefine (8 × A100) | Satu GPU 48 GB dengan LoRA, dengan anggaran yang direncanakan sekitar $45 untuk seluruh eksperimen |
| **G4** Hasil satu *run* mengira *noise* antar-*seed* sebagai efek metode | Henderson [9]; Agarwal [10]; Search-R1 dan CaRR melatih setiap konfigurasi satu kali | 3 *seed* per metode; SK *bootstrap* yang memperhitungkan *seed* |

Secara ringkas, penelitian ini tidak mengusulkan algoritma RL baru dan tidak menargetkan akurasi *state-of-the-art*. Penelitian ini mengajukan pertanyaan yang lebih sempit dalam *setting* yang sepenuhnya terkontrol: **ketika bukti dapat diperiksa secara tepat, apakah pemberian *reward* atas bukti tersebut mengubah perilaku agen pencarian, dibandingkan dengan pemberian *reward* atas jawaban saja?**
