# BAB 5 KESIMPULAN DAN SARAN

## 5.1 Kesimpulan

Tugas akhir ini mengkaji apakah agen pencarian yang dilatih dengan *reinforcement learning* dapat diberi *reward* atas jawaban yang ter-*grounding*, bukan hanya jawaban yang benar, tanpa *LLM judge* dan dengan anggaran mahasiswa.

Agen Qwen2.5-7B-Instruct dengan LoRA dilatih menggunakan *policy gradient* bergaya GRPO dengan normalisasi per kelompok pada HotpotQA. Agen melakukan pencarian dengan BM25 pada kumpulan tertutup berisi 10 paragraf milik setiap pertanyaan, lalu menjawab disertai sitasi tingkat kalimat yang dapat diperiksa berbasis aturan terhadap *supporting facts* emas HotpotQA. Enam belas *run* dilatih pada satu GPU dengan total biaya sekitar $49, yaitu lima konfigurasi *reward* dan *optimizer* masing-masing dengan tiga *seed*, ditambah satu *run* diagnostik. Setiap *run* dievaluasi pada 300 pertanyaan *holdout* yang dibekukan, dengan selang kepercayaan (*confidence interval*, CI) *bootstrap* yang memperhitungkan *seed* (selanjutnya disingkat SK).

**RQ1: Apakah pelatihan GRPO meningkatkan performa agen pencarian 7B dibandingkan dengan model tanpa pelatihan?** Ya, sebagian besar melalui pembelajaran protokol. Setiap metode memperoleh kenaikan sekitar +20 poin EM dibandingkan dengan model tanpa pelatihan (22,3%), dan setiap SK yang memperhitungkan *seed* tidak memuat nol. Sebagian besar kenaikan tersebut berasal dari kemampuan menghasilkan aksi yang valid. Model tanpa pelatihan gagal pada 130 dari 300 episode, sedangkan *run* yang dilatih gagal paling banyak pada 22 episode. EM pada episode yang selesai hanya naik dari 39,4% menjadi 45,2% (R1).

**RQ2: Apakah VEGR mencegah jalan pintas dari *reward* berbasis jawaban saja tanpa kehilangan akurasi?** Ya, dalam pengaturan penelitian ini. Rinciannya adalah sebagai berikut.

- **Akurasi.** *Reward* berbasis jawaban saja dan paket VEGR lengkap mencapai akurasi yang sama (43,6% vs. 42,6% EM; +1,0, SK yang memperhitungkan *seed* [−3,6; +5,4]).
- ***Grounding*.** *Precision-grounded* EM VEGR **24,4 poin lebih tinggi** (SK [+8,8; +37,3]).
- **Hilangnya sitasi pada *reward* berbasis jawaban saja.** Dua dari tiga *seed* dengan *reward* berbasis jawaban saja berhenti memberikan sitasi. *Rollout* pelatihan menjelaskan penyebabnya: dengan *reward* tersebut, sitasi yang tidak valid bernilai −1 sedangkan sitasi yang benar tidak memperoleh apa pun, dan pada awal pelatihan sekitar separuh dari seluruh percobaan sitasi tidak valid. *Seed* ketiga tetap memberikan sitasi, tetapi bergeser ke perilaku pencarian berulang, yang tidak dihukum maupun diberi *reward* oleh *reward* tersebut.
- **Cakupan kesimpulan perbandingan.** R1 berbeda dari R0 pada enam komponen *reward* dan pada jumlah *update*-nya, sehingga hasil ini berlaku untuk paket VEGR secara keseluruhan.
- **Benar, tetapi belum lengkap.** VEGR membuat sitasi menjadi benar, tetapi belum lengkap. Cakupan buktinya tidak lebih tinggi daripada model tanpa pelatihan, dan Joint-EM-nya rendah serta bervariasi antar-*seed*.

**RQ3: Bagaimana pengaruh varian *reward* dan perubahan *optimizer* terhadap *grounding* dan perilaku pencarian?** Hasil ini bersifat eksploratif, dan tidak ada varian yang terbukti lebih baik daripada VEGR biasa. Rinciannya adalah sebagai berikut.

- Penalti bergerbang (R2) dan *advantage* tanpa normalisasi standar deviasi (R4) memberikan hasil yang tidak konklusif. Kenaikan +6 poin EM yang tampak pada satu *seed* R2 tidak tereplikasi.
- Penggandaan bobot cakupan (R3) menaikkan cakupan bukti dari 0,44 menjadi 0,57 dan memberikan Joint-EM HotpotQA terbaik (+6,3 dibandingkan dengan R1, SK [+0,2; +13,1]). Namun, R3 juga menurunkan *precision-grounded* EM sebesar 8,0 poin (SK [−16,1; −0,2]). Metode mana yang lebih baik bergantung pada metrik yang digunakan.
- Pada satu *seed* yang diuji, *dynamic sampling* (R0-dyn) menaikkan akurasi *reward* berbasis jawaban saja sebesar 8,0 poin EM, bersamaan dengan jumlah pertanyaan pelatihan yang dua kali lipat, tetapi tidak mencegah hilangnya sitasi.

## 5.2 Kontribusi

Kontribusi tugas akhir ini adalah sebagai berikut.

1. **VEGR**, yaitu *reward* bukti tanpa *judge* pada tingkat kalimat yang juga menegakkan protokol aksi. Komponen-komponennya secara individual bukanlah hal baru; kontribusinya terletak pada kombinasi komponen tersebut dan evaluasinya yang terkontrol.
2. **Penjelasan tentang bagaimana *reward* berbasis jawaban saja kehilangan sitasi**, dalam pengaturan kecil yang dapat direproduksi dan perilakunya dapat dideteksi berbasis aturan. Sitasi yang tidak valid diberi penalti, sedangkan sitasi yang benar tidak pernah diberi *reward*. Tugas akhir ini juga menunjukkan bahwa pemulihan *update* yang hilang akibat kelompok dengan varians nol tidak mencegah keruntuhan tersebut pada *seed* yang diuji.
3. **Perbandingan multi-*seed* yang terkontrol** dengan SK yang memperhitungkan *seed*. Perbandingan ini menunjukkan bagaimana hasil dari satu *seed* dapat menyesatkan, dan bahwa kesimpulan tentang *grounding* bergantung pada metrik *grounding* yang digunakan.
4. ***Pipeline* satu GPU yang terdokumentasi.** Satu *run* membutuhkan biaya sekitar $2,25. Pembagian data, konfigurasi, log pelatihan, hasil per episode, dan skrip analisis tersedia di repositori. *Adapter* LoRA dari 15 dari 16 *run* disimpan; *adapter* R1 *seed* 7 hilang.

## 5.3 Keterbatasan

Keterbatasan penelitian ini adalah sebagai berikut.

1. **Pengaturan.** *Retrieval* dilakukan dengan BM25 pada 10 paragraf milik setiap pertanyaan, pada satu *dataset*, sehingga masalah pencariannya jauh lebih mudah dibandingkan dengan pencarian web terbuka. Model tanpa pelatihan sudah mengambil 83% dokumen emas.
2. **Validitas eksternal.** VEGR membutuhkan bukti emas pada tingkat kalimat, yang hanya disediakan oleh sedikit *dataset*, dan VEGR hanya diuji pada satu *dataset* dengan satu ukuran model. Hasil penelitian ini belum menyatakan apa pun tentang pengaturan tanpa label tersebut.
3. **Skala.** Pelatihan menggunakan 256 pertanyaan per *run* selama satu *epoch*, pada model 7B dengan LoRA, dan berhenti sebelum *reward* konvergen.
4. **Statistik.** Tiga *seed* dan 300 pertanyaan *holdout* cukup untuk mendeteksi efek yang besar, seperti selisih *grounding* antara R1 dan R0. Namun, keduanya tidak cukup untuk mendeteksi selisih EM sebesar beberapa poin, dan frekuensi keruntuhan R0 pada dasarnya tidak diketahui (2 dari 3 *seed*, SK 95% Clopper–Pearson [0,09; 0,99]).
5. **Desain.** Perbandingan R1–R0 mengubah seluruh paket *reward* dan jumlah *update*. R2–R4 dan R0-dyn bersifat eksploratif dan tidak dikoreksi untuk perbandingan berganda. R0-dyn hanya memiliki satu *seed* dan mencapai 250 dari target 256 *update*.
6. **Tumpang tindih metrik dan *reward*.** VEGR dilatih pada fakta emas yang sama dengan yang digunakan oleh metrik *grounding*. *Precision-grounded* EM bersifat longgar terhadap bukti yang tidak disertakan, dan tidak ada pemeriksaan atribusi yang independen dari label emas yang dijalankan.
7. **Algoritma.** *Update* pelatihan hanya berupa satu langkah *on-policy* untuk setiap kelompok pertanyaan, sehingga *clipping* PPO tidak pernah aktif. Algoritma yang digunakan adalah *policy gradient* bergaya GRPO, bukan GRPO secara penuh.
8. **Lingkungan.** Lingkungan menerima jawaban tanpa sitasi, pemisah sitasi bertabrakan dengan karakter '&' pada beberapa judul, dan bobot *reward* tidak di-*tuning*.

## 5.4 Etika dan Dampak

Jawaban yang disertai sitasi tampak dapat dipercaya, dan kesan tersebut dapat membuat pengguna terlalu memercayainya. Sitasi yang cocok dengan kalimat emas menunjukkan bahwa agen menunjuk pada bukti yang relevan. Namun, sitasi tersebut tidak menunjukkan bahwa jawaban benar-benar diturunkan dari bukti itu, sehingga *precision-grounded* EM bukanlah ukuran *faithfulness*. Bahkan agen VEGR terbaik masih menyisakan sekitar seperempat jawaban benarnya dengan setidaknya satu sitasi yang bukan kalimat emas, dan agen tersebut jarang mengutip bukti secara lengkap. Oleh karena itu, sistem yang dibangun dengan cara ini sebaiknya menyajikan sitasi sebagai penunjuk untuk pemeriksaan, bukan sebagai pembuktian.

Penelitian ini hanya menggunakan data dan model publik. HotpotQA didistribusikan di bawah lisensi CC BY-SA 4.0. Qwen2.5-7B-Instruct dirilis oleh tim Qwen di bawah lisensi modelnya sendiri, yang mengatur setiap penggunaan maupun redistribusi *adapter* hasil pelatihan. Total jejak komputasinya tergolong kecil, yaitu sekitar $49 waktu sewa pada satu GPU L40S (Tabel 3.2). Temuan penelitian ini terbatas pada satu *dataset* dengan label bukti emas, satu model, dan kumpulan *retrieval* tertutup, sehingga tidak boleh dibaca sebagai bukti tentang agen pencarian web terbuka.

## 5.5 Saran

Saran untuk penelitian selanjutnya adalah sebagai berikut.

1. **Memisahkan bagian-bagian paket VEGR.** Latih VEGR tanpa komponen buktinya (cakupan dan presisi), dengan tetap mempertahankan komponen jawaban dan pencarian. Ini adalah ablasi yang belum dilakukan dan dibutuhkan untuk mengetahui apakah kenaikan *grounding* berasal dari pemberian *reward* pada bukti atau dari *reward* yang lebih padat dengan jumlah *update* yang lebih banyak. Varian yang hanya berisi komponen bukti akan melengkapi gambaran tersebut.
2. **Menutup celah tanpa sitasi dan memperbaiki format sitasi.** Wajibkan minimal satu sitasi, dan gunakan pemisah yang tidak mungkin muncul pada judul (Subbab 4.7). Setelah itu, uji apakah pelatihan dengan *reward* berbasis jawaban saja masih kehilangan sitasi, atau justru menemukan jalan pintas lain.
3. **Memberi *reward* pada bukti yang lengkap tanpa kehilangan presisi.** Contohnya adalah *reward* berbasis F1 dari himpunan sitasi, atau komponen cakupan yang hanya dihitung untuk sitasi emas, alih-alih bobot yang lebih besar pada cakupan saja (R3).
4. **Memeriksa atribusi tanpa label emas.** Gunakan pemeriksaan sitasi berbasis NLI [14] untuk menguji apakah kenaikan yang diperoleh VEGR tetap bertahan pada ukuran yang tidak digunakan dalam pelatihannya.
5. **Menambahkan *baseline* berbasis *prompt* saja.** Karena sebagian besar kenaikan pada RQ1 merupakan pembelajaran protokol, bandingkan dengan *prompt few-shot* yang menampilkan contoh-contoh format aksi.
6. **Kredit pada tingkat langkah.** Berikan *reward* pada setiap pencarian berdasarkan informasi yang ditambahkannya, seperti pada StepSearch [24] dan STAMP [8], alih-alih hanya menilai episode akhir.
7. ***Retrieval* yang lebih sulit dan data yang lebih banyak.** Perluas pengaturan ke korpus terbuka, ke *dataset multi-hop* lain yang memiliki bukti emas seperti 2WikiMultiHopQA dan MuSiQue, serta ke set pelatihan yang lebih besar.
8. **Lebih banyak *seed* untuk *run* diagnostik.** Jalankan R0-dyn pada ketiga *seed* dengan jumlah pertanyaan pelatihan yang sama dengan R0, sehingga efek *dynamic sampling* dapat dipisahkan dari efek data tambahan.
