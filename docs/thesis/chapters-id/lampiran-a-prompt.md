# LAMPIRAN A — PROMPT LENGKAP AGEN

Lampiran ini memuat *prompt* lengkap yang diterima agen untuk satu pertanyaan *holdout* nyata. Teks tersebut dihasilkan oleh fungsi `build_prompt` (`src/needle/inference/runner.py`), yaitu fungsi yang sama dengan yang digunakan pada pelatihan dan evaluasi. *Prompt* ini diberikan kepada Qwen2.5-7B-Instruct sebagai satu pesan pengguna melalui *chat template* milik model.

## A.1 Cara Contoh Direproduksi

- **Pertanyaan:** `5abd94525542992ac4f382d2` dari *holdout* beku (`configs/evaluation/holdout_v1_ids.json`). Baris data ini diambil dari *split* validasi *distractor* HotpotQA di Hugging Face (`hotpotqa/hotpot_qa`, `distractor/validation-00000-of-00001.parquet`) dan dikonversi ke format asli HotpotQA.
- **Riwayat:** sebuah `SearchEnvironment` (top-k = 3, anggaran = 3) memutar ulang dua pencarian yang dilakukan R1 *seed* 8 pada episode *holdout*-nya (`results/2026-10-10-batch/r1_full_seed8/episodes.jsonl`). Fungsi `build_prompt` kemudian dipanggil pada titik ketika agen memilih aksi akhirnya.
- ***Supporting facts* emas:** `2014 S/S|0` dan `Winner (band)|0`.
- **Aksi agen pada titik ini:** `ANSWER[YG Entertainment] CITATIONS[2014 S/S|0; Winner (band)|0]`. Jawaban tersebut benar dan kedua kalimat emas disitasi (Tabel 3.1, episode A).

## A.2 Teks *Prompt*

*Prompt* direproduksi apa adanya. Spasi di awal beberapa kalimat memang terdapat pada teks sumber HotpotQA.

```text
You are a careful evidence-seeking QA agent.
Return exactly one action: SEARCH[query] or ANSWER[answer] CITATIONS[title|index; ...].
SEARCH[query] is search-only: never append CITATIONS[...] to a SEARCH action.
ANSWER[...] is terminal; when the search budget reaches 0, answer immediately.
Example: ANSWER[United States] CITATIONS[Ursula K. Le Guin|1; The Left Hand of Darkness|1].
Use one title|index citation per entry; separate entries with semicolons, never comma-separated indices.
Do not add a period, explanation, or any text after the final closing bracket.
Copy citation titles exactly as shown in the search results, including punctuation and accents.
Never invent a document title; use CITATIONS[] when no citation can be validated.
Use only the supplied search results for citations.
Question: 2014 S/S is the debut album of a South Korean boy group that was formed by who?
Remaining search budget: 1
Search results so far:
Search 1: S/S 2014 debut South Korean boy group
[2014 S/S]
0: 2014 S/S is the debut album of South Korean group WINNER.
1:  It was released on August 12, 2014 by the group's record label, YG Entertainment.
2:  The members were credited for writing the lyrics and composing the majority of the album's songs.
[History (band)]
0: History (Korean: 히스토리 ) was a South Korean boy group formed by LOEN Entertainment in 2013.
1:  They debuted on April 26, 2013 with "Dreamer", featuring the narration of their labelmate IU.
2:  They were LOEN Entertainment's first boy group.
3:  They officially disbanded on May 12, 2017.
[SF9 (band)]
0: SF9 (Korean: 에스에프나인 ; shortened from Sensational Feeling 9) is a South Korean boy group formed by FNC Entertainment.
1:  SF9 is the company's first dance boy group to ever debut.
2:  SF9 debuted on October 5, 2016 with the release of their first single album "Feeling Sensation".
Search 2: WINNER YG Entertainment
[Winner (band)]
0: Winner (Hangul: 위너), often stylized as WINNER, is a South Korean boy group formed in 2013 by YG Entertainment and debuted in 2014.
1:  It currently consists of four members, Jinwoo, Seunghoon, Mino and Seungyoon.
2:  Originally a five-piece group with Taehyun, who later departed from the group in November 2016.
[2014 S/S]
0: 2014 S/S is the debut album of South Korean group WINNER.
1:  It was released on August 12, 2014 by the group's record label, YG Entertainment.
2:  The members were credited for writing the lyrics and composing the majority of the album's songs.
[History (band)]
0: History (Korean: 히스토리 ) was a South Korean boy group formed by LOEN Entertainment in 2013.
1:  They debuted on April 26, 2013 with "Dreamer", featuring the narration of their labelmate IU.
2:  They were LOEN Entertainment's first boy group.
3:  They officially disbanded on May 12, 2017.
```

## A.3 Catatan tentang *Prompt*

- **Giliran pertama.** Sebelum ada pencarian, bagian riwayat berisi `(none)` dan sisa anggaran bernilai 3.
- **Dokumen berulang.** Setiap pencarian mencantumkan seluruh 3 hasil teratasnya, bahkan ketika sebuah dokumen sudah dikembalikan oleh pencarian sebelumnya ("2014 S/S" dan "History (band)" muncul dua kali di atas). *Prompt* bertambah panjang pada setiap pencarian, yang menjadi salah satu alasan anggaran dibatasi hingga 3.
- **Alasan adanya aturan format.** Aturan mengenai format sitasi (satu `title|index` per entri, judul persis, tanpa teks tambahan di akhir) ditujukan untuk kegagalan protokol yang teramati pada model tanpa pelatihan, yang kehilangan 37 dari 300 episode *holdout* akibat aksi yang tidak sesuai format (Subbab 4.2).
- **Celah tanpa sitasi.** Baris "use CITATIONS[] when no citation can be validated" itulah yang membuat daftar sitasi kosong menjadi aksi yang valid (celah yang dibahas pada Subbab 3.3).
