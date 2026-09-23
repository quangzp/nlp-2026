# REPORT — Báo cáo tiến độ dự án: Sửa lỗi chính tả tiếng Việt (VSEC)

> **Mốc báo cáo**: 23/09/2026 · Ghi nhận toàn bộ công việc từ đầu dự án đến thời điểm hiện tại.
> Các lần chạy thí nghiệm trên Kaggle (GPU T4): ngày **22/09/2026** (nb0, nb1, nb2) và **22–23/09/2026** (nb3, nb3b).
> Số liệu trong báo cáo được trích **trực tiếp từ output đã chạy** nhúng trong các notebook tại `notebooks/` — không phải con số mục tiêu.
> Tham chiếu: `PROJECT.md` (bối cảnh, ràng buộc), `DESIGN.md` (thiết kế, lộ trình, ngưỡng quyết định).

---

## 1. Tóm tắt nhanh

### 1.1. Tiến độ các notebook

| Notebook | Vai trò | Phase | Trạng thái | Ngày chạy (Kaggle) |
|---|---|---|---|---|
| `notebooks/nb0-data-pre.ipynb` | Chuẩn bị dữ liệu: NFC, dedupe, chia train/val | 0 | ✅ Hoàn thành | 22/09/2026 (manifest 07:16:53) |
| `notebooks/nb1-align-annotate.ipynb` | Align âm tiết + pseudo-annotation bộ test | 0 | ✅ Hoàn thành | 22/09/2026 (report 08:49:01) |
| `notebooks/nb2-pilot-dict-noise.ipynb` | Pilot 1 (từ điển) + Pilot 2 (nhiễu tổng hợp) | 1 | ✅ Hoàn thành | 22/09/2026 (report 15:11:16) |
| `notebooks/nb3-baseline-train-eval.ipynb` | Baseline LoRA BARTpho (Run 1 thuần / Run 2 + augmentation) | 1 | ✅ Hoàn thành | 22–23/09/2026 |
| `notebooks/nb3b-zeroshot-baseline.ipynb` | Baseline đối chứng Identity & Zero-shot (eval-only) | 1 | ✅ Hoàn thành | 22–23/09/2026 |

→ **Phase 0 + Phase 1 đã hoàn thành đầy đủ.** Dự án đang ở **điểm quyết định Phase 2** (chọn trục dữ liệu / trục kiến trúc theo bảng quyết định `DESIGN.md` §5).

### 1.2. Kết quả quan trọng nhất (chi tiết ở các mục sau)

1. **Baseline đạt Detection F1 80,45% trên test** (Run 2, LoRA BARTpho + augmentation); correction accuracy tại vị trí detect đúng 67,94%.
2. **Augmentation (pha câu sạch + nhiễu tổng hợp) hiệu quả và rẻ**: Detection F1 test 76,43% → 80,45% (+4,0 điểm), recall +6,3 điểm, không làm over-correction xấu đi đáng kể.
3. **Over-correction KHÔNG phải vấn đề nghiêm trọng** trên bộ test này: 1,82–1,88% (baseline) < ngưỡng 2–3% của `DESIGN.md` §5; và con số này **có sẵn ở base model** (zero-shot 1,56%) → fine-tune không phải nguồn gốc chính.
4. **Non-word rate 22,7% < ngưỡng 40%** → theo quy tắc quyết định, **bỏ sớm hướng kiến trúc hybrid từ điển (C)**, giữ seq2seq làm trục chính.
5. Bộ test tự thu thập có **10,9% câu sạch** (653/5.983) → đo được Clean Retention; nhưng **dày lỗi hơn hẳn VSEC** (46,7% câu có ≥4 edit block so với 0,8% của VSEC) → domain shift đáng kể.
6. Quy trình Phase 0 **sạch về data leakage**: cả 3 invariant đều PASS; dedupe chéo VSEC↔test chỉ 0 trùng (thấp hơn dự kiến).

---

## 2. Phase 0 — `nb0-data-pre.ipynb`: chuẩn bị dữ liệu

### 2.1. Đã làm gì

- Load VSEC từ Hugging Face (`nguyenthanhasia/vsec-vietnamese-spell-correction`, split `train` duy nhất).
- Chuẩn hóa Unicode **NFC** + tokenize âm tiết (tách theo whitespace, khớp quy ước VSEC).
- **Sanity-check vị trí annotation**: đối chiếu `syllable_annotations[].position` với token index.
- **Dedupe trong VSEC**: exact → near-duplicate (`rapidfuzz.token_set_ratio`, ngưỡng 92).
- **Tự dò bộ test 6k** (file `.csv` có `text` + `label`) → dedupe trong test → **dedupe chéo VSEC↔test** (trùng thì giữ phía test, loại khỏi VSEC).
- **Chia train/val 90/10** stratified theo `error_count` binned (1/2/3/≥4), **seed 42**, chia theo câu, thực hiện sau tất cả bước dedupe.
- Xuất `vsec_train.jsonl`, `vsec_val.jsonl`, `test_normalized.jsonl`, `manifest.json`.

### 2.2. Thống kê

**Dòng dữ liệu VSEC (raw 9.341 câu):**

| Bước | Số câu |
|---|---|
| VSEC raw | 9.341 |
| Sau exact-dedupe (−0) | 9.341 |
| Sau near-dedupe (−71) | **9.270** |
| Sau cross-dedupe vs test (−0) | 9.270 |
| **Train (90%)** | **8.343** |
| **Val (10%)** | **927** |

**Phân bố `error_count` VSEC raw**: {1: 7.853, 2: 1.224, 3: 191, 4: 47, 5: 20, 6: 3, 7: 2, 8: 1} — tổng **11.202 lỗi**; `has_errors` = true trên 100% câu.

**Bộ test 6.000 câu** (nguồn: Kaggle dataset `cquangnguynl/nlp-2026-test-set/6000.csv`, cột `text` + `label`): exact-dedupe −0, near-dedupe −17 → **5.983 câu** sau dedupe.

**Bins train/val** (min(error_count, 4)): train {1: 7.016, 2: 1.089, 3: 172, ≥4: 66} · val {1: 780, 2: 121, 3: 19, ≥4: 7}.

### 2.3. Phát hiện

- **6/9.341 câu (112 annotation) lệch vị trí** giữa `syllable_annotations[].position` và token index sau tokenize — xác nhận các edge-case đã cảnh báo trong `PROJECT.md` §3.2 (âm tiết tách sai, dính từ, ký tự thừa). Không fail cứng, xử lý triệt để bằng aligner ở nb1.
- **Cả 3 invariant PASS**: len(train)+len(val) = sau-dedupe; giao text train∩val = 0; giao VSEC(train+val)∩test = 0 → **không rò rỉ dữ liệu giữa các tập**.
- Dedupe chéo VSEC↔test chỉ **0 câu trùng** (exact + near) — thấp hơn rủi ro dự kiến trong `DESIGN.md` §2.

---

## 3. Phase 0 — `nb1-align-annotate.ipynb`: align âm tiết & pseudo-annotation bộ test

### 3.1. Đã làm gì

Bộ test tự thu thập **chỉ có `text` + `corrected_text`** (không có annotation âm tiết) → xây **aligner Levenshtein block-based** (`align-v1`) suy ra pseudo-annotation (vị trí lỗi, cặp lỗi/sửa) để tính được 3 nhóm chỉ số bắt buộc trên test:

- Kiểm chứng aligner **3 tầng trên gold VSEC** (nơi có annotation thật) trước khi tin kết quả trên test.
- Align toàn bộ 5.983 câu test; kiểm tra **roundtrip invariant** (`src + edit_blocks == tgt`).
- Thống kê câu sạch, độ dày lỗi, phân bố loại edit block; **QA soát tay 50 mẫu** stratified (seed 42).
- Xuất `test_aligned.jsonl`, `qa_samples.json`, `align_report.json`.

### 3.2. Thống kê

**Kiểm chứng aligner trên gold VSEC (9.270 câu):**

| Thang đo | Kết quả |
|---|---|
| (a) Count agreement (số lỗi đếm được khớp gold) | 8.970/9.270 — **96,8%** |
| (b) Recall gold pairs | 10.702/11.151 — **96,0%** |
| (b) Precision edit blocks | 10.292/10.782 — **95,5%** |
| (b) Câu khớp hoàn hảo (count + đúng hết pairs) | 8.489/9.270 — **91,6%** |
| (c) Vị trí trong span (câu eligible) | 67/67 — **100%** |
| Roundtrip invariant (VSEC + test, 15.253 câu) | **100% PASS** |

**Bộ test sau align (5.983 câu):**

| Chỉ số | Giá trị |
|---|---|
| Align failed (tuyệt đối) | **0** |
| **Câu sạch (0 edit block)** | **653 (10,9%)** |
| Câu suspect (edit dày, ratio ≥ 30%) | 2.316 (38,7%) |
| Câu chỉ khác dấu câu | 0 (9 block punct_only) |

**Phân bố loại edit block (test)**: substitute 18.959 · multi 2.729 · split 176 · insert 73 · merge 1.

**Độ dày lỗi — test (pseudo) vs VSEC (gold)**:

| Số lỗi/câu | Test | VSEC |
|---|---|---|
| 0 | 653 (10,9%) | 0 (0,0%) |
| 1 | 672 (11,2%) | 7.796 (84,1%) |
| 2 | 952 (15,9%) | 1.210 (13,1%) |
| 3 | 914 (15,3%) | 191 (2,1%) |
| ≥4 | 2.792 (46,7%) | 73 (0,8%) |

### 3.3. Phát hiện

- **Trả lời mục treo `DESIGN.md` §11.1**: bộ test **có 10,9% câu sạch** → metric Clean Retention (giữ nguyên câu sạch) khả thi trên test.
- **Domain shift về độ dày lỗi**: test dày lỗi hơn hẳn VSEC (46,7% câu ≥4 block so với 0,8%) — mọi kết quả đánh giá trên test phải đọc kèm đặc điểm này; test về bản chất **khó hơn** phân phối VSEC.
- Aligner đủ tin cậy để làm pseudo-annotation (~96% agreement với gold), tuy nhiên sai số ~4% tồn tại — xem mục 9 (hạn chế).
- QA 50 mẫu soát tay cho thấy bản sửa của test có cả lỗi thật (telex/vni kiểu `dduwợc`, `nhưbg`), lỗi gõ nhầm phím vùng miền (`cko`→`cho`, `fó`→`phó`), và một số edit phi chính tả (chèn số liệu, viết tắt `BHXH`) — nhất quán với nguồn báo chí.

---

## 4. Phase 1 — `nb2-pilot-dict-noise.ipynb`: Pilot từ điển & nhiễu tổng hợp

### 4.1. Đã làm gì

**Pilot 1 — Non-word rate (trần của detector từ điển):**
- Tải bảng âm tiết công khai (nguồn độc lập, không suy từ dữ liệu — chống leakage): `vietnameselanguage/syllable` — **7.884 âm tiết** (có ghi SHA256); bảng sensitivity (generative) **17.974 âm tiết** để đo độ nhạy.
- Đo trên **VSEC-val (927 câu)**: phân loại từng lỗi gold thành non-word (âm tiết sai ∉ bảng) / real-word (âm tiết sai hợp lệ, cần ngữ cảnh) / structural; đo OOV trên phía token sạch.

**Pilot 2 — Chất lượng nhiễu tổng hợp (nguồn cho augmentation):**
- Xây **confusion pairs thực nghiệm** từ `correction_pairs` của **train** (chỉ train — đúng anti-leakage `DESIGN.md` §9).
- Xây bộ rule sinh lỗi: `keyboard` (phím QWERTY liền kề/thêm/xóa ký tự), `tone` (hoán dấu thanh), `vowel` (hoán nguyên âm ơ/â, ê/e…), `regional` (s/x, tr/ch, r/d/gi, ng/n…).
- Sinh 200 câu nhiễu demo từ câu sạch `corrected_text` train (sampler deterministic, invariant vị trí PASS); **QA soát tay 50 mẫu** theo rubric.
- Xuất `syllable_table.json`, `noise_model.json`, `noise_qa_samples.json`, `pilot_report.json`.

### 4.2. Thống kê

**Pilot 1 (VSEC-val):**

| Thang đo | Giá trị |
|---|---|
| Lỗi gold phân loại | 252 non-word / 858 real-word / 1 structural / 1 skip-digit |
| **Non-word rate (bảng chính, phía lỗi)** | **22,7%** |
| Clean-side OOV (phía token sạch, 27.708 token) | 456 — **1,65%** |
| Sensitivity (bảng generative 17.974) | non-word 15,4% · clean OOV 2,36% |

- Top lỗi non-word: `gía, nôị, qúa, tâp, dộ, cac, só, hoc, vâỵ, vầ…`
- Top lỗi real-word: `đô, các, khoẻ, hang, thoả, trong, dung, qua, uỷ…` (âm tiết hợp lệ nhưng sai ngữ cảnh).
- Top token sạch bị flag nhầm (OOV thật): `HS, TH, KH, Honda, GV, Internet, FDI, DN, ODA…` — chủ yếu tên riêng/từ viết tắt.

**Pilot 2 (train 8.343 câu):**

| Thang đo | Giá trị |
|---|---|
| Confusion pairs từ gold | 10.039 → giữ **8.915** (loại 314 rỗng, 805 multi-token, 5 punct/digit) |
| Âm tiết sạch có ≥1 confusion đã biết | 1.285 |
| Error-vocabulary (âm tiết dạng lỗi) | 2.843 |
| Top pairs | thỏa→thoả (55) · dùng→dung (49) · những→nhưng (48) · hệ→hê (47) · hiện→hiên (45) · ủy→uỷ (44) |
| Nhiễu sinh (200 câu demo) | 248 lỗi; nguồn: keyboard 114 · empirical 113 · tone 16 · vowel 3 · regional 2 |

### 4.3. Phát hiện & quyết định

- **Non-word rate 22,7% < ngưỡng 40%** (`DESIGN.md` §5) → **cân nhắc bỏ sớm hướng C (hybrid từ điển âm tiết)**: chỉ ~1/4 lỗi bắt được bằng tra bảng với precision cao; phần lớn lỗi là real-word cần ngữ cảnh. Kết luận nhất quán trên cả bảng sensitivity (15,4%).
- **Trần precision của detector từ điển ~98,3%** (clean OOV 1,65%) — đủ cao, nhưng recall trần chỉ ~23% nên không đủ làm trục chính.
- **Calibration nhiễu tổng hợp: lệch 28,2 điểm %** — non-word rate nhiễu sinh **51,6%** so với lỗi thật train **23,4%**; nhiễu sinh dày non-word/keyboard hơn phân phối lỗi thật. Ghi nhận để hiệu chỉnh nếu dùng ở Phase 2 (nb3 đã dùng bản chưa hiệu chỉnh cho Run 2 — xem 5.3).
- QA 50 mẫu: nhiễu sinh tự nhiên ở mức chấp nhận được (đúng vị trí, đọc được, đa dạng nguồn lỗi); invariant vị trí và determinism PASS.

---

## 5. Phase 1 — `nb3-baseline-train-eval.ipynb`: baseline LoRA BARTpho + augmentation

### 5.1. Đã làm gì

- Fine-tune **`vinai/bartpho-syllable`** (fp16, T4) bằng **LoRA**: r=16, α=32, dropout 0,05, target q/v/k/out_proj → trainable **4.718.592/482.514.944 tham số (0,98%)**; batch 8 × grad-accum 4, lr 2e-4, **6 epochs**, max len 256, gradient checkpointing; decode greedy.
- **Run 1 (Pure baseline)**: 8.343 câu VSEC train thuần.
- **Run 2 (Baseline + augmentation)**: 13.347 câu = Run 1 + **2.502 câu sạch** (từ `corrected_text` train, input=output) + **2.502 câu nhiễu** (sinh từ noise model nb2) — augmentation **chỉ trên train**, đúng anti-leakage; có assert từng record `split=train`.
- Hàm `evaluate_predictions` tính đủ 3 nhóm chỉ số bắt buộc (`PROJECT.md` §6) trên **val và test** + phân tích stratified non-word/real-word; sanity check PASS; identity floor check ở nb3b.
- Xuất `predictions_val_run{1,2}.jsonl`, `predictions_test_run{1,2}.jsonl`, `eval_report.json`, 2 LoRA adapter.

### 5.2. Kết quả chính

| Chỉ số | VAL · Run 1 | VAL · Run 2 | TEST · Run 1 | TEST · Run 2 |
|---|---|---|---|---|
| Detection Precision | 80,51% | 79,00% | 86,03% | 86,67% |
| Detection Recall | 72,27% | 70,78% | 68,77% | 75,07% |
| **Detection F1** | **76,16%** | **74,67%** | **76,43%** | **80,45%** |
| Correction Accuracy @TP | 84,75% | 86,53% | 66,20% | 67,94% |
| Over-correction Rate | 0,75% | 0,81% | 1,82% | 1,88% |
| Clean Sent Retention | 33,33% (1/3) | 33,33% (1/3) | 86,68% (≈566/653) | 85,91% (≈561/653) |

**Stratified trên VSEC-val (recall phát hiện):** non-word (gold 296): Run 1 75,68% / Run 2 74,32% · real-word (gold 847): Run 1 71,07% / Run 2 69,54%.

**Mẫu over-correction (FP) tiêu biểu Run 1:** phần lớn là **deletion** — token đúng bị xóa thay vì thay thế: `khỏe → ""`, `Hòa → ""`, `tùy → ""`, `các → ""`; một số thay sai: `thỏa → nhằm`.

### 5.3. Phát hiện

- **Augmentation hiệu quả trên test**: Detection F1 test 76,43% → **80,45%** (+4,0 điểm), recall 68,77% → 75,07% (+6,3 điểm), precision tăng nhẹ; correction acc cũng tăng (66,20% → 67,94%). Trên val, Run 2 không thắng về F1 (74,67% so với 76,16%) — lợi ích của augmentation thể hiện rõ nhất trên test (bộ khó, dày lỗi).
- **Correction accuracy tụt mạnh val → test** (84,75% → 66,20% tại Run 1): hệ quả domain shift + test dày lỗi hơn hẳn (nb1); model detect được vị trí nhưng sửa đúng nội dung khó hơn trên test.
- **Over-correction thấp: 1,82–1,88% (test) < ngưỡng 2–3%** của `DESIGN.md` §5 → theo quy tắc quyết định, **vấn đề over-correction không nghiêm trọng trên bộ test này** → ưu tiên phân tích stratified sâu / pipeline, không dồn lực chống over-correction.
- Clean Retention trên test tốt (~86%) dù model chỉ từng thấy ≤1 câu 0 lỗi trong val — nhờ 2.502 câu sạch được pha vào Run 2 (duy trì ~86% tương đương Run 1).
- FP pattern **deletion** là đặc điểm nổi bật cần xử lý ở Phase 2 (post-processing hoặc ràng buộc decode).

---

## 6. Phase 1 — `nb3b-zeroshot-baseline.ipynb`: baseline đối chứng Identity & Zero-shot

### 6.1. Đã làm gì

Notebook **eval-only, không train** (chống leakage hiển nhiên — không nạp train, không fit gì):

1. **Identity baseline** (`pred = src`) — sàn lý thuyết của mọi mô hình, có assert floor.
2. **Zero-shot baseline** — BARTpho-syllable gốc fp16, greedy, **không LoRA/không train**, chạy trên cả val (927) và test (5.983).

Mục đích: trả lời 2 câu hỏi — (i) LoRA fine-tune đóng góp bao nhiêu? (ii) Over-correction ~1,8% do fine-tune hay do base model? Kết hợp với nb3 tạo **bảng so sánh 4-way**; xuất `zeroshot_eval_report.json` (file riêng, không ghi đè nb3).

### 6.2. Kết quả chính

**Mốc gold (từ identity check)**: VSEC-val 1.143 gold edits · 3 câu sạch; test 24.885 gold edits · 653 câu sạch.

| Chỉ số | VAL Identity | VAL Zero-shot | VAL R1 | VAL R2 | TEST Identity | TEST Zero-shot | TEST R1 | TEST R2 |
|---|---|---|---|---|---|---|---|---|
| Detection P | 0% | 29,49% | 80,51% | 79,00% | 0% | 50,85% | 86,03% | 86,67% |
| Detection R | 0% | 5,60% | 72,27% | 70,78% | 0% | 9,88% | 68,77% | 75,07% |
| Detection F1 | 0% | 9,41% | 76,16% | 74,67% | 0% | 16,55% | 76,43% | 80,45% |
| Correction Acc @TP | 0% | 28,12% (18/64) | 84,75% | 86,53% | 0% | 13,91% (342/2.459) | 66,20% | 67,94% |
| Over-correction | 0% | 0,57% | 0,75% | 0,81% | 0% | 1,56% | 1,82% | 1,88% |
| Clean Retention | 100% (3/3) | 0% (0/3) | 33,33% (1/3) | 33,33% (1/3) | 100% (653/653) | 83,77% (547/653) | 86,68% | 85,91% |

Stratified val (recall): non-word — Identity 0% / Zero-shot 4,73% / R1 75,68% / R2 74,32%; real-word — Identity 0% / Zero-shot 5,90% / R1 71,07% / R2 69,54%.

### 6.3. Phát hiện

- **LoRA fine-tune đóng góp rất lớn**: Detection F1 val 9,41% (zero-shot) → 76,16% (Run 1) — BARTpho gốc gần như không tự sửa lỗi chính tả khi được yêu cầu trực tiếp.
- **Over-correction có sẵn ở base model**: zero-shot test 1,56% ≈ Run 1 1,82% → fine-tune **không phải nguồn gốc chính** của over-correction; nó là đặc tính của kiến trúc/decode seq2seq (cả zero-shot lẫn fine-tune đều có pattern deletion FP: `khỏe → ""`, `Hòa → ""`, và FP dấu câu `: → .`, `/ → và`).
- **Zero-shot trên test cao hơn val** về hầu hết chỉ số (F1 16,55% vs 9,41%; Clean Retention 83,77% vs 0%): nhất quán với domain shift nb1 — test có nhiều câu/đoạn dài sạch hơn và lỗi loãng hơn, base model có xu hướng copy → được "điểm miễn phí" trên clean retention.
- Identity khớp đúng floor lý thuyết trên cả 2 tập → pipeline đo lường nhất quán giữa nb3 và nb3b (cùng `align-v1`, cùng bộ sanity check).

---

## 7. Tổng hợp phát hiện chính (cross-phase)

1. **Nền dữ liệu sạch, chống leakage đúng quy trình** (`DESIGN.md` §9): dedupe trước chia tập, split theo câu stratified seed 42, 3 invariant PASS, augmentation/bảng âm tiết chỉ dùng nguồn train hoặc nguồn công khai độc lập. Dedupe chéo VSEC↔test chỉ 0 trùng.
2. **Bộ test tự thu có 10,9% câu sạch nhưng dày lỗi hơn hẳn VSEC** (46,7% câu ≥4 block vs 0,8%): mọi số liệu trên test phải đọc kèm đặc điểm này; test khó hơn phân phối VSEC rõ rệt (correction acc tụt ~18 điểm val→test).
3. **Trục kiến trúc có lời giải sớm**: non-word rate 22,7% < 40% → bỏ hướng hybrid từ điển (C) làm trục chính; seq2seq giữ vai trò trung tâm, hướng pipeline bảo thủ (A) vẫn là ứng viên nếu xử lý FP deletion.
4. **Augmentation là lever rẻ và hiệu quả**: +2×2.502 câu (sạch + nhiễu) → F1 test +4,0 điểm, recall +6,3 điểm; đồng thời duy trì Clean Retention ~86%.
5. **Over-correction không phải vấn đề nghiêm trọng trên bộ test này** (1,8–1,9% < 2–3%) và có sẵn ở base model → hạ ưu tiên "chống over-correction", chuyển sang nâng recall/correction accuracy.
6. **Điểm yếu còn mở**: correction accuracy trên test còn thấp (66–68%); FP deletion của model; noise model chưa hiệu chuẩn (non-word sinh 51,6% vs thật 23,4% — lệch 28,2 điểm).

## 8. Trạng thái lộ trình & bước tiếp theo đề xuất (Phase 2)

Theo lộ trình `DESIGN.md` §4 và bảng quy tắc quyết định §5:

| Kết quả đo được | Ngưỡng | Quyết định đề xuất |
|---|---|---|
| Over-correction baseline 1,82–1,88% (test) | < 2–3% → "không nghiêm trọng" | Không dồn lực chống over-correction; ưu tiên phân tích stratified sâu |
| Non-word rate 22,7% | < 40% → "bỏ C sớm" | Bỏ hybrid từ điển (C) làm trục chính; (tùy chọn) giữ làm ablation nhỏ |
| Augmentation F1 test +4,0 điểm | — | Giữ augmentation làm thành phần mặc định; hiệu chỉnh noise model (lệch 28,2 điểm) trước khi scale tỷ lệ pha |
| FP pattern deletion | — | Phân tích/post-processing chống deletion; cân nhắc pipeline bảo thủ (A) |

Việc còn mở (không bắt buộc, theo `DESIGN.md`): full-FT ablation 1 lần cuối; factorial 2×2 hai trục nếu kịp thời gian; phê duyệt phạm vi Phase 2.

## 9. Hạn chế của đo lường (đọc kết quả cần lưu ý)

- **Metric trên test dựa trên pseudo-annotation** do aligner suy ra (đã kiểm chứng ~96% agreement với gold VSEC, roundtrip 100%) — nhưng không phải gold thật; sai số ~4% của aligner nằm trong cả gold lẫn prediction trên test.
- **Clean Retention trên val chỉ có 3 câu sạch** (VSEC 100% câu có lỗi) — không có ý nghĩa thống kê; con số đáng tin là trên test (653 câu). Số tuyệt đối val/test trong bảng 5.2 được suy từ tỷ lệ × số câu sạch.
- **Stratified gold nhỏ**: 296 non-word / 847 real-word trên val — chênh lệch vài điểm % giữa Run 1/Run 2 chưa đủ kết luận thống kê.
- **Số liệu là snapshot** các lần chạy Kaggle 22–23/09/2026 (seed 42, greedy decode); re-run có thể lệch nhỏ do phi tất định phần cứng.
- Bộ test 6k chỉ có `text` + `label` tự thu thập; chất lượng bản sửa của test là pseudo-gold (đã QA soát tay 50 mẫu, tỷ lệ suspect 38,7% được đo nhưng chưa adjudicate).

## 10. Danh mục artifact đã sinh (output Kaggle)

| Notebook | File output |
|---|---|
| nb0 | `vsec_train.jsonl`, `vsec_val.jsonl`, `test_normalized.jsonl`, `manifest.json` |
| nb1 | `test_aligned.jsonl`, `qa_samples.json`, `align_report.json` |
| nb2 | `syllable_table.json`, `noise_model.json`, `noise_qa_samples.json`, `pilot_report.json` |
| nb3 | `predictions_val_run{1,2}.jsonl`, `predictions_test_run{1,2}.jsonl`, `eval_report.json`, `lora_adapter_run{1,2}/` |
| nb3b | `predictions_val_zeroshot.jsonl`, `predictions_test_zeroshot.jsonl`, `zeroshot_eval_report.json` |

Các artifact nằm ở `/kaggle/working` của từng notebook Kaggle (đã dùng làm Input nối tiếp nb0 → nb1 → nb2/nb3 → nb3b); repo hiện chỉ chứa mã notebook + tài liệu.
