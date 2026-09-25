# REPORT — Báo cáo tiến độ dự án: Sửa lỗi chính tả tiếng Việt (VSEC)

> **Mốc báo cáo**: 25/09/2026 · Ghi nhận toàn bộ công việc từ đầu dự án đến thời điểm hiện tại.
> Các lần chạy thí nghiệm trên Kaggle: ngày **22/09/2026** (nb0, nb1, nb2 — GPU T4), **22–23/09/2026** (nb3, nb3b — GPU T4), **24/09/2026** (nb4, nb4b — CPU, eval-only; nb5 — GPU T4, eval-only; nb4c — chạy local) và **25/09/2026** (nb6 — CPU, eval-only; nb7 — GPU T4).
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
| `notebooks/nb4-error-analysis.ipynb` | Error analysis stratified trên TEST + adjudication pseudo-gold (eval-only, CPU) | 2 | ✅ Hoàn thành | 24/09/2026 |
| `notebooks/nb4b-adjudication-llm-fill.ipynb` | Điền nhãn adjudication bằng LLM TypeSafe Choice (CPU) | 2 | ✅ Hoàn thành | 24/09/2026 |
| `notebooks/nb4c-adjudication-audit.ipynb` | Audit người 25 mẫu adjudication blind + consensus (chạy cục bộ) | 2 | ✅ Hoàn thành | 24/09/2026 (local) |
| `notebooks/nb5-external-eval.ipynb` | Đối sánh external NomVN: A — Run 1/Run 2 trên eval-real 150 câu; B — nomvn-base trên test 6k (eval-only) | 2 | ✅ Hoàn thành | 24/09/2026 |
| `notebooks/nb6-post-processing.ipynb` | Post-processing chống deletion FP — tune trên val Run 2, frozen lên test (eval-only, CPU) | 2 | ✅ Hoàn thành | 25/09/2026 |
| `notebooks/nb7-run3-train-eval.ipynb` | Run 3: noise v2 hiệu chỉnh + gate Telex + post-processing re-tune (GPU T4) | 2 | ✅ Hoàn thành | 25/09/2026 |

→ **Phase 0 + Phase 1 + Phase 2 hoàn thành đầy đủ** (Bước 1: error analysis + adjudication — mục 7; Bước 2: post-processing `nb6` + Run 3 `nb7` — mục 13). Cấu hình cuối của đồ án: **Run 2 + post-processing nb6 — Detection F1 test ≈ 81,38% · over-correction ≈ 1,46% · Clean Retention ≈ 96,5%**. Đối sánh external NomVN xem mục 12.

### 1.2. Kết quả quan trọng nhất (chi tiết ở các mục sau)

1. **Baseline đạt Detection F1 80,45% trên test** (Run 2, LoRA BARTpho + augmentation); correction accuracy tại vị trí detect đúng 67,94%.
2. **Augmentation (pha câu sạch + nhiễu tổng hợp) hiệu quả và rẻ**: Detection F1 test 76,43% → 80,45% (+4,0 điểm), recall +6,3 điểm, không làm over-correction xấu đi đáng kể.
3. **Over-correction KHÔNG phải vấn đề nghiêm trọng** trên bộ test này: 1,82–1,88% (baseline) < ngưỡng 2–3% của `DESIGN.md` §5; và con số này **có sẵn ở base model** (zero-shot 1,56%) → fine-tune không phải nguồn gốc chính.
4. **Non-word rate 22,7% < ngưỡng 40%** → theo quy tắc quyết định, **bỏ sớm hướng kiến trúc hybrid từ điển (C)**, giữ seq2seq làm trục chính.
5. Bộ test tự thu thập có **10,9% câu sạch** (653/5.983) → đo được Clean Retention; nhưng **dày lỗi hơn hẳn VSEC** (46,7% câu có ≥4 edit block so với 0,8% của VSEC) → domain shift đáng kể.
6. Quy trình Phase 0 **sạch về data leakage**: cả 3 invariant đều PASS; dedupe chéo VSEC↔test chỉ 0 trùng (thấp hơn dự kiến).
7. **Error analysis (nb4) định hướng thẳng Bước 2**: FP **deletion 45,4%** tổng FP Run 2 với pattern lặp (dấu câu `,`, họ âm tiết `tòa/họa/khỏe/thỏa…`); 77,2% vị trí detect-đúng-nhưng-sai là thay nhầm **từ thật**; 92% FN là **under-edit** (model sửa chỗ khác rồi bỏ sót) — không phải nhầm vị trí kề.
8. **Pseudo-gold của test sạch ở mức nhiễu thấp**: adjudication 100 mẫu suspect (LLM-assisted) cho nhiễu 0,0% (CI95 [0; 3,7%], n=100 — dưới ngưỡng 8%); **đã kiểm chứng**: audit người 25 mẫu blind đạt đồng thuận 25/25 (`nb4c`, 24/09).
9. **Đối sánh external (nb5)**: trên bench ngoài `eval-real` của NomVN, Run 1/Run 2 đạt word acc ≈55,6% — chỉ +3,6pp trên identity floor (51,96%) và cách xa nhóm dẫn đầu (nrl base 79,62%); điểm yếu rõ nhất là **lỗi Telex/VNI nhiều phím** (news_real 31%, telex_real 15%). Ngược chiều, `nomvn-base` trên test 6k của ta đạt **Detection F1 87,26% > 80,45%** (so sánh ở mức định hướng — quyết định 24/09 bỏ cross-dedupe, confound trùng corpus ghi nhận như hạn chế mục 10) → **mở gate rule Telex/VNI cho Run 3** (mục 12).
10. **Post-processing chống deletion (nb6) ĐẠT tiêu chí pre-registered**: config **G3-r1r2-r3k1** (punct guard + veto-delete âm tiết hợp lệ + whitelist k=1) tune trên val Run 2 (ΔF1 +4,69, over-corr 0,81% → 0,20%), frozen lên test: **ΔF1 test Run 2 +0,93** (~40% trần lý thuyết +2,33) · over-correction **giảm** 1,88% → ≈1,46% · Clean Retention 85,91% → ≈96,5% · Run 1 test +0,83 (robustness PASS) → **cấu hình cuối đồ án: Run 2 + post-processing, F1 test ≈ 81,38%**.
11. **Run 3 (noise v2 hiệu chỉnh + gate Telex, nb7) KHÔNG thắng Run 2 — kết quả âm có giá trị**: test F1 raw 78,68% (< 80,45%); sau post-processing re-tune riêng 79,78% (vẫn < 81,38% của Run 2 + PP). Calibration chỉ đạt **27,0%** non-word (không vào [23–25%]); **gate Telex PASS (14,6% ≥ 5%) nhưng telex_share chọn 0,0** vì thêm lỗi telex làm non-word rate tăng → lỗ hổng Telex giữ nguyên là giới hạn. → Hủy Run 4 synthetic-scaled; chi tiết mục 13.
12. **Quy trình kiểm chứng gold được tách thành giao thức tái sử dụng** (mục 14): chọn mẫu suspect stratified từ nhóm edit-dày → nhãn LLM có ràng buộc (Choice + confidence) → Wilson CI cận trên → audit người mù pre-registered (đồng thuận 25/25) — đóng góp phương pháp luận độc lập với F1, áp dụng được cho mọi test set GEC có nhãn suy diễn.

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
- Aligner đủ tin cậy để làm pseudo-annotation (~96% agreement với gold), tuy nhiên sai số ~4% tồn tại — xem mục 10 (hạn chế).
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

## 7. Phase 2 (Bước 1) — `nb4-error-analysis.ipynb` (+ `nb4b`): error analysis & adjudication pseudo-gold

### 7.1. Đã làm gì

- Notebook **eval-only, CPU** (không model, không train, không GPU): tái tạo **đúng semantics** `evaluate_predictions` của nb3 ở mức per-position events, đối chiếu **tuyệt đối** với `eval_report.json` — **CONSISTENCY CHECK PASS 4/4 set** (val/test × run1/run2) trước khi tin các bảng; sanity 7/7 case nhân tạo PASS; đối chiếu gold block tính tại chỗ vs `error_count` nb1: **0/5.983 lệch**.
- Phân tích 7 câu hỏi (Q1–Q7) từ **prediction có sẵn** của nb3: `{val,test} × {run1,run2}` (lần chạy này không attach Input zero-shot → không có cột nb3b).
- **Adjudication pseudo-gold (Q7)**: xuất 100 mẫu suspect stratified theo `edit_ratio` (seed 42; strata 88/10/2 trên pool suspect **2.316 câu = 38,7% test**) → `nb4b` điền nhãn A/B/C/D bằng LLM TypeSafe (primitive **Choice**, model `jev-latest`; pilot gate 10 mẫu; mỗi nhãn kèm `llm_confidence`) → upload làm Input → re-run nb4 tính nhiễu + Wilson CI.
- Xuất `error_analysis_report.json`, `adjudication_samples.json`.

### 7.2. Kết quả Q1–Q7 (TEST · Run 2 nếu không ghi chú)

| Q | Câu hỏi | Kết quả chính |
|---|---|---|
| Q1 | Sai ở non-word hay real-word? | CorrAcc@TP: non-word **67,57%** (10.591/15.673) · real-word **69,84%** (2.100/3.007) — chênh nhỏ; detection recall non-word 77,09% > real-word 66,02% |
| Q2 | Bỏ sót khi câu dày lỗi? | **Ngược giả thuyết ban đầu**: recall *tăng* theo mật độ — 1 lỗi/câu 70,87% < 2 lỗi 73,12% < 3 lỗi 75,10% < ≥4 lỗi 75,49%; câu thưa lỗi mới là điểm yếu |
| Q3 | FP deletion chiếm bao nhiêu? | **45,4%** (1.304/2.872 FP; Run 1: 54,2%) · merge 27,3% · multi 23,5% · substitute 3,6% · insert 0 |
| Q4 | TP sửa sai thành gì? | **77,2% là âm tiết hợp lệ** — thay nhầm từ thật (4.622/5.989) · deletion 11,6% · non-word tgt 10,7% · same_as_src 0,2% |
| Q5 | FN là copy thuần? | Chỉ **7,6%** pure-copy; **92,2%** FN ở câu model *có* sửa, cách edit gần nhất TB **6,5 token** → **under-editing**, không phải nhầm vị trí kề |
| Q6 | Bias vị trí/độ dài? | Yếu: recall 74,7–76,4% giữa các bucket vị trí; câu ngắn <15 token CorrAcc thấp nhất (59,6%) |
| Q7 | Pseudo-gold nhiễu bao nhiêu? | Xem 7.3 |

**Pattern bất ngờ từ T3/T7 (đầu vào trực tiếp cho Bước 2):**

- Token bị xóa oan **không ngẫu nhiên** — dấu câu `,` (×101) và **một họ âm tiết lặp lại**: `tòa` ×49, `họa` ×31, `khỏe` ×28, `hòa/thỏa/khóa/hủy/ủy/dọa/thủy` ×14–27 — chính họ token xuất hiện trong các ví dụ phá câu sạch (`Biếm họa → Biếm`, `thủy tinh → tinh`, `tòa án → án`) → khả năng lỗi hệ thống (tokenization/âm tiết hiếm dấu), chặn được bằng rule tất định.
- FP non-delete có **phrase "ám ảnh" lặp lại**: `'F Paris'` ×216, `'Nguwo'` ×95, `'Đ'` ×60 — cần soát xem có dồn ở vài câu nguồn hay không (chưa xử lý lần này).

### 7.3. Adjudication + audit người (Q7)

- Nhãn LLM (nb4b): **99A / 1C** → nhiễu pseudo-gold (B+D) = **0,0%**, Wilson 95% CI [0; 3,7%], n=100 **— đã audit người 25 mẫu blind, đồng thuận 25/25** — dưới ngưỡng pre-registered 8% → **không cần caveat** cho các chỉ số TEST theo quy tắc nb4 §6.
- **Hạn chế phải ghi rõ**: nhãn đến từ **một annotator LLM duy nhất**; gần như đồng thuận tuyệt đối trên nhóm *suspect* là kết quả "sạch bất thường" → chưa đủ để chốt trước khi có đồng thuận người.
- **Sửa diễn giải sensitivity (nb4, 24/09)**: dòng in cũ `F1 76,43% → 78,29% (loại 5.883 câu B/D)` gây hiểu nhầm — đó là metric tính lại trên **đúng 100 câu được adjudicate** (subset suspect edit-dày), **không phải** metric trên test sau khi loại câu nhiễu (wording đã sửa, không đổi số). Diễn giải đúng: 0/100 nhiễu → cận trên Wilson 3,7% → ngoại suy ≤ ~86/2.316 câu suspect (≈1,4% test) → ảnh hưởng cận trên đến chỉ số test ở mức **sub-1%**.
- **Audit người 25 mẫu blind (`nb4c`, pre-registered — hoàn tất 24/09)**: thiết kế gốc 2 low-confidence + toàn bộ stratum `>=0.5` + nhãn C + random seed-42; thực tế chạy phương án suy biến vì local chưa có `adjudication_filled.json` khi chọn mẫu — **25 mẫu = 23 random + toàn bộ stratum `>=0.5`** (2 mẫu edit-dày nhất, nhóm rủi ro non-parallel cao). Kết quả: **binary consensus 25/25 (100%) · full-label agreement 25/25** (nhầm lẫn chỉ có A→A) → quyết định pre-registered **accept: chốt nhiễu 0,0% (human-audited)** — không mở rộng 100 mẫu, không phát sinh `adjudication_filled_v2.json`. Hạn chế ghi nhận: audit là 1 annotator × 1 LLM (không phải đa annotator); mẫu nhãn C (1/100) nằm ngoài 25 mẫu nhưng C không tính là nhiễu theo rubric.

### 7.4. Phát hiện chính

1. **Val ≠ test về thành phần lỗi**: suy từ T1, gold positions trên test ≈ **81,7% non-word** (≈20.330/24.885) so với val ≈ 26% (khớp non-word rate 22,7% của VSEC ở Pilot 1) — giải thích phần lớn gap CorrAcc val→test (86,5% → 67,9%); lưu ý proxy non-word đếm cả từ ngoại/tên riêng tiếng Anh (xem mục 10).
2. **Deletion FP là lỗi hệ thống có pattern** — mục tiêu can thiệp rẻ nhất hiện tại: post-processing tất định, không cần train lại.
3. **Under-editing là nút cổ chai recall**: model chỉ sửa vài vị trí nổi bật/câu; recall thấp nhất ở câu 1 lỗi (70,9%).
4. **Run 2 (augmentation) không đánh đổi sang hại**: over-correction 1,82% → 1,88%; Clean Retention 86,68% → 85,91% (92/653 câu sạch bị phá, trong đó **106/135 FP positions là deletion** — nhất quán với pattern T3).

## 8. Tổng hợp phát hiện chính (cross-phase)

1. **Nền dữ liệu sạch, chống leakage đúng quy trình** (`DESIGN.md` §9): dedupe trước chia tập, split theo câu stratified seed 42, 3 invariant PASS, augmentation/bảng âm tiết chỉ dùng nguồn train hoặc nguồn công khai độc lập. Dedupe chéo VSEC↔test chỉ 0 trùng.
2. **Bộ test tự thu có 10,9% câu sạch nhưng dày lỗi hơn hẳn VSEC** (46,7% câu ≥4 block vs 0,8%): mọi số liệu trên test phải đọc kèm đặc điểm này; test khó hơn phân phối VSEC rõ rệt (correction acc tụt ~18 điểm val→test).
3. **Trục kiến trúc có lời giải sớm**: non-word rate 22,7% < 40% → bỏ hướng hybrid từ điển (C) làm trục chính; seq2seq giữ vai trò trung tâm, hướng pipeline bảo thủ (A) vẫn là ứng viên nếu xử lý FP deletion.
4. **Augmentation là lever rẻ và hiệu quả**: +2×2.502 câu (sạch + nhiễu) → F1 test +4,0 điểm, recall +6,3 điểm; đồng thời duy trì Clean Retention ~86%.
5. **Over-correction không phải vấn đề nghiêm trọng trên bộ test này** (1,8–1,9% < 2–3%) và có sẵn ở base model → hạ ưu tiên "chống over-correction", chuyển sang nâng recall/correction accuracy.
6. **Điểm yếu còn mở**: correction accuracy trên test còn thấp (66–68%); FP deletion của model (nb4 định lượng: 45,4% tổng FP Run 2, có pattern cụ thể — xem mục 7); noise model chưa hiệu chuẩn (non-word sinh 51,6% vs thật 23,4% — lệch 28,2 điểm).

## 9. Trạng thái lộ trình & bước tiếp theo đề xuất (Phase 2)

Theo lộ trình `DESIGN.md` §4 và bảng quy tắc quyết định §5:

| Kết quả đo được | Ngưỡng | Quyết định đề xuất |
|---|---|---|
| Over-correction baseline 1,82–1,88% (test) | < 2–3% → "không nghiêm trọng" | Không dồn lực chống over-correction; ưu tiên phân tích stratified sâu |
| Non-word rate 22,7% | < 40% → "bỏ C sớm" | Bỏ hybrid từ điển (C) làm trục chính; (tùy chọn) giữ làm ablation nhỏ |
| Augmentation F1 test +4,0 điểm | — | Giữ augmentation làm thành phần mặc định; hiệu chỉnh noise model (lệch 28,2 điểm) trước khi scale tỷ lệ pha |
| FP pattern deletion = 45,4% tổng FP Run 2, có pattern lặp (nb4) | Trần lý thuyết +2,33 F1 nếu chặn hết delete-FP (precision 86,7% → 92,3%) | **Bước 2 — post-processing**: punctuation guard · veto delete trên âm tiết hợp lệ · blacklist phrase từ val; derive/tune trên val, áp frozen lên test |

Kế hoạch Bước 2 (chốt 24/09, nhật ký `DESIGN.md` §12): (1) post-processing chống deletion như bảng trên — rule tất định gold-agnostic, chống leakage đúng `DESIGN.md` §9; (2) **Run 3** noise hiệu chỉnh giữ mục tiêu non-word **23–25%** (không tối theo thành phần ~82% non-word đo được trên test — tránh test-peeking, chỉ report stratified); (3) tùy chọn eval-only: **multi-pass inference** cho under-editing (92% FN — nb4 §7.2); (4) full-FT ablation theo quota.

Việc còn mở (không bắt buộc, theo `DESIGN.md`): full-FT ablation 1 lần cuối; factorial 2×2 hai trục nếu kịp thời gian.

**Cập nhật 25/09 — Bước 2 đã thực thi xong (`nb6` + `nb7`, chi tiết mục 13)**: post-processing đạt ngưỡng pre-registered và được nhận làm thành phần mặc định; Run 3 (noise v2) không thắng Run 2 → hủy Run 4. Còn mở (tùy chọn): multi-pass inference cho under-editing; full-FT ablation theo quota. Hướng mở mới đề xuất từ brainstorm — Telex/VNI input normalization · MLM candidate ranking (**chưa thực thi, chờ phê duyệt**): xem mục 15.

## 10. Hạn chế của đo lường (đọc kết quả cần lưu ý)

- **Metric trên test dựa trên pseudo-annotation** do aligner suy ra (đã kiểm chứng ~96% agreement với gold VSEC, roundtrip 100%) — nhưng không phải gold thật; sai số ~4% của aligner nằm trong cả gold lẫn prediction trên test.
- **Clean Retention trên val chỉ có 3 câu sạch** (VSEC 100% câu có lỗi) — không có ý nghĩa thống kê; con số đáng tin là trên test (653 câu). Số tuyệt đối val/test trong bảng 5.2 được suy từ tỷ lệ × số câu sạch.
- **Stratified gold nhỏ**: 296 non-word / 847 real-word trên val — chênh lệch vài điểm % giữa Run 1/Run 2 chưa đủ kết luận thống kê.
- **Số liệu là snapshot** các lần chạy Kaggle 22–23/09/2026 (seed 42, greedy decode); re-run có thể lệch nhỏ do phi tất định phần cứng.
- Bộ test 6k chỉ có `text` + `label` tự thu thập; chất lượng bản sửa của test là pseudo-gold: đã adjudicate 100 mẫu suspect bằng LLM (nhiễu 0,0%, CI95 [0; 3,7%]) và **audit người 25 mẫu blind đã hoàn tất với đồng thuận 25/25** (`nb4c`, xem mục 7.3) — lưu ý vẫn là 1 annotator × 1 LLM, không phải đa annotator.
- **Sensitivity của nb4 không phải exclusion trên test**: số "F1 76,43% → 78,29%" là metric tính lại trên đúng 100 câu được adjudicate (subset suspect edit-dày), không phải trên test sau khi loại câu nhiễu (wording đã sửa 24/09 — xem mục 7.3).
- **Proxy non-word là phép đo gián tiếp**: phân loại theo bảng âm tiết 7.884 nên từ ngoại/tên riêng tiếng Anh (`hacktivisme`, `Preferences`…) cũng bị đếm là non-word — thành phần ~82% non-word trên test cần đọc với hạn chế này (val ~26%, khớp Pilot 1 22,7%).
- **Kết quả external nb5 (mục 12) đọc kèm 3 hạn chế**: (1) bench `eval-real` chỉ n=150 (CI bootstrap rộng ±4–6pp) và word acc là **định nghĩa tự có của nb5** — chỉ so hướng (ordinal) với anchor công bố của họ; (2) phần A là **re-run inference** của adapter nb3 (greedy) — có thể lệch nhỏ so với run gốc do hardware nondeterminism; (3) Hướng B còn **confound đã ghi nhận, không loại trừ**: khả năng một phần câu test trùng (dạng sạch) với corpus Wiki+news 600K của họ — quyết định 24/09 **bỏ cross-dedupe** (tập trung pipeline nội bộ): kết luận so sánh chỉ ở mức định hướng, không dùng để tuyên bố tuyệt đối.
- **Post-processing nb6 tạo artifact khoảng trắng trước dấu câu** trong file `predictions_*_postproc.jsonl`: khi một block bị veto, câu được rebuild bằng join khoảng trắng → dấu câu tách khỏi từ (vd `Sê San.` → `Sê San .`). Metric không ảnh hưởng (canon_tokenize tách dấu câu thành token riêng sẵn) nhưng output cần detokenize lại nếu dùng làm văn bản trình bày.
- **ΔRecall âm nhẹ của post-processing** (test Run 2 −0,32 · val −2,01): R2 thỉnh thoảng hoàn tác cả khối xóa là gold (gold có lỗi thật dạng xóa) — đánh đổi được chấp nhận vì precision +2,63 (test) mà over-correction giảm.
- **Kết quả Run 3 (nb7) đọc kèm 3 hạn chế**: (1) calibration noise v2 dừng ở **27,0%** non-word — không vào vùng mục tiêu [23–25%] dù quét đủ 18 tổ hợp; (2) **gate Telex PASS (14,6%) nhưng telex_share = 0,0** — thêm lỗi telex làm non-word rate tăng, xung đột mục tiêu hiệu chuẩn → rule Telex thực chất không tham gia augmentation, lỗ hổng Telex của nb5 chưa được xử lý; (3) Run 3 tốt hơn Run 2 trên val (80,42% vs 79,35% sau PP) nhưng kém hơn trên test (79,78% vs 81,38%) — quyết định dựa trên test (benchmark chính, n=5.983); val n=927 quá nhỏ để đảo ngược.

## 11. Danh mục artifact đã sinh (output Kaggle)

| Notebook | File output |
|---|---|
| nb0 | `vsec_train.jsonl`, `vsec_val.jsonl`, `test_normalized.jsonl`, `manifest.json` |
| nb1 | `test_aligned.jsonl`, `qa_samples.json`, `align_report.json` |
| nb2 | `syllable_table.json`, `noise_model.json`, `noise_qa_samples.json`, `pilot_report.json` |
| nb3 | `predictions_val_run{1,2}.jsonl`, `predictions_test_run{1,2}.jsonl`, `eval_report.json`, `lora_adapter_run{1,2}/` |
| nb3b | `predictions_val_zeroshot.jsonl`, `predictions_test_zeroshot.jsonl`, `zeroshot_eval_report.json` |
| nb4 | `error_analysis_report.json`, `adjudication_samples.json` |
| nb4b | `adjudication_filled.json`, `review_queue.json` |
| nb5 | `predictions_evalreal_run{1,2}.jsonl`, `predictions_test_nomvnbase.jsonl`, `external_bench_report.json` |
| nb6 | `postprocess_config.json`, `postprocess_report.json`, `predictions_{val,test}_run{1,2}_postproc.jsonl` |
| nb7 | `predictions_{val,test}_run3.jsonl`, `eval_report_run3.json`, `noise_model_v2.json`, `postprocess_run3_report.json`, `lora_adapter_run3/` |

Artifact audit cục bộ (nb4c, chạy tại repo): `data/audit_samples.json` (25 mẫu blind) → `data/audit_filled.json` (nhãn người, 25/25 — đồng thuận với nhãn LLM) → `adjudication_filled_v2.json` không phát sinh (không có lệch nhãn).

Các artifact nằm ở `/kaggle/working` của từng notebook Kaggle (đã dùng làm Input nối tiếp nb0 → nb1 → nb2/nb3 → nb3b → nb4/nb4b → nb5); repo hiện chỉ chứa mã notebook + tài liệu + artifact audit cục bộ.

## 12. Đối sánh external với NomVN (`nb5-external-eval.ipynb`, eval-only)

> Chạy 24/09/2026 trên Kaggle (T4, fp16; hạ tầng phải downgrade `transformers<5` — bug v5 với tokenizer sentencepiece, đúng khuyến nghị cài đặt của NomVN). Đối chiếu hai hướng: **A** — model của ta trên benchmark ngoài của họ; **B** — model công khai của họ trên test 6k của ta. Pre-registered: eval-only tuyệt đối, không tune bất cứ gì theo bench.

### 12.1. Thiết lập

- **Hướng A**: LoRA adapter Run 1 + Run 2 (nb3) trên `nrl-ai/vn-spell-correction-eval-real` (**CC0**, 150 cặp hand-curated, 6 register × 25 câu: forum / mobile / news_real / legal_real / ocr / telex_real) → word acc + sentence exact (bootstrap CI 95%, seed 42, n=1000 resample) + bộ 3 chỉ số qua pseudo-annotation `align-v1`. Tính toàn vẹn: roundtrip PASS 150/150, `align_failed=0`, 10 câu sạch, identity floor word acc **51,96%** (bench dày lỗi nặng).
- **Hướng B**: `nrl-ai/vn-spell-correction-base` (Apache-2.0, ViT5-base 220M, revision `ba1e57f`) trên test 6k — cùng gold `align-v1` với nb3/nb4; truncate 12/5.983 câu (0,2%), empty preds 0%.
- Word acc là **định nghĩa tự có của nb5** (align token canonical; equal / clean-tokens) — so anchor công bố chỉ theo hướng (ordinal).

### 12.2. Hướng A — model của ta trên eval-real

| | Word acc [CI95] |
|---|---|
| Identity floor | 51,96% [46,96–57,26] |
| **Run 1** | **55,65%** [50,20–61,39] |
| **Run 2** | **55,54%** [50,29–60,93] |

Anchor công bố của NomVN (WA của họ): nrl base 79,62 · small 77,55 · Toshiiiii1 77,40 · qthuan 72,42 · chamdent 51,69 · bmd1905 49,21 · iAmHieu 45,57 → Run 1/Run 2 (≈55,6%) hơn hẳn nhóm dưới (45–49%) nhưng cách xa nhóm trên (72–80%) và **chỉ +3,6pp trên identity floor**.

Bộ 3 chỉ số trên eval-real (Run 2): Detection P 96,04% · **R 19,64%** · F1 32,61% · CorrAcc@TP 24,31% · Over-correction 0,92% → **under-edit nặng ngoài phân phối** (bảo thủ tới mức bỏ sót ~4/5 vị trí lỗi).

Per-register (WA Run1/Run2 · F1 Run1/Run2): forum 78,4/79,3% · 42,2/52,1% — mobile 97,3/96,8% · 66,7/64,0% — legal_real 59,6/56,6% · 39,1/32,8% — ocr 53,4/53,5% · 33,5/29,2% — **news_real 30,8/31,1% · 17,5/18,7%** — **telex_real 14,4/15,9% · 32,4/39,4%**.

**Giả thuyết pre-registered "news/legal cạnh tranh" bị PHẢN BÁC**: news_real tệ nhất (WA 31%) dù trùng domain báo chí — slice này dày **lỗi Telex/VNI nhiều phím** (`đuwowjc`, `cafng`, `trfọ`, `chiêu đã9i`) mà noise model/VSEC không covering. **Domain khớp ≠ kiểu lỗi khớp.** Điểm yếu Telex nay được xác nhận trên cả bench ngoài lẫn bench trong (QA mẫu Hướng B bên dưới).

### 12.3. Hướng B — nomvn-base trên test 6k của ta

| Chỉ số (test 6k) | Run 2 (nb3) | nomvn-base |
|---|---|---|
| Detection Precision | 86,67% | 88,62% |
| Detection Recall | 75,07% | **85,94%** |
| Detection F1 | 80,45% | **87,26%** |
| Correction Acc @TP | 67,94% | 67,58% |
| Over-correction Rate | 1,88% | 1,80% |
| Clean Retention | 85,91% | **95,10%** |
| Word acc | 93,14% | **94,80%** |

→ "545K synthetic + full-FT" thắng "8.3K lỗi thật + augmentation nhỏ" trên sân nhà ta, chủ yếu nhờ **recall (+10,9 điểm)** và Clean Retention (+9,2 điểm); CorrAcc@TP và over-correction ngang bằng.

**Confound đã ghi nhận — quyết định KHÔNG cross-dedupe (24/09)**: corpus train của họ = Wiki+news+legal 600K; test 6k của ta nguồn web/báo → khả năng một phần câu test nằm (dạng sạch) trong corpus của họ là có thật (guard anti-leak của họ chỉ áp cho eval set của chính họ). Quyết định của người dùng: **bỏ cross-dedupe, tập trung pipeline nội bộ** — so sánh hai hướng giữ ở **mức định hướng**, không dùng để tuyên bố tuyệt đối; confound này là hạn chế đã công bố (mục 10). QA mẫu cho thấy model họ cũng không sửa được Telex-heavy trên test (`vowj`, `đuwowj`, `hoom` giữ nguyên) — nhất quán với 12.2 và với quan sát của chính họ ("Telex là điểm yếu chung").

### 12.4. Hệ quả quyết định (nhật ký `DESIGN.md` §12)

1. **Mở gate rule Telex/VNI cho Run 3** (P2 → thành phần của Run 3): bằng chứng hai chiều từ bench ngoài + bench trong, không phải test-peeking; khi implement vẫn kiểm thống kê `correction_pairs` train.
2. Vị thế đồ án không đổi: đóng góp = **phương pháp đánh giá** (bộ 3 chỉ số position-level, aligner kiểm chứng, adjudication + human audit — giao thức đầy đủ xem mục 14) + phân tích lỗi thật — không đua scale dữ liệu.
3. **Run 4 (synthetic-scaled) vẫn HOÃN** — cân nhắc sau Run 3 theo quota; adapter Telex từ `nom.text.noise` giữ ở P2 (license repo `nrl-ai/nom-vn` chưa xác nhận — có thể tự cài rule thay thế).
4. **Bỏ cross-dedupe test 6k ↔ corpus train của họ** (quyết định 24/09 — tập trung pipeline nội bộ): so sánh external giữ ở mức định hướng; confound trùng corpus chuyển thành hạn chế đã công bố (mục 10), không còn là việc đang chờ.

## 13. Phase 2 (Bước 2) — `nb6-post-processing.ipynb` + `nb7-run3-train-eval.ipynb`

> Chạy 25/09/2026 trên Kaggle (nb6: CPU, eval-only; nb7: T4, fp16 — 1 run train duy nhất trong plan). Thực thi kế hoạch Bước 2 chốt 24/09 (`DESIGN.md` §12), tiêu chí đều pre-registered: post-processing chỉ nhận nếu gain ≥ 0,3 F1 val + over-correction không tăng; Run 3 chốt mục tiêu non-word [23–25%] trước khi nhìn test. Chống leakage `DESIGN.md` §9: rule/resources derive chỉ từ val; findings nb4 trên test chỉ dùng đối chiếu chiều hướng sau khi freeze.

### 13.1. nb6 — Post-processing chống deletion: ĐẠT tiêu chí pre-registered

**Đã làm gì**

- Rule engine R1–R4 ở cấp độ edit block (cùng hệ `align-v1`): **R1** punct guard (block punct-only) · **R2** veto-delete âm tiết hợp lệ (mọi token bị xóa ∈ bảng 7.884) · **R3** whitelist token bị xóa oan tần suất ≥ k (derive từ val) · **R4** phrase blacklist ≥2 token lặp (derive từ val). Block bị veto → hoàn nguyên tgt := src rồi rebuild; không block nào bị veto → prediction gốc nguyên vẹn (identity guarantee). Sanity rule engine nhân tạo PASS; NFC + roundtrip token hóa PASS 4/4 set.
- Mining FP **chỉ trên val Run 2**: 145 FP blocks (delete 105 · substitute 32 · merge 5) · 784 block mixed không hoàn nguyên được · top token bị xóa oan `hòa` ×10, `thủy` ×9, `khỏe` ×8, `và` ×8, `,` ×8 — nhất quán chiều hướng nb4 trên test (45,4% delete), chỉ dùng đối chiếu.
- Sweep grid 6 tổ hợp lồng nhau trên val Run 2 (n=927); Run 1 test = robustness check.

**Sweep trên val Run 2** (baseline F1 74,67% · over-corr 0,81%):

| Config | ΔF1 val | F1 | Over-corr | Ghi chú |
|---|---|---|---|---|
| G1-r1 | +0,67 | 75,34% | 0,73% | R1 đơn lẻ yếu |
| G2-r1r2 | +4,13 | 78,80% | 0,25% | |
| **G3-r1r2-r3k1 (CHỌN)** | **+4,69** | **79,35%** | **0,20%** | k=1 thắng k=3/5 |
| G4 / G5 (k=3/5), G6 (k=5 +R4) | +4,13 | 78,80% | 0,25% | R4 / R3 k≥3 không thêm |

**Frozen áp 4 set** (Δ so với raw):

| Set | ΔPrec | ΔRec | ΔF1 | ΔOver-corr | ΔCleanRet | Hit rule |
|---|---|---|---|---|---|---|
| val Run 1 | +13,61 | −2,27 | +4,12 | −0,56 | +33,33 | R2:114 · R1:18 · R3:3 |
| val Run 2 | +14,79 | −2,01 | +4,69 | −0,61 | +33,33 | R2:117 · R1:22 · R3:5 |
| test Run 1 (robustness) | +2,72 | −0,35 | **+0,83** | −0,41 | +10,11 | R2:550 · R1:58 · R3:1 |
| test Run 2 (apply) | +2,63 | −0,32 | **+0,93** | −0,42 | +10,57 | R2:549 · R1:81 · R3:1 |

**Phát hiện**

1. Đạt ~40% trần lý thuyết +2,33 (rule chỉ chặn được delete-FP có pattern) · over-correction test **giảm** (1,88% → ≈1,46%) — đúng ràng buộc "không tăng".
2. **Cấu hình cuối đồ án (test Run 2 + post-processing): Detection F1 ≈ 81,38% · over-correction ≈ 1,46% · Clean Retention ≈ 96,5%** (từ 80,45 / 1,88 / 85,91).
3. R2 gánh gần như toàn bộ (549/631 hit trên test Run 2); R3 whitelist từ val chỉ bắn 1 hit trên test → pattern FP test ít chồng whitelist val. ΔRecall âm nhẹ do R2 hoàn tác nhầm cả khối delete là gold — đánh đổi chấp nhận được (precision +2,63).

### 13.2. nb7 — Run 3 (noise v2 hiệu chỉnh + gate Telex): KHÔNG thắng Run 2 (kết quả âm có giá trị)

**Đã làm gì**

- Đo lại real non-word rate train: **23,4%** (2.344/10.032 pair 1-token) — khớp nb2.
- **Gate Telex PASS**: 1.395/9.533 pair train telex-like = **14,6%** ≥ ngưỡng 5% → bật rule `telex_grammar` (bảng chuyển tự telex/VNI tất định; sanity `được → dduwowjc / d9u7o75c` PASS).
- **Calibration 18 tổ hợp** ({P_EMPIRICAL 0,5/0,65/0,8} × {keyboard_strict} × {telex_share 0/0,05/0,10}) trên 2.000 câu seeded: **không tổ hợp nào vào [23%, 25%]**. `keyboard_strict` là đòn bẩy chính (51,6% → ~27–30%); **thêm telex làm non-word tăng** (27,0 → 31,6/34,9% cùng cột) → chọn gần nhất `p_emp=0,8 · keyboard_strict=True · telex_share=0,0` → **27,0%** (không ép cực đoan, log đầy đủ — đúng giao thức pre-registered).
- Train Run 3 = công thức Run 2 giữ nguyên (LoRA 0,98% trainable · 6 epochs · seed 42) + 2.502 sạch + 2.502 nhiễu v2 = 13.347 câu; assert split=train · NFC · invariant vị trí PASS.
- Post-processing **re-tune cùng rule-shape nb6 trên val Run 3** (mining lại whitelist/blacklist — không tái dùng Run 2): chọn lại **G3-r1r2-r3k1**, gain val **+6,17** (F1 74,25 → 80,42 · over-corr 0,98 → 0,23); frozen lên test Run 3: ΔF1 +1,09 · Δover-corr −0,47 · ΔCleanRet +9,80.

**Kết quả test (cùng pipeline so trực diện)**

| Test | Prec | Rec | F1 | CorrAcc | Over-corr | CleanRet |
|---|---|---|---|---|---|---|
| Run 2 raw | 86,67% | 75,07% | **80,45%** | 67,94% | 1,88% | 85,91% |
| Run 3 raw | 85,68% | 72,75% | 78,68% | 67,38% | 1,98% | 86,52% |
| Run 3 + PP (re-tune val Run 3) | 88,68% | 72,50% | 79,78% | — | 1,51% | ≈96,3% |
| **Run 2 + PP (nb6)** | ≈89,30% | ≈74,75% | **≈81,38%** | — | ≈1,46% | ≈96,5% |

Stratified Run 3 (test, proxy bảng âm tiết): non-word gold 20.330 — recall 74,35% · corrected/gold 49,87% · real-word gold 4.555 — recall 65,60% · corrected/gold 45,23%.

**Phát hiện**

1. **Noise v2 theo thiết kế này không cải thiện Run 2**: test F1 raw −1,77 điểm; kể cả sau post-processing riêng vẫn −1,60 điểm so Run 2 + PP; over-correction raw tăng nhẹ (1,88 → 1,98%).
2. **Tín hiệu trái chiều val/test**: trên val Run 3 + PP cao hơn Run 2 + PP (80,42% vs 79,35%) nhưng trên test thấp hơn (79,78% vs 81,38%) — nhất quán pattern val≠test của augmentation (mục 5.3); quyết định dựa trên test (benchmark chính, n=5.983).
3. **Gate Telex PASS nhưng không dùng được**: share telex xung đột trực tiếp với mục tiêu hiệu chuẩn non-word → telex_share = 0,0. Lỗ hổng Telex/VNI của nb5 **chưa được xử lý** — chuyển thành giới hạn đã công bố (mục 10); không tối riêng cho slice telex (chống test-peeking).
4. Calibration dừng ở 27,0%: phần dư so với 23,4% thật đến từ chính confusion thực nghiệm (nặng dạng non-word) + rule tone/vowel/regional cũng sinh non-word; giảm sâu hơn cần nguồn lỗi real-word-biased — ngoài phạm vi quota còn lại.

### 13.3. Hệ quả quyết định (đề xuất nhật ký `DESIGN.md` §12, ngày 25/09)

1. **Nhận post-processing nb6 (G3-r1r2-r3k1) làm thành phần mặc định** của pipeline. **Cấu hình cuối đồ án: Run 2 + post-processing — Detection F1 test ≈ 81,38% · over-correction ≈ 1,46% · Clean Retention ≈ 96,5%**.
2. **Run 3 không thắng → hủy Run 4 (synthetic-scaled)**: noise v2 gần đạt calibration vẫn thua Run 2 → scaling cùng nguồn noise nhiều khả năng không có lợi.
3. Lỗ hổng Telex/VNI giữ là **giới hạn đã công bố** (mục 10), không dồn thêm nỗ lực augmentation.
4. Còn mở (tùy chọn theo quota): multi-pass inference cho under-editing (92% FN — mục 7.2); full-FT ablation 1 lần.

## 14. Đóng góp phương pháp luận — Giao thức kiểm chứng nhãn test: LLM-assisted adjudication + human audit

> Tách từ cụm `nb4`/`nb4b`/`nb4c` (mục 7) thành một giao thức độc lập, tái sử dụng được — cho mọi bộ test chỉnh sửa văn bản (GEC/denoising/OCR post-correction) có nhãn suy diễn (pseudo-gold) và không có gold thật để đối chiếu. Đây là đóng góp phương pháp luận của đồ án, song hành với bộ 3 chỉ số position-level và aligner kiểm chứng.

### 14.1. Vấn đề

Bộ test tự thu (6k câu) chỉ có `text` + bản sửa tự tạo → nhãn vị trí lỗi là pseudo-gold do aligner sinh ra (đã kiểm chứng ~96% agreement với gold VSEC — mục 3, nhưng sai số ~4% nằm trong cả gold lẫn prediction). Toàn bộ chỉ số trên test phụ thuộc chất lượng pseudo-gold này, trong khi soát tay toàn bộ 5.983 câu là bất khả thi. Câu hỏi của giao thức: **định lượng "độ tin cậy của gold" với ngân sách annotator tối thiểu nhưng công bố được con số có cơ sở thống kê**.

### 14.2. Giao thức 4 bước (như đã thực thi trong dự án)

1. **Chọn mẫu suspect có hệ thống** (`nb4`): pool "câu nghi ngờ" = edit dày (edit_ratio ≥ 30% — nhóm rủi ro phi song song cao nhất: 2.316 câu = 38,7% test); lấy mẫu stratified theo `edit_ratio`, seed cố định, ghi rõ strata (100 mẫu = 88/10/2) — **không random toàn tập**, because nhóm rủi ro cao phải được sample đủ.
2. **Điền nhãn bằng LLM có ràng buộc** (`nb4b`): primitive **Choice** đóng khung nhãn (A song song-đúng / B sai-LM-đúng / C sai-khác / D gold-đúng-LM-sai), pilot gate 10 mẫu trước khi chạy full, mỗi nhãn kèm `llm_confidence` để stratify bước audit.
3. **Định lượng nhiễu + cận trên thống kê** (`nb4`): định nghĩa nhiễu trước (nhãn B+D); báo cáo **Wilson 95% CI** — kết quả: 0,0% [0; 3,7%], n=100, dưới ngưỡng pre-registered 8%; ngoại suy cận trên lên toàn pool (≤ ~86/2.316 ≈ 1,4% test) thay vì dùng điểm số. Cảnh báo diễn giải đã ghi nhận (mục 7.3): metric tính lại trên đúng 100 câu adjudicate **không được** đọc thành "metric test sau làm sạch".
4. **Audit người mù danh tính, pre-registered** (`nb4c`): quy tắc chọn mẫu chốt **trước khi** nhìn nhãn (toàn bộ stratum `≥0.5` — 2 mẫu edit-dày nhất + random seed 42); annotator không biết nhãn LLM; ngưỡng chốt: binary consensus đạt → accept, lệch → mở rộng `v2`. Kết quả thực thi: đồng thuận 25/25 → chốt nhiễu 0,0% (human-audited), không phát sinh hiệu chỉnh nhãn.

### 14.3. Vì sao giao thức tổng quát

- Hạn chế "1 annotator × 1 LLM" (mục 10) là giới hạn **thực thi của lần chạy này**; cấu trúc giao thức không phụ thuộc số annotator — nâng cấp đa annotator (majority vote) là tăng n, không đổi quy trình.
- Khác biệt với "LLM-as-judge" thông thường, nằm ở 4 ràng buộc: (i) mẫu lấy **có hệ thống từ nhóm rủi ro cao nhất**, không random toàn tập; (ii) **pre-registration** ngưỡng nhiễu và quy tắc accept/expand trước khi nhìn nhãn; (iii) báo cáo **CI cận trên**, không dùng điểm số làm metric; (iv) **audit người mù** bắt buộc trước khi chốt — nhãn LLM chưa bao giờ được tính là gold.
- Miền áp dụng: dataset GEC tự crawl, pseudo-label distillation, QA dữ liệu synthetic — bất kỳ đâu nhãn đến từ aligner/pipeline tự động và chi phí soát tay toàn bộ là bất khả thi.

### 14.4. Vị trí trong tổng thể đóng góp đồ án

Hạ tầng đánh giá tái sử dụng gồm 3 thành phần: (a) bộ 3 chỉ số position-level — detection / correction / over-correction (`PROJECT.md` §6); (b) aligner block-based kiểm chứng trên gold (mục 3: count agreement 96,8% · recall 96,0% · roundtrip 100%); (c) giao thức kiểm chứng gold của mục này. Nhất quán với định vị đã chốt (`DESIGN.md` §12, 24/09): đóng góp = phương pháp đánh giá + phân tích lỗi thật, không đua scale dữ liệu.

## 15. Hướng mở & kế hoạch tiếp theo (đề xuất 25/09 — CHƯA thực thi, chờ phê duyệt phạm vi)

> Sản phẩm của brainstorm có cấu trúc (12 ứng viên qua 2 bộ khung ideation → filter Explain-it / Problem-first / Simplicity / Feasibility → 3 top). Ứng viên #6 (giao thức kiểm chứng gold) đã tách thành mục 14. Hai hướng dưới đây đã qua two-sentence test và filter nhưng **chưa chạy** — mọi con số nêu ra là nền đo được từ các mục trước, không phải kết quả mới. Thực thi cần phê duyệt theo `PROJECT.md` §0.

### 15.1. Hướng A — Telex/VNI xử lý ở input normalization (ưu tiên 1 · eval-only)

- **Động cơ**: WA telex_real chỉ 14,4–15,9% trên eval-real (mục 12.2), `nomvn-base` cũng không sửa được Telex-heavy trên test (mục 12.3); nb7 chứng minh đưa telex vào augmentation **xung đột mục tiêu hiệu chuẩn** (telex_share bị ép về 0 — mục 13.2) → giải còn lại của không gian là sửa **input trước model**.
- **Cơ chế**: detector token ASCII-heavy trong ngữ cảnh có dấu (≥k chữ ASCII ∧ ≥m token có dấu trong cửa sổ) + bảng chuyển tự tất định nb7 (`transliterate_style`, sanity `được → dduwowjc / d9u7o75c` PASS); expansion chỉ nhận nếu **∈ bảng âm tiết**, ambiguous → bỏ qua vị trí (bảo thủ — cùng triết lý "không sửa oan" của nb6).
- **Thí nghiệm**: E1 dựng normalizer · E2 A/B frozen (adapter Run 2 + PP, có/không normalizer) trên eval-real 150 per-register + slice telex của test 6k (xác định bằng chính detector — gold-agnostic) · E3 guard ablation: normalizer trên câu không-telex phải ~no-op (đo FP rate phía sạch).
- **Ngưỡng pre-registered**: nhận nếu WA telex_real cải thiện có nghĩa **∧** over-correction toàn test không tăng; không đạt → ghi negative result vào REPORT.
- **Chống leakage**: rule từ bảng chuyển tự công khai + bảng âm tiết nguồn độc lập; không derive từ test.
- **Chi phí**: eval-only, 1 phiên Kaggle ngắn (pilot chạy slice telex trước).

### 15.2. Hướng B — Correction bằng selection: MLM candidate ranking (ưu tiên 2 · bắt đầu từ oracle)

- **Động cơ**: CorrAcc@TP test 67,94% là chỉ số thấp nhất; 77,2% TP-sai là thay nhầm **từ thật** (nb4 Q4) → model detect tốt nhưng sinh nội dung kém; selection ≤ generation về độ khó.
- **Cơ chế**: giữ seq2seq làm detector; tại vị trí đã detect, dựng candidate set (MLM top-k + confusion table train của nb2 + identity) và rank bằng pseudo-log-likelihood ngữ cảnh của LM công khai.
- **Thí nghiệm rẻ-first**: E0 **oracle test** trên prediction có sẵn (thuần CPU): CorrAcc oracle@k tại vị trí detect đúng — nếu oracle@10 không vượt 67,94% đủ xa thì dừng, không xây ranker · E1 training-free rerank (MLM pre-trained) · E2 (tùy E1) train ranker nhỏ chỉ trên train/val.
- **Ngưỡng pre-registered**: E0 phải cho oracle@k ≥ +10 điểm so với 67,94% mới mở E1; E1 nhận nếu CorrAcc@TP tăng ∧ over-correction không tăng.
- **Chống leakage**: MLM nguồn công khai độc lập; ranker (nếu train) fit train/val; confusion từ train (nb2).
- **Chi phí**: E0 gần như miễn phí; E1 cần 1 lần inference MLM.

### 15.3. Thứ tự đề xuất

1. Pilot Hướng A (slice telex trước — nếu WA telex_real đi từ ~15% về hướng forum-level ~78% mới chạy full).
2. E0 Hướng B (oracle, CPU).
3. Chờ quota: sparse-error augmentation ("1 lỗi / ngữ cảnh dài" — ứng viên #3 của brainstorm, đánh trực tiếp under-editing Q2/Q5, cần 1 run T4; sẽ thiết kế riêng nếu tới).
Tất cả các bước trên **chưa được phê duyệt thực thi** — đây là danh sách kế hoạch, không phải trạng thái dự án.
