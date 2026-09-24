# Dự án: Sửa lỗi chính tả tiếng Việt (NLP Course Project)

> File này dành cho coding agent đọc để hiểu bối cảnh, mục tiêu, ràng buộc và quy trình làm việc của dự án trước khi viết bất kỳ dòng code nào.

## 0. QUY TẮC LÀM VIỆC BẮT BUỘC (đọc trước tiên)

- **KHÔNG được tự ý sửa code, chỉnh sửa file, hoặc thay đổi bất cứ thứ gì trong repo mà chưa được xác nhận.** Trước khi triển khai (viết code mới, sửa file có sẵn, cài package, đổi cấu trúc thư mục...), phải trình bày kế hoạch/diff dự kiến và **chờ người dùng xác nhận rõ ràng**, sau đó mới thực hiện.
- Khi cần làm rõ yêu cầu, hỏi lại thay vì tự suy đoán rồi triển khai luôn.
- Ưu tiên đề xuất từng bước nhỏ, dễ review, hơn là làm một lúc nhiều thay đổi lớn.

## 1. Bối cảnh

- Đây là đồ án cho **môn học NLP** (mục tiêu học thuật: hiểu và chứng minh khả năng làm việc với bài toán NLP thực tế — xử lý dữ liệu, mô hình hóa, đánh giá — không phải xây sản phẩm thương mại).
- Phạm vi hiện tại: **chỉ tập trung sửa lỗi chính tả (spelling correction) tiếng Việt**, ở mức âm tiết (syllable). Chưa xử lý lỗi ngữ pháp, dấu câu, hay phong cách.

## 2. Mục tiêu

Xây dựng và **đánh giá** một hệ thống phát hiện + sửa lỗi chính tả tiếng Việt trên dataset VSEC, với trọng tâm:

1. Baseline: fine-tune một mô hình seq2seq (ví dụ BARTpho-syllable) trên VSEC để sửa lỗi chính tả.
2. So sánh với cách tiếp cận dùng LLM có sẵn (few-shot prompting) để trả lời câu hỏi: khi nào nên fine-tune riêng, khi nào dùng LLM tổng quát là đủ.
3. Đánh giá tách bạch hai khía cạnh thay vì chỉ 1 con số accuracy tổng:
   - **Detection**: precision/recall của việc phát hiện đúng vị trí âm tiết bị lỗi.
   - **Correction**: độ chính xác sửa đúng tại các vị trí đã detect đúng.
   - **Over-correction rate**: tỷ lệ âm tiết vốn đã đúng nhưng bị mô hình sửa nhầm (rất quan trọng vì VSEC không có câu hoàn toàn đúng để học điều này).

Các hướng mở rộng (tùy thời gian, không bắt buộc — cần thảo luận và xác nhận phạm vi cụ thể trước khi triển khai):

- Pipeline 2 giai đoạn: detection bảo thủ (rule/tagger) → correction có kiểm soát (seq2seq), tránh để model tự do viết lại cả câu.
- Entity masking cho tên riêng/số liệu/mã số trước khi đưa vào model.
- Data augmentation: sinh thêm câu đúng hoàn toàn + lỗi tổng hợp (synthetic noise) để bù cho việc VSEC 100% câu đều có lỗi.
- Fine-tuning tiết kiệm tài nguyên bằng LoRA/PEFT thay vì full fine-tune.

## 3. Dataset: VSEC (Vietnamese Spelling Error Correction)

- **Nguồn**: Hugging Face — `nguyenthanhasia/vsec-vietnamese-spell-correction`
  https://huggingface.co/datasets/nguyenthanhasia/vsec-vietnamese-spell-correction
- **Paper gốc**: Do, Dinh-Truong, Nguyen, Ha Thanh, Bui, Thang Ngoc, Vo, Hieu Dinh. _"VSEC: Transformer-based Model for Vietnamese Spelling Correction"_, PRICAI 2021, pp. 259–272.
  ```bibtex
  @inproceedings{do2021vsec,
    title={Vsec: Transformer-based model for vietnamese spelling correction},
    author={Do, Dinh-Truong and Nguyen, Ha Thanh and Bui, Thang Ngoc and Vo, Hieu Dinh},
    booktitle={Pacific Rim International Conference on Artificial Intelligence},
    pages={259--272},
    year={2021},
    organization={Springer}
  }
  ```
- **Thống kê**: 9.341 câu, 11.202 lỗi, ~5.211 loại lỗi khác nhau, trung bình ~1.2 lỗi/câu.
- **Chỉ có 1 split `train`** (không có split test/dev chính thức từ nguồn) → cần tự chia (xem mục 5, chống data leakage).
- **Đặc điểm quan trọng cần lưu ý khi mô hình hóa**: **100% câu trong dataset đều có lỗi** (`has_errors` luôn `true`). Không có ví dụ câu hoàn toàn đúng để model học "khi nào không cần sửa gì" → nguy cơ over-correction cao nếu chỉ train trực tiếp trên VSEC.
- **Loại lỗi**: gồm "mistyped errors" (lỗi gõ phím — non-word/real-word errors) và "misspelled errors" (do phát âm vùng miền hoặc từ khó viết). Đây là lỗi **thật do con người viết ra**, không phải lỗi giả lập.
- **License**: dạng "other", có ràng buộc bản quyền — nguồn dữ liệu gốc từ báo chí + tài liệu giáo dục (TaiLieu.VN) có bản quyền. Dùng cho nghiên cứu/giáo dục (đúng mục đích đồ án này) là phù hợp; nếu sau này mở rộng sang mục đích thương mại cần xem lại license kỹ hơn.

### 3.1. Cấu trúc trường dữ liệu (Data Fields)

| Trường                               | Kiểu    | Mô tả                                                                                 |
| ------------------------------------ | ------- | ------------------------------------------------------------------------------------- |
| `text`                               | string  | Câu gốc, có thể chứa lỗi chính tả                                                     |
| `corrected_text`                     | string  | Câu đã được sửa đúng                                                                  |
| `syllable_annotations`               | list    | Annotation chi tiết theo từng âm tiết                                                 |
| `syllable_annotations[].syllable`    | string  | Nội dung âm tiết                                                                      |
| `syllable_annotations[].is_correct`  | boolean | Âm tiết viết đúng hay sai                                                             |
| `syllable_annotations[].corrections` | list    | Danh sách gợi ý sửa (nếu sai)                                                         |
| `syllable_annotations[].position`    | int     | **Vị trí âm tiết trong câu** (0-indexed, syllable index — không phải character index) |
| `error_count`                        | int     | Tổng số lỗi trong câu                                                                 |
| `error_positions`                    | list    | Danh sách vị trí (syllable index) xuất hiện lỗi                                       |
| `correction_pairs`                   | list    | Danh sách cặp (lỗi, sửa) kèm vị trí                                                   |
| `correction_pairs[].error`           | string  | Âm tiết/cụm sai                                                                       |
| `correction_pairs[].correction`      | string  | Âm tiết/cụm đúng                                                                      |
| `correction_pairs[].position`        | int     | Vị trí lỗi (syllable index)                                                           |
| `has_errors`                         | boolean | Câu có lỗi hay không (luôn `true` trong dataset này)                                  |

### 3.2. Ví dụ dữ liệu

```json
{
  "text": "Thông qua công tác tuyên truyền, vận động này phụ huynh sẽ hiểu rõ hơn tầm quan trọng của việc gìn giữ môi trường sanh , sạch , đẹp.",
  "corrected_text": "Thông qua công tác tuyên truyền, vận động này phụ huynh sẽ hiểu rõ hơn tầm quan trọng của việc gìn giữ môi trường xanh , sạch , đẹp.",
  "error_count": 1,
  "error_positions": [51],
  "correction_pairs": [
    { "error": "sanh", "correction": "xanh", "position": 51 }
  ],
  "has_errors": true
}
```

Lưu ý thêm khi đọc dữ liệu thật: `syllable_annotations` có thể chứa các trường hợp đặc biệt — âm tiết bị tách sai (ví dụ "Tuy nh iên" → annotation trên "nh" và "iên,"), âm tiết bị dính hai từ ("vàahệ" → "và hệ"), hoặc lỗi chèn ký tự thừa ở đầu từ ("aNăng" → "Năng"). Cần xử lý các trường hợp edge-case này khi viết code tiền xử lý, không giả định mỗi lỗi luôn là 1-đổi-1 âm tiết.

## 4. Bài toán & hướng mô hình hóa (đề xuất — cần thảo luận cụ thể trước khi code)

Có thể mô hình hóa theo 1 trong các hướng sau (agent cần hỏi người dùng chọn hướng nào trước khi triển khai):

1. **Sequence labeling + correction candidate**: gắn nhãn đúng/sai cho từng âm tiết (binary classification per token) + sinh/chọn gợi ý sửa cho âm tiết sai.
2. **Sequence-to-sequence**: câu lỗi → câu đúng (ví dụ fine-tune BARTpho-syllable), có thể kết hợp copy-mechanism để hạn chế viết lại ngoài ý muốn.
3. **Pipeline 2 giai đoạn** (tham khảo hướng nghiên cứu 2026, xem mục 7): detection riêng (rule-based/tagger, bảo thủ) → correction riêng (seq2seq), chỉ sửa tại vị trí đã được detect.

## 5. Tiền xử lý dữ liệu & chống Data Leakage (BẮT BUỘC tuân thủ)

Theo yêu cầu của người dùng, mọi code training/finetuning phải đảm bảo **không rò rỉ dữ liệu (data leakage)**:

- Chia tập **train/validation/test theo câu** (không theo âm tiết), thực hiện chia tập **trước khi** làm bất kỳ bước tiền xử lý nào cần thống kê toàn cục (ví dụ: xây từ điển lỗi thường gặp, tần suất âm tiết, vocabulary...).
- Nếu dùng augmentation (tự sinh thêm lỗi từ câu đúng, hoặc thêm câu đúng hoàn toàn để giảm over-correction): augmentation chỉ áp dụng trên tập train, **không** áp dụng lên dev/test.
- Lọc/deduplicate câu trùng lặp hoặc gần giống (near-duplicate) trước khi chia tập, tránh cùng một câu (hoặc biến thể) xuất hiện ở cả train và test.
- Không dùng `corrected_text`, `correction_pairs`, hay bất kỳ nhãn nào của tập test/dev trong bước fit tham số/thống kê của mô hình hoặc pipeline tiền xử lý.
- Cố định seed khi chia tập để tái lập được; log rõ tỷ lệ chia và phương pháp (random/stratified theo `error_count`...).
- Nếu bổ sung dữ liệu từ nguồn ngoài (ví dụ câu đúng hoàn toàn để cân bằng), phải đảm bảo nguồn đó không trùng lặp với phần được dùng làm test.

## 6. Đánh giá (Evaluation)

Không chỉ báo cáo accuracy tổng thể. Cần tách riêng:

- **Detection**: precision, recall, F1 của việc xác định đúng vị trí âm tiết lỗi (so với `error_positions`).
- **Correction accuracy**: trong số các vị trí detect đúng, tỷ lệ sửa đúng thành `correction_pairs[].correction`.
- **Over-correction rate**: tỷ lệ âm tiết vốn `is_correct = true` nhưng bị mô hình sửa/thay đổi — chỉ số này đặc biệt quan trọng do đặc điểm dataset (100% câu có lỗi, không có mẫu câu đúng hoàn toàn để học).
- (Tùy chọn, nếu làm phần so sánh LLM prompting) So sánh các chỉ số trên giữa: baseline fine-tune vs. LLM few-shot prompting.

## 7. Tài liệu tham khảo liên quan (2026)

- Huynh, H.T.N., Nguyen, L.S.T., Nguyen, N.H., Nguyen, H.M., Quan, T.T. _"A Two-Stage Vietnamese Spelling Correction Pipeline Combining Underthesea and BARTpho"_, VNU Journal of Science: Comp. Science & Com. Eng., Vol. 42, No. 1 (2026), pp. 79–91.
  https://jcsce.vnu.edu.vn/index.php/jcsce/article/view/7020/222
  → Nêu vấn đề over-correction của seq2seq đơn thuần dưới domain shift; đề xuất pipeline: text normalization + conservative detection (Underthesea) + entity masking → context-aware correction (BARTpho) → detector-guided post-processing + iterative masked refinement.

## 8. Trạng thái hiện tại

- [x] Xác định dataset (VSEC) và hiểu rõ schema.
- [x] Xác định mục tiêu học thuật cho môn NLP.
- [x] Phase 0 hoàn thành: `nb0_data_prep` (chuẩn hóa, chia tập 90/10 stratified seed 42) & `nb1_align_annotate` (align Levenshtein cho bộ test 6k).
- [x] Phase 1 Pilot hoàn thành: `nb2_pilot_dict_noise` (đo non-word rate = 22.7% < 40% $\to$ chính thức bỏ hướng C; bootstrap noise model từ train).
- [x] Đã chốt hướng mô hình hóa: Seq2Seq neural dùng `vinai/bartpho-syllable` + LoRA fp16; so sánh đối chứng 2 run (Run 1: thuần, Run 2: kèm augmentation).
- [x] Đã chốt: Bỏ phần so sánh LLM prompting (22/09/2026).
- [x] Triển khai `nb3-baseline-train-eval.ipynb` trên Kaggle 2× T4 (22–23/09/2026): Run 1 thuần + Run 2 augmentation — Detection F1 test 76,43% → 80,45%, over-correction 1,82–1,88% (< ngưỡng 2–3%).
- [x] Triển khai `nb3b-zeroshot-baseline.ipynb` (eval-only, 22–23/09/2026): Identity + Zero-shot — fine-tune đóng góp lớn (F1 val 9,41% → 76,16%); over-correction có sẵn ở base model (zero-shot 1,56%).
- [x] Đối sánh external với NomVN (`nb5`, eval-only, 24/09/2026): Run 1/Run 2 trên bench ngoài `eval-real` (WA ≈55,6%; identity floor 51,96%; nhóm dẫn đầu 77–80%) + `nomvn-base` trên test 6k (Detection F1 87,26% > 80,45% — so sánh định hướng; confound trùng corpus ghi nhận làm hạn chế) → mở gate rule Telex/VNI cho Run 3 (chi tiết REPORT.md mục 12).
- [ ] Phase 2: hiệu chỉnh noise model (lệch 28,2 điểm) → Run 3; post-processing chống FP deletion; (tùy quota) full-FT ablation.

---

_Agent đọc file này cần tuân thủ mục 0 (Quy tắc làm việc bắt buộc) trong suốt quá trình làm việc trên dự án._
