# nlp-2026 — Sửa lỗi chính tả tiếng Việt (VSEC)

Đồ án môn NLP: **phát hiện + sửa lỗi chính tả tiếng Việt mức âm tiết** trên dataset VSEC. Hướng tiếp cận đã chốt: seq2seq `vinai/bartpho-syllable` + LoRA (fp16 trên Kaggle 2× T4), đánh giá tách bạch 3 nhóm chỉ số — detection / correction / over-correction. Bối cảnh, mục tiêu và ràng buộc đầy đủ: xem [`PROJECT.md`](PROJECT.md).

## Tài liệu dự án (thứ tự đọc)

| File | Nội dung | Đọc khi |
|---|---|---|
| [`PROJECT.md`](PROJECT.md) | Bối cảnh, mục tiêu, dataset VSEC (schema §3.1), quy tắc làm việc bắt buộc (§0), chống data leakage (§5), định nghĩa metric (§6) | Đầu tiên — bắt buộc |
| [`DESIGN.md`](DESIGN.md) | Quyết định phạm vi đã chốt (§1), lộ trình phase (§4), định nghĩa metric (§6), hình thức notebook-first trên Kaggle (§7), checklist leakage (§9), nhật ký quyết định (§12) | Trước khi code / thí nghiệm |
| [`REPORT.md`](REPORT.md) | Kết quả chi tiết theo từng notebook: tiến độ (§1.1), kết quả chính (§1.2), chi tiết Phase 0–2 (§2–§7), hạn chế đo lường (§10), danh mục artifact (§11) | Khi cần số liệu / căn cứ |
| [`AGENTS.md`](AGENTS.md) | Quy tắc làm việc cho coding agent (tiếng Anh) | Agent đọc trước khi làm việc |

## Trạng thái hiện tại

Đang ở **Phase 2 · Bước 2**: post-processing chống deletion + Run 3 (noise hiệu chỉnh, non-word mục tiêu 23–25%) — chi tiết tại `REPORT.md`, mục 1.1 và 9; nhật ký quyết định tại `DESIGN.md`, mục 12.

> Bảo trì: mỗi lần cập nhật `REPORT.md` / `DESIGN.md`, soát lại dòng trạng thái ở trên.

## Cấu trúc repo

| Đường dẫn | Nội dung |
|---|---|
| `notebooks/` | 8 notebook Kaggle: `nb0-data-pre` → `nb1-align-annotate` → `nb2-pilot-dict-noise` → `nb3-baseline-train-eval` → `nb3b-zeroshot-baseline` → `nb4-error-analysis` → `nb4b-adjudication-llm-fill` → `nb4c-adjudication-audit` — vai trò từng notebook: `REPORT.md`, mục 1.1 |
| `data/` | `6000.csv` (bộ test tự thu thập: cột `text` + `label`) và artifact audit cục bộ của nb4c (`adjudication_samples.json`, `audit_filled.json`) — danh mục artifact đầy đủ: `REPORT.md`, mục 11 |

## Cách chạy

Các notebook chạy trên **Kaggle (2× T4, fp16)**, nối Output/Input theo chuỗi nb0 → nb1 → nb2/nb3 → nb3b → nb4/nb4b; `nb4c` chạy cục bộ trong repo. Quy trình, tham số và tài nguyên chi tiết: `DESIGN.md`, mục 7. Repo **cố ý không chứa mã `src/`** (notebook-first — chỉ trích module khi thật cần, xem `DESIGN.md`, mục 7).
