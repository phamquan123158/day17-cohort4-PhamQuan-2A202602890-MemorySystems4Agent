# Bước 8 — Phân tích kết quả benchmark

## Kết quả offline

| Benchmark | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---:|---:|---:|---:|---:|---:|
| Standard | Baseline | 1803 | 15692 | 0.00 | 0.10 | 0 | 0 |
| Standard | Advanced | 1785 | 25254 | 0.78 | 0.82 | 3172 | 0 |
| Long-context stress | Baseline | 1127 | 26223 | 0.00 | 0.10 | 0 | 0 |
| Long-context stress | Advanced | 393 | 15037 | 0.75 | 0.82 | 268 | 4 |

## 1. Vì sao Advanced có recall tốt hơn Baseline?

Ở Standard Benchmark, `Cross-session recall` của Baseline là **0.00**, còn
Advanced là **0.78**. Ở Long-Context Stress Benchmark, hai giá trị tương ứng
là **0.00** và **0.75**.

Advanced có recall tốt hơn vì `extract_profile_updates()` trích các fact ổn
định từ message, `_persist_updates()` ghi chúng vào `User.md`, sau đó
`_offline_response()` đọc lại profile khi người dùng hỏi trong thread mới.
Baseline chỉ lưu message theo `thread_id`, nên không có dữ liệu persistent khi
thread thay đổi.

Recall của Advanced chưa đạt 1.00 vì extraction hiện vẫn là heuristic. Những
cách diễn đạt mới, câu ghép hoặc correction không theo mẫu đã biết có thể
không được trích chính xác.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Trong Standard Benchmark, `Agent tokens only` của Baseline là **1803**, còn
Advanced là **1785**. Tuy nhiên `Prompt tokens processed` của Baseline là
**15692**, còn Advanced là **25254**, tức Advanced cao hơn **9562 prompt
tokens** trong lần chạy này.

Nguyên nhân là mỗi lượt Advanced phải mang theo `User.md`, summary và các
message gần nhất trong context. Advanced cũng phải cập nhật profile khi phát
hiện fact mới. Với hội thoại ngắn, lịch sử chưa đủ dài để compact tạo lợi thế,
nên chi phí persistent memory chưa được bù lại. Vì vậy Advanced có thể tốn hơn
ở prompt processing dù số output token hiện tại hơi thấp hơn Baseline.

## 3. Vì sao compact có lợi thế ở hội thoại dài?

Trong Long-Context Stress Benchmark, `Prompt tokens processed` của Baseline là
**26223**, còn Advanced là **15037**, giảm **11186 prompt tokens**. Baseline có
**0 compaction**, trong khi Advanced có **4 compactions**.

`CompactMemoryManager` chuyển message cũ thành summary và chỉ giữ lại một số
message gần nhất. Vì vậy compact tối ưu trực tiếp cột **`Prompt tokens
processed`**: những lượt sau không phải gửi lại toàn bộ lịch sử nguyên văn.

Compact không được hiểu đơn giản là luôn làm giảm `Agent tokens only`. Trong
bảng này output tokens là **1127** với Baseline và **393** với Advanced, nhưng
đây là kết quả của offline response ngắn hơn; chỉ số mà compact được thiết kế
để tối ưu là prompt context load.

## 4. File memory tăng trưởng ra sao và rủi ro gì?

Ở Standard Benchmark, `Memory growth (bytes)` của Baseline là **0**, còn
Advanced là **3172 bytes**. Ở Stress Benchmark, Baseline là **0**, Advanced là
**268 bytes**, và Advanced thực hiện **4 compactions**.

Baseline không có `User.md`, nên không phát sinh persistent memory. Advanced
ghi các fact ổn định vào `User.md`, vì vậy file tăng theo số fact và có thể
phình theo thời gian. Nếu ghi duplicate facts hoặc ghi cả message không ổn
định, profile sẽ làm prompt ngày càng đắt.

Rủi ro khác là một fact sai từ một lượt nhiễu có thể bị giữ lâu dài và ảnh
hưởng các thread sau. Project xử lý rủi ro này bằng confidence threshold để
bỏ qua câu hỏi/câu đùa, đồng thời dùng conflict handling để correction mới
thay thế fact cũ. Trade-off là threshold quá cao có thể bỏ sót một fact đúng
nhưng diễn đạt không rõ.

## Kết luận

Baseline đơn giản và phù hợp làm control: nó không nhớ dài hạn và có chi phí
context tăng theo lịch sử thread. Advanced phức tạp hơn nhưng có continuity
qua nhiều session, recall cao hơn, và compact giúp giảm prompt context trong
hội thoại dài. Đổi lại, Advanced phải kiểm soát memory growth, conflict và
confidence của fact trước khi lưu.
