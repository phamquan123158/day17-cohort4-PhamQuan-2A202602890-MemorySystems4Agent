# Phase 2, Track 3, Day 17: Memory Systems for AI Agent

Trong Day 17 này, các bạn sẽ tập trung vào một câu hỏi rất thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** mà vẫn kiểm soát được chi phí token.

Trong bài lab này, các bạn sẽ xây dựng và so sánh hai agent:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài

Mục tiêu cuối cùng không phải chỉ là “agent nhớ nhiều hơn”, mà là hiểu rõ trade-off giữa:

- độ nhớ dài hạn
- chất lượng phản hồi
- chi phí token
- độ phức tạp của hệ thống memory

## Các bạn sẽ làm gì trong track này?

Sau khi hoàn thành, các bạn cần có khả năng:

- phân biệt `short-term memory`, `persistent memory`, và `compact memory`
- xây dựng agent baseline và advanced trên cùng một benchmark
- lưu hồ sơ người dùng bằng `User.md`
- kích hoạt compact memory khi hội thoại dài vượt ngưỡng
- benchmark hai agent bằng cùng một bộ dữ liệu tiếng Việt
- đọc kết quả benchmark theo các chỉ số recall, token, memory growth, chất lượng phản hồi

## Cấu trúc codebase

```
.
├── README.md        # giới thiệu track (file này)
├── Guide.md         # hướng dẫn từng bước
├── Rubric.md        # tiêu chí chấm điểm
├── data/            # dữ liệu benchmark dùng chung
│   ├── conversations.json
│   └── advanced_long_context.json
└── src/             # bản scaffold dành cho sinh viên (pseudocode + TODO)
    ├── model_provider.py
    ├── config.py
    ├── memory_store.py
    ├── agent_baseline.py
    ├── agent_advanced.py
    ├── benchmark.py
    └── test_agents.py
```

Khi chạy, agent sẽ ghi trạng thái (ví dụ `state/profiles/<user>/User.md`) vào thư mục `state/`. Thư mục này đã nằm trong `.gitignore`.

### Vai trò từng file trong `src/`

Các file được liệt kê theo thứ tự nên triển khai:

| File | Vai trò | Thành phần chính |
|---|---|---|
| `model_provider.py` | Khởi tạo chat model cho từng provider | `ProviderConfig`, `normalize_provider()`, `build_chat_model()` |
| `config.py` | Cấu hình chung của lab | `LabConfig` (đường dẫn, ngưỡng compact, model chính + judge), `load_config()` |
| `memory_store.py` | Lõi memory layer | `estimate_tokens()`, `UserProfileStore` (read/write/edit `User.md`), `extract_profile_updates()`, `summarize_messages()`, `CompactMemoryManager` |
| `agent_baseline.py` | Agent A: chỉ nhớ trong cùng thread | `BaselineAgent.reply()`, `token_usage()`, `prompt_token_usage()` |
| `agent_advanced.py` | Agent B: short-term + `User.md` + compact | `AdvancedAgent.reply()`, `_reply_offline()`, `_estimate_prompt_context_tokens()`, `_offline_response()` |
| `benchmark.py` | So sánh hai agent trên hai bộ dữ liệu | `run_agent_benchmark()`, `recall_points()`, `heuristic_quality()`, `format_rows()` |
| `test_agents.py` | Kiểm chứng hành vi memory | test `User.md`, compact trigger, cross-session recall, giảm prompt load |

### Luồng xử lý một lượt của Advanced Agent

```
message người dùng
  → extract_profile_updates()      # trích fact ổn định: tên, nơi ở, nghề, style...
  → ghi vào User.md                # persistent memory
  → CompactMemoryManager.append()  # short-term memory, tự compact khi vượt ngưỡng
  → prompt = User.md + summary + recent messages
  → sinh câu trả lời → cập nhật bộ đếm token
```

Baseline Agent chỉ giữ danh sách message theo `thread_id`. Sang thread mới, nó **phải quên** toàn bộ fact cũ.

Cả hai agent nên có **chế độ offline** cho ra kết quả lặp lại được, để benchmark và test chạy được mà không cần API key. Chế độ live (LangChain/LangGraph) là phần mở rộng.

## Dữ liệu benchmark

| File | Nội dung | Mục tiêu |
|---|---|---|
| `data/conversations.json` | 10 hội thoại khoảng 10 lượt, user `dungct`, kèm `recall_questions` | Standard benchmark: đo recall qua nhiều phiên bình thường |
| `data/advanced_long_context.json` | 1 hội thoại 16 lượt rất dài, user `dungct_stress` | Long-context stress benchmark: ép compact xảy ra nhiều lần |

Mỗi hội thoại có dạng:

```json
{
  "id": "conv-01",
  "user_id": "dungct",
  "turns": ["...", "..."],
  "recall_questions": [
    { "question": "...", "expected_contains": ["DũngCT", "cà phê sữa đá"] }
  ]
}
```

`recall_questions` được hỏi ở **thread mới**. Điểm recall dựa trên số chuỗi trong `expected_contains` xuất hiện trong câu trả lời.

Dữ liệu cố tình chứa các tình huống khó:

- **correction**: nơi ở đổi giữa Đà Nẵng và Huế, agent phải giữ fact mới nhất
- **nhiễu**: "Hà Nội" chỉ là nơi đi họp, "product manager" chỉ là câu đùa
- **ngữ cảnh dài**: nhiều đoạn tin tức dài trong stress test để làm lộ chi phí prompt của baseline

## Provider hỗ trợ

Trong bản solved lab, runtime hỗ trợ các provider sau:

- `openai`
- `custom` (OpenAI-compatible base URL)
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Điều này quan trọng vì memory system không nên bị khóa vào một provider duy nhất.

## Chỉ số benchmark cần hiểu

Khi hoàn thiện bài, benchmark nên cho các cột sau:

- `Agent tokens only`: token sinh ra trực tiếp trong hội thoại của agent
- `Prompt tokens processed`: lượng ngữ cảnh agent phải kéo theo qua các lượt
- `Cross-session recall`: khả năng nhớ facts qua thread hoặc session mới
- `Response quality`: chất lượng phản hồi
- `Memory growth (bytes)`: tốc độ phình của file memory
- `Compactions`: số lần compact memory đã nén lịch sử cũ

Điểm quan trọng nhất của track này là:

- ở hội thoại ngắn, `Advanced` có thể tốn hơn `Baseline` về token usage
- ở hội thoại rất dài, compact memory nên giúp `Advanced` xử lý ngữ cảnh hiệu quả hơn đáng kể + tiết kiệm usage.

## Setup môi trường

Các bạn cần chuẩn bị môi trường Python `>= 3.11` và cài các package cần thiết cho LangChain, LangGraph, provider SDK, `python-dotenv`, `tabulate`, và `pytest`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

Nếu muốn chạy chế độ live với LLM thật, hãy tạo file `.env` ở root repo (đã nằm trong `.gitignore`). Tên biến môi trường do các bạn quyết định khi viết `load_config()`. Ví dụ:

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=...
```

## Chạy benchmark và test

Sau khi hoàn thiện `src/`, chạy từ root repo:

```bash
python src/benchmark.py
```

```bash
pytest src/test_agents.py -v
```

Benchmark cần in ra hai bảng: **Standard Benchmark** và **Long-Context Stress Benchmark**. Mỗi bảng so sánh Baseline với Advanced theo đủ 6 cột trong phần "Chỉ số benchmark cần hiểu".

## Phân tích kết quả benchmark

Kết quả offline hiện tại là:

| Benchmark | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Memory growth (bytes) | Compactions |
|---|---|---:|---:|---:|---:|---:|
| Standard | Baseline | 1803 | 15692 | 0.00 | 0 | 0 |
| Standard | Advanced | 1785 | 25254 | 0.78 | 3172 | 0 |
| Long-context stress | Baseline | 1127 | 26223 | 0.00 | 0 | 0 |
| Long-context stress | Advanced | 393 | 15037 | 0.75 | 268 | 4 |

Bốn kết luận chính, với số liệu đứng trước cơ chế giải thích:

- **Advanced có recall tốt hơn Baseline.** Ở Standard Benchmark,
  `Cross-session recall` của Baseline là **0.00**, còn Advanced là **0.78**;
  ở Long-Context Stress Benchmark, hai giá trị tương ứng là **0.00** và
  **0.75**. Cơ chế tạo ra chênh lệch này là
  `extract_profile_updates()` trích fact ổn định, `_persist_updates()` ghi
  fact vào `User.md`, rồi `_offline_response()` đọc lại profile khi câu hỏi
  được gửi trong thread mới. Giới hạn là recall Advanced chưa đạt 1.00 vì
  heuristic extraction không hiểu mọi cách diễn đạt, nên correction và câu
  ghép vẫn cần được kiểm thử riêng.
- **Advanced có thể tốn hơn ở hội thoại ngắn.** Ở Standard Benchmark,
  `Agent tokens only` là **1803** với Baseline và **1785** với Advanced;
  riêng `Prompt tokens processed` là **15692** và **25254**, tức Advanced
  đang cao hơn **9562 token prompt** trong lần chạy này. Advanced mỗi lượt
  còn mang theo `User.md` cùng summary/message context và cập nhật file,
  trong khi hội thoại ngắn chưa đủ dài để compact bù lại chi phí persistent
  memory; vì vậy “có thể tốn hơn” chủ yếu thể hiện ở prompt cost, dù output
  tokens của lần chạy hiện tại hơi thấp hơn Baseline.
- **Compact có lợi thế ở hội thoại dài.** Ở stress benchmark,
  `Prompt tokens processed` của Baseline là **26223**, còn Advanced là
  **15037**, giảm **11186 token**, trong khi `Compactions` là **0** và **4**
  tương ứng. `CompactMemoryManager` chuyển message cũ thành summary và chỉ
  giữ message gần nhất, nên compact tối ưu trực tiếp cột **Prompt tokens
  processed**. Nó không đồng nghĩa với việc giảm `Agent tokens only`: trong
  bảng này output tokens giảm từ **1127** xuống **393**, nhưng đó là kết quả
  của response offline ngắn hơn, không phải chỉ số mà compact được thiết kế
  để tối ưu.
- **File memory tăng trưởng và có rủi ro tích lũy sai fact.** Ở Standard,
  `Memory growth (bytes)` là **0** với Baseline và **3172** với Advanced; ở
  stress benchmark, hai giá trị là **0** và **268**, với Advanced thực hiện
  **4 compactions**. `User.md` tăng theo các fact được ghi và summary giữ
  context cũ, nên file có thể phình theo thời gian hoặc prompt có thể đắt
  hơn nếu ghi duplicate facts. Rủi ro thứ hai là một fact sai từ một lượt
  nhiễu bị giữ lại lâu dài; confidence threshold và conflict handling trong
  code được thêm để bỏ qua câu đùa/câu hỏi và thay fact cũ bằng correction
  mới.

- **Độ tin cậy:** Persistent memory cần conflict handling. Khi người dùng đính
  chính nơi ở hoặc nghề nghiệp, fact mới phải thay thế fact cũ; các danh sách
  như interests có thể merge. Summary chỉ nên giữ context hội thoại, không nên
  là nơi duy nhất lưu facts dài hạn.
- **Bonus confidence threshold:** Advanced chấm confidence cho candidate fact
  trước khi ghi vào `User.md`. Câu khẳng định rõ như “mình đang làm MLOps
  engineer” được nhận, còn câu hỏi, câu đùa hoặc thông tin chỉ dùng làm ví dụ
  bị bỏ qua. Threshold này giảm nguy cơ profile ghi sai nhưng có trade-off là
  một câu nói mơ hồ có thể không được lưu, vì vậy correction rõ ràng vẫn cần
  được ưu tiên xử lý.

Khi đọc bảng kết quả, cần nhìn cả ba chiều: recall, prompt cost và memory
growth. Advanced tốt hơn khi cần continuity qua nhiều phiên hoặc khi context
dài; Baseline đơn giản hơn và phù hợp làm control để đo chi phí của persistent
memory.

## Cách dùng repo này

Nếu các bạn là sinh viên:

- làm bài trong `src/`
- dùng `data/` làm benchmark input

Nếu các bạn là giảng viên hoặc reviewer:

- dùng `src/` để đánh giá scaffold giao cho sinh viên và kết quả hoàn thiện cuối cùng

## Tài liệu nên đọc tiếp

- `Guide.md`: hướng dẫn từng bước để hoàn thành lab
- `Rubric.md`: tiêu chí chấm điểm và bonus

Track này được thiết kế để các bạn không chỉ “dùng agent”, mà còn bắt đầu nghĩ như một người thiết kế **memory system** cho agent production.
