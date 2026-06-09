# Codelab: Xây Dựng Hệ Thống Multi-Agent với A2A Protocol

**Thời gian:** 2 giờ  
**Ngôn ngữ:** Python 3.11+  
**Công nghệ:** LangGraph, LangChain, A2A SDK

## Mục Tiêu Học Tập

Sau khi hoàn thành codelab này, bạn sẽ:
- Hiểu cách LLM hoạt động từ cơ bản đến nâng cao
- Biết cách tích hợp tools và RAG vào LLM
- Xây dựng được single agent với ReAct pattern
- Tạo multi-agent system với LangGraph
- Triển khai distributed agents với A2A protocol

## Chuẩn Bị

### Yêu Cầu Hệ Thống
- Python 3.11 trở lên
- [uv](https://docs.astral.sh/uv/) package manager
- API key từ [OpenRouter](https://openrouter.ai)

### Cài Đặt

```bash
# Clone repository
git clone <repo-url>
cd legal_multiagent

# Cài đặt dependencies
uv sync

# Cấu hình environment
cp .env.example .env
# Sửa file .env, thêm OPENROUTER_API_KEY của bạn
```

---

## Phần 1: Direct LLM Calling (20 phút)

### Lý Thuyết

LLM (Large Language Model) ở dạng cơ bản nhất là một API nhận input text và trả về output text. Không có memory, không có tools, chỉ dựa vào training data.

**Ưu điểm:**
- Đơn giản, dễ implement
- Phản hồi nhanh

**Nhược điểm:**
- Không có kiến thức real-time
- Không thể tra cứu database
- Không có context giữa các lần gọi

### Thực Hành

**Bước 1:** Chạy demo Stage 1

```bash
uv run python stages/stage_1_direct_llm/main.py
```

**Bước 2:** Đọc và hiểu code

Mở file `stages/stage_1_direct_llm/main.py` và trả lời:

1. LLM được khởi tạo như thế nào? (Tìm hàm `get_llm()`)
Trong file, câu lệnh llm = get_llm() sẽ gọi đến hàm get_llm(). Hàm này khởi tạo một đối tượng ChatOpenAI (một class của thư viện LangChain) nhưng được cấu hình để "hướng đầu ra" về phía nền tảng OpenRouter (một dịch vụ proxy cho phép gọi nhiều mô hình AI khác nhau).

Cụ thể, các tham số cấu hình khi khởi tạo bao gồm:

model: Lấy từ biến môi trường OPENROUTER_MODEL. Nếu không được thiết lập trong file .env, nó sẽ mặc định sử dụng mô hình anthropic/claude-sonnet-4-5.

openai_api_key: Sử dụng API Key của OpenRouter thông qua biến môi trường OPENROUTER_API_KEY.

openai_api_base: Được chỉ định cứng (hardcode) là "https://openrouter.ai/api/v1". Việc thay đổi base URL này giúp class ChatOpenAI gửi request đến máy chủ OpenRouter thay vì máy chủ mặc định của OpenAI.
2. Message được gửi đến LLM có cấu trúc gì?
Cấu trúc tin nhắn (messages) được gửi đến LLM là một Mảng/Danh sách (List) tuần tự gồm các đối tượng tin nhắn có phân vai (Role-based messages).

Trong đoạn code, danh sách này chứa 2 phần tử theo đúng thứ tự:

Phần tử đầu tiên: Một đối tượng SystemMessage chứa câu lệnh định hướng vai trò.

Phần tử thứ hai: Một đối tượng HumanMessage chứa câu hỏi thực tế của người dùng (QUESTION).
3. Tại sao cần có `SystemMessage` và `HumanMessage`?
Việc chia tách thành hai loại Message này nhằm mục đích phân định rõ ràng quyền hạn và bản chất của thông tin truyền vào mô hình:
- SystemMessage (Tin nhắn hệ thống): * Vai trò: Dùng để thiết lập "bối cảnh nền", quy định phong cách, tính cách, giới hạn hành vi và định dạng câu trả lời cho AI trước khi cuộc hội thoại bắt đầu.
- HumanMessage (Tin nhắn của người dùng): * Vai trò: Đại diện cho nội dung, câu hỏi hoặc yêu cầu trực tiếp từ phía người dùng (ở đây là biến QUESTION về hậu quả pháp lý khi vi phạm NDA).
**Bài Tập 1.1:** Thay đổi câu hỏi

Sửa biến `QUESTION` thành câu hỏi pháp lý khác (tiếng Việt hoặc tiếng Anh) và chạy lại.

**Bài Tập 1.2:** Thêm temperature control

Thêm parameter `temperature=0.3` vào hàm `get_llm()` trong `common/llm.py` để làm output ổn định hơn.

---

## Phần 2: LLM + RAG & Tools (30 phút)

### Lý Thuyết

**RAG (Retrieval-Augmented Generation):** Cho phép LLM tra cứu knowledge base trước khi trả lời.

**Tools:** Các function mà LLM có thể gọi để thực hiện tác vụ cụ thể (tính toán, query database, gọi API).

**Function Calling Flow:**
1. LLM nhận câu hỏi + danh sách tools
2. LLM quyết định gọi tool nào (hoặc không gọi)
3. Tool được execute, trả về kết quả
4. LLM nhận kết quả và tạo câu trả lời cuối cùng

### Thực Hành

**Bước 1:** Chạy demo Stage 2

```bash
uv run python stages/stage_2_rag_tools/main.py
```

**Bước 2:** Phân tích code

Mở `stages/stage_2_rag_tools/main.py` và tìm:

1. Hàm `@tool` decorator được dùng ở đâu?
Decorator @tool (được import từ langchain_core.tools) được sử dụng ngay phía trên định nghĩa của các hàm để biến các hàm Python thông thường thành các công cụ (Tools) mà LangChain và LLM có thể hiểu và gọi được: search_legal_database(query: str), 
calculate_damages(breach_type: str, contract_value: float).

2. `LEGAL_KNOWLEDGE` được cấu trúc như thế nào?
LEGAL_KNOWLEDGE đóng vai trò như một cơ sở dữ liệu tri thức pháp luật giả lập (simulated knowledge base). Nó được cấu trúc dưới dạng Một danh sách (List) chứa các từ điển (Dictionaries).

Mỗi một phần tử (mỗi từ điển) trong danh sách đại diện cho một mẩu thông tin pháp lý và có cấu trúc gồm 3 cặp key-value cố định:

- id (Kiểu dữ liệu: String): Mã định danh duy nhất cho mẩu thông tin đó (Ví dụ: "ucc_breach", "nda_trade_secret").

- keywords (Kiểu dữ liệu: List của các Strings): Danh sách các từ khóa liên quan. Hàm search_legal_database sẽ quét qua danh sách từ khóa này để so khớp với câu hỏi của người dùng và tính điểm tương đồng.

- text (Kiểu dữ liệu: String): Nội dung chi tiết của điều luật hoặc án lệ (ví dụ: nội dung về bộ luật thương mại Mỹ UCC, luật bảo vệ bí mật kinh doanh DTSA, v.v.).
3. LLM được bind với tools ra sao? (Tìm `.bind_tools()`)
Quá trình liên kết (bind) các công cụ vào LLM được thực hiện bên trong hàm main() thông qua phương thức .bind_tools() của đối tượng llm.

Quy trình diễn ra qua các bước cụ thể trong code như sau:

- Gom nhóm các công cụ: Đầu tiên, hai hàm đã được gắn decorator @tool được gom lại vào một danh sách đặt tên là TOOLS
- Khởi tạo LLM gốc: Khởi tạo đối tượng mô hình ngôn ngữ
- Thực hiện Bind (Liên kết): Gọi phương thức .bind_tools(TOOLS) và gán vào một biến mới là llm_with_tools
**Bài Tập 2.1:** Thêm knowledge base entry

Thêm một entry mới vào `LEGAL_KNOWLEDGE` về luật lao động:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
```

**Bài Tập 2.2:** Tạo tool mới

Tạo một tool `@tool` mới tên `check_statute_of_limitations` nhận vào `case_type` (string) và trả về thời hiệu khởi kiện:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.
    
    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

Thêm tool này vào danh sách tools và test.

---

## Phần 3: Single Agent với ReAct (25 phút)

### Lý Thuyết

**ReAct Pattern:** Reasoning + Acting

Agent tự động lặp lại chu trình:
1. **Think:** Suy nghĩ cần làm gì
2. **Act:** Gọi tool
3. **Observe:** Nhận kết quả
4. Lặp lại cho đến khi có câu trả lời cuối cùng

LangGraph cung cấp `create_react_agent` để tự động hóa pattern này.

### Thực Hành

**Bước 1:** Chạy demo Stage 3

```bash
uv run python stages/stage_3_single_agent/main.py
```

**Bước 2:** Quan sát output

Chú ý cách agent tự động:
- Quyết định tool nào cần gọi
- Gọi nhiều tools liên tiếp
- Tổng hợp kết quả

**Bước 3:** Đọc code

Mở `stages/stage_3_single_agent/main.py`:

1. Tìm `create_react_agent()` — đây là magic function
2. So sánh với Stage 2: không còn manual tool loop
3. Xem `agent_executor.invoke()` — chỉ cần gọi một lần

**Bài Tập 3.1:** Thêm tool tra cứu án lệ

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa.
    
    Args:
        keywords: Từ khóa tìm kiếm
    """
    cases = {
        "breach": "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }
    for key, case in cases.items():
        if key in keywords.lower():
            return case
    return "Không tìm thấy án lệ phù hợp"
```

Thêm vào tools list và test với câu hỏi về breach of contract.

**Bài Tập 3.2:** Debug agent reasoning

Thêm `verbose=True` vào `create_react_agent()` để xem chi tiết quá trình suy nghĩ của agent.

---

## Phần 4: Multi-Agent In-Process (30 phút)

### Lý Thuyết

**Multi-Agent System:** Nhiều agents chuyên môn hóa cùng làm việc.

**Ưu điểm:**
- Mỗi agent tập trung vào domain riêng
- Có thể chạy song song (parallel execution)
- Dễ maintain và mở rộng

**LangGraph StateGraph:**
- Định nghĩa state (dữ liệu chia sẻ giữa các nodes)
- Tạo nodes (các bước xử lý)
- Định nghĩa edges (luồng điều khiển)

**Send API:** Cho phép dispatch nhiều tasks song song.

### Thực Hành

**Bước 1:** Chạy demo Stage 4

```bash
uv run python stages/stage_4_milti_agent/main.py
```

**Bước 2:** Phân tích kiến trúc

Mở `stages/stage_4_milti_agent/main.py`:

1. Tìm `class State(TypedDict)` — đây là shared state
2. Tìm các agent functions: `law_agent`, `tax_agent`, `compliance_agent`
3. Tìm `Send()` API — dispatch parallel tasks
4. Xem `graph.add_node()` và `graph.add_edge()`

**Bước 3:** Vẽ graph

```python
# Thêm vào cuối file main.py
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

**Bài Tập 4.1:** Thêm agent mới

Tạo `privacy_agent` chuyên về GDPR và privacy law:

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về luật bảo vệ dữ liệu cá nhân."""
    llm = get_llm()
    
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.
    
Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy và GDPR (nếu có).
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

Thêm node này vào graph và kết nối với `aggregate_results`.

**Bài Tập 4.2:** Implement conditional routing

Sửa `check_routing` để chỉ gọi privacy_agent khi câu hỏi có từ khóa "data", "privacy", "gdpr":

```python
def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []
    
    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))
    
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))
    
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))
    
    return tasks if tasks else [Send("aggregate_results", state)]
```

---

## Phần 5: Distributed A2A System (15 phút)

### Lý Thuyết

**A2A (Agent-to-Agent) Protocol:** Chuẩn giao tiếp giữa các agents qua HTTP.

**Khác biệt với Stage 4:**
- Mỗi agent là một service độc lập
- Giao tiếp qua HTTP thay vì in-process
- Dynamic discovery qua Registry
- Có thể scale từng agent riêng biệt

**Kiến trúc:**
```
Registry (10000) ← agents register on startup
    ↓
Customer Agent (10100) → Law Agent (10101)
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            Tax Agent (10102)   Compliance Agent (10103)
```

### Thực Hành

**Bước 1:** Khởi động toàn bộ hệ thống

```bash
./start_all.sh
```

Chờ ~10 giây để tất cả services khởi động.

**Bước 2:** Test hệ thống

```bash
uv run python test_client.py
```

**Bước 3:** Quan sát logs

Mở 5 terminal tabs và xem logs của từng service:
- Registry: port 10000
- Customer Agent: port 10100
- Law Agent: port 10101
- Tax Agent: port 10102
- Compliance Agent: port 10103

**Bài Tập 5.1:** Trace request flow

Trong logs, tìm `trace_id` và theo dõi request đi qua các agents. Vẽ sequence diagram.

**Bài Tập 5.2:** Test dynamic discovery

1. Dừng Tax Agent (Ctrl+C)
2. Chạy lại `test_client.py`
3. Quan sát lỗi và cách hệ thống xử lý

**Bài Tập 5.3:** Modify agent behavior

Sửa `tax_agent/graph.py`, thay đổi system prompt để agent trả lời ngắn gọn hơn. Restart tax agent và test lại.

---

## Phần 6: Tổng Kết & Mở Rộng (10 phút)

### So Sánh 5 Stages

| Stage | Pattern | Use Case | Complexity |
|---|---|---|---|
| 1 | Direct LLM | Câu hỏi đơn giản, không cần tools | ⭐ |
| 2 | LLM + Tools | Cần tra cứu data hoặc tính toán | ⭐⭐ |
| 3 | ReAct Agent | Tự động orchestration, multi-step | ⭐⭐⭐ |
| 4 | Multi-Agent | Nhiều domains, parallel processing | ⭐⭐⭐⭐ |
| 5 | Distributed A2A | Production, scalable, fault-tolerant | ⭐⭐⭐⭐⭐ |

### Câu Hỏi Ôn Tập

1. Khi nào nên dùng single agent thay vì multi-agent?
- Tác vụ đơn giản, tuyến tính: Quy trình xử lý cố định, ít rẽ nhánh tư duy phức tạp.

- Phạm vi kiến thức hẹp: Chỉ cần tương tác với một vài công cụ thuộc cùng một chuyên môn.

- Tối ưu tài nguyên: Cần tiết kiệm token, giảm chi phí API và hạ thấp độ trễ (latency) hệ thống.

- Dễ phát triển/debug: Quản lý một State tập trung, không phân tán.
2. Ưu điểm của A2A protocol so với gRPC hoặc REST thông thường?
- Chuẩn hóa cấu trúc tin nhắn AI: Định nghĩa sẵn định dạng chuyên biệt cho LLM gồm phân vai (Role), các phần nội dung (Parts) và kết quả công cụ (Artifacts).

- Quản lý vòng đời tác vụ phân tán: Hỗ trợ cơ chế tự động ủy quyền (delegation) và đính kèm lịch sử hội thoại giữa các agent phân tán.

- Khả năng quan sát (Observability): Tích hợp sẵn cơ chế theo dõi luồng request (như trace_id) xuyên qua nhiều dịch vụ.
3. Làm thế nào để prevent infinite delegation loops trong A2A?
- Giới hạn độ sâu ủy quyền (Max Depth): Gắn một biến đếm số lượt chuyển giao (depth) vào metadata; nếu vượt ngưỡng (ví dụ: > 5) lập tức ngắt luồng.

- Lưu vết lịch sử định tuyến (Routing History): Đính kèm danh sách các agent đã xử lý request; chặn không cho chuyển giao đến agent đã có tên trong danh sách.

- Thiết kế đồ thị một chiều (DAG): Ép luồng đi theo cấu trúc phân cấp nghiêm ngặt (ví dụ: Chuyên gia không được gọi ngược lại Điều phối viên).
4. Tại sao cần Registry service? Có thể hardcode URLs không?
- Tại sao cần Registry: Đóng vai trò như một danh bạ trung tâm để các Agent tự động đăng ký địa chỉ khi khởi động và giúp các Agent khác tự động tìm thấy nhau mà không cần cấu hình thủ công.

- Có thể hardcode không: * Có: Khi làm bài tập nhỏ ở local (ví dụ: gán cứng http://localhost:10102).

  - Không nên (khi lên Production): Vì làm mất khả năng tự động mở rộng (Scaling), không thể tự động phát hiện lỗi khi một máy chủ bị sập (Fault-tolerance), và cực kỳ khó bảo trì khi hệ thống tăng lên hàng chục Agent.
### Bài Tập Nâng Cao (Tự Học)

**Challenge 1:** Thêm memory/conversation history

Implement conversation memory để agent nhớ các câu hỏi trước đó.

**Challenge 2:** Add authentication

Thêm API key authentication cho các A2A endpoints.

**Challenge 3:** Implement retry logic

Khi một agent fail, tự động retry với exponential backoff.

**Challenge 4:** Monitoring & Observability

Tích hợp LangSmith hoặc Prometheus để monitor agent performance.

---

## Tài Liệu Tham Khảo

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [A2A Protocol Spec](https://github.com/google/A2A)
- [OpenRouter API](https://openrouter.ai/docs)
- Architecture diagrams: `docs/*.svg`

## Hỗ Trợ

Nếu gặp vấn đề:
1. Check `.env` file có đúng API key không
2. Đảm bảo tất cả ports (10000-10103) không bị chiếm
3. Xem logs trong terminal để debug
4. Đọc error messages cẩn thận — thường có hint rõ ràng

---

## Bài Tập Cộng Điểm:
1. Vite Code HTML File Để demo các tương tác của các Agent ở stage 4 hoặc stage 5
2. Sau khi chạy full Stage 5 (test_client.py) trả lời 2 câu hỏi:
- Latency (Tổng thời gian trả lời 1 câu hỏi của hệ thống) là bao nhiêu giây?
- Đề xuất phương án giảm latency và demo + show thời gian xử lý đã giảm được khi apply phương án?
**Chúc các bạn học tốt! 🚀**
