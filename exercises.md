# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Khi temperature tăng từ 0.0 → 1.5, phản hồi thường trở nên đa dạng, sáng tạo và ít ổn định hơn. Ở 0.0 câu trả lời thường nhất quán, còn ở 1.0–1.5 mô hình có xu hướng đưa ra cách diễn đạt hoặc sự thật khác nhau, đôi khi bất ngờ hơn.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *ý do là chatbot cần trả lời ổn định, chính xác và nhất quán, đặc biệt với các thông tin như giá sản phẩm, chính sách đổi trả, bảo hành và hướng dẫn sử dụng. Temperature thấp giúp giảm việc chatbot trả lời quá sáng tạo hoặc thay đổi câu trả lời giữa các lần hỏi. Tuy nhiên, không nên đặt bằng 0 trong mọi trường hợp vì một chút độ linh hoạt vẫn giúp câu trả lời tự nhiên hơn.*

    ### Câu 1.3 — Đánh đổi chi phí
    Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
    mỗi lần trung bình ~350 token đầu ra.

    **Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
    trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
    > *Với 10.000 người dùng × 3 lần/ngày × 350 token = 10,5 triệu token đầu ra/ngày. Theo giá API hiện tại, GPT-4o là $10/1 triệu token output, còn GPT-4o-mini là $0,60/1 triệu, nên GPT-4o đắt khoảng 16,7 lần cho phần output này.

GPT-4o xứng đáng khi cần chất lượng suy luận và xử lý các yêu cầu phức tạp, quan trọng. GPT-4o-mini nên dùng cho các tác vụ đơn giản, số lượng lớn như chatbot hỗ trợ khách hàng, phân loại hoặc tạo phản hồi ngắn để tiết kiệm chi phí.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Với system prompt là giáo viên tiểu học, câu trả lời thường ngắn gọn, dùng từ vựng đơn giản và các ví dụ gần gũi với trẻ 8 tuổi. Với system prompt là chuyên gia tài chính, câu trả lời thường dài và chuyên sâu hơn, sử dụng các thuật ngữ như blockchain, distributed ledger, consensus và cryptography. System prompt định hướng vai trò, cách diễn đạt, mức độ chuyên môn và phong cách trả lời của model. Vì vậy, cùng một câu hỏi nhưng system prompt khác nhau có thể tạo ra những phản hồi rất khác nhau.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Với một đoạn tiếng Việt khoảng 100 từ, cách ước lượng số từ / 0.75 cho khoảng 133 token, trong khi tiktoken có thể đếm khoảng 170–200 token tùy nội dung, tức chênh lệch khoảng 28–50%. Nguyên nhân là tokenizer không đếm theo từ mà chia văn bản thành các token nhỏ hơn; tiếng Việt có nhiều từ có dấu và cách biểu diễn Unicode khiến một từ có thể bị tách thành nhiều token hơn. Vì vậy, công thức số từ / 0.75 chỉ là ước lượng, còn tiktoken cho số token thực tế mà API sử dụng.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng nhất khi người dùng cần phản hồi ngay lập tức, đặc biệt với chatbot hoặc câu trả lời dài, vì nội dung được hiển thị từng phần giúp giảm cảm giác phải chờ đợi. Ngược lại, non-streaming phù hợp khi cần nhận toàn bộ kết quả rồi mới xử lý tiếp, chẳng hạn như các tác vụ backend, lưu dữ liệu hoặc khi câu trả lời ngắn và thời gian chờ không đáng kể.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *So với delay cố định 1 giây, exponential backoff tăng thời gian chờ sau mỗi lần retry, ví dụ 1 → 2 → 4 → 8 giây, giúp API có thêm thời gian phục hồi và giảm số lượng request gửi lại trong thời gian ngắn. Nếu hàng nghìn client cùng dùng delay cố định giống nhau, chúng có thể retry đồng thời sau mỗi 1 giây, tạo ra các đợt request lớn liên tục (retry storm), khiến API càng quá tải và khó phục hồi hơn.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Bạn là một trợ lý học tập Python và AI dành cho người mới bắt đầu.
Hãy giải thích rõ ràng, dễ hiểu bằng tiếng Việt, sử dụng ví dụ đơn giản
và ưu tiên code cơ bản. Khi người học mắc lỗi, hãy chỉ ra nguyên nhân
và hướng dẫn cách sửa từng bước. Trả lời ngắn gọn nhưng đầy đủ, tránh
sử dụng thuật ngữ quá chuyên môn nếu chưa giải thích trước.

Tôi chọn "dễ hiểu bằng tiếng Việt" để người học mới có thể tiếp thu kiến thức nhanh hơn và tránh khó khăn do thuật ngữ tiếng Anh. Tôi chọn "chỉ ra nguyên nhân và hướng dẫn cách sửa từng bước" vì khi học lập trình, hiểu tại sao code sai quan trọng hơn việc chỉ đưa ra đáp án đúng.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất của trợ lý hiện tại là history chỉ lưu được khoảng 3 lượt hội thoại, nên khi cuộc trò chuyện dài, model có thể không nhớ được những thông tin quan trọng ở các lượt trước. Một cải thiện cụ thể là tăng khả năng lưu trữ lịch sử hội thoại bằng cách lưu các message vào database hoặc file và lấy lại những đoạn liên quan khi có request mới. Có thể kết hợp thêm cơ chế tóm tắt history cũ để giảm số token gửi lên API nhưng vẫn giữ được ngữ cảnh quan trọng.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
