# Lab 20 Exit Ticket: Multi-Agent Research System

**Student Name / MSSV**: Quoc Anh (2A202601080)  
**Topic**: Multi-Agent Research System (Supervisor, Researcher, Analyst, Writer)

---

## 1. Case nào NÊN dùng multi-agent? Vì sao?

### Trường hợp khuyến nghị:
- **Nghiên cứu kỹ thuật chuyên sâu & đa chiều** (Complex Technical Research): ví dụ khảo sát GraphRAG, tối ưu hóa agentic workflows, so sánh kiến trúc microservices.
- **Tổng hợp chứng cứ đa nguồn & đối soát mâu thuẫn** (Multi-hop Evidence Synthesis & Contradiction Resolution): các bài toán yêu cầu trích xuất dữ liệu từ nhiều tài liệu, so sánh đối chiếu quan điểm và đánh giá độ tin cậy của nguồn.
- **Báo cáo kiểm toán & tuân thủ quy định** (Compliance & Factuality Auditing): nơi mà mọi tuyên bố (claims) đều phải được kiểm chứng và trích dẫn trực tiếp nguồn tham khảo.

### Lý do (Dựa trên số liệu thực nghiệm từ `reports/benchmark_report.md`):
1. **Chất lượng vượt trội & Độ phủ trích dẫn cao**:
   - Multi-Agent đạt điểm chất lượng **8.8 - 9.1 / 10** so với **5.0 - 6.0 / 10** của Single-Agent.
   - Đạt độ phủ trích dẫn chứng cứ **60% - 69%** với các thẻ `[Source 1]`, `[Source 2]` trỏ trực tiếp về nguồn tài liệu, trong khi Single-Agent đạt **0%** (hoàn toàn dựa vào internal weights dễ gây ảo giác).
2. **Cô lập Bounded Context Window**:
   - Tách bạch `Researcher` (chuyên tìm kiếm & lọc facts) $\to$ `Analyst` (chuyên phân tích trade-offs) $\to$ `Writer` (chuyên tổng hợp văn bản) giúp context của mỗi bước luôn ngắn gọn, tập trung, loại bỏ hiện tượng context dilution.
3. **Tính minh bạch và Khả năng kiểm tra (Observability & Traceability)**:
   - Dễ dàng kiểm tra trace từng bước trung gian qua `route_history` và `state.agent_results` để biết lỗi phát sinh từ khâu thu thập tài liệu hay khâu phân tích.

---

## 2. Case nào KHÔNG NÊN dùng multi-agent? Vì sao?

### Trường hợp khuyến nghị KHÔNG dùng:
- **Tra cứu thông tin đơn giản / Single-hop QA**: Trả lời câu hỏi định nghĩa trực tiếp, tra cứu thời tiết, tóm tắt email ngắn, dịch thuật văn bản.
- **Tác vụ yêu cầu phản hồi tức thì (Real-time Latency Critical)**: Chatbot chăm sóc khách hàng trực tuyến, voice assistant, autocompletion.
- **Hệ thống bị giới hạn ngân sách nghiêm ngặt (Cost-Constrained Production)**.

### Lý do (Dựa trên số liệu thực nghiệm từ `reports/benchmark_report.md`):
1. **Độ trễ cao hơn gấp 2-3 lần**:
   - Single-Agent hoàn thành chỉ trong **9.69s - 15.05s** (1 turn duy nhất).
   - Multi-Agent mất **30.19s - 31.34s** (qua 4 turns phối hợp Supervisor $\leftrightarrow$ Workers).
2. **Chi phí token đắt gấp 4 lần**:
   - Single-Agent tiêu tốn trung bình **~800 tokens** (~`$0.00047`).
   - Multi-Agent tiêu tốn trung bình **~6,300 tokens** (~`$0.00201`) do phải chuyền shared state và gọi LLM ở nhiều giai đoạn.
3. **Chi phí điều phối (Coordination Overhead) & Rủi ro vòng lặp**:
   - Với câu hỏi đơn giản, một LLM call duy nhất đã đủ năng lực giải quyết hoàn hảo. Việc xây dựng đồ thị điều phối tạo ra overhead không cần thiết và tiềm ẩn rủi ro lỗi handoff giữa các agents.
