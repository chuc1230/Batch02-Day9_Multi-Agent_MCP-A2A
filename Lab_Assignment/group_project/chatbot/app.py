# group_project/chatbot/app.py
import streamlit as st
import sys
from pathlib import Path

st.set_page_config(page_title="RAG Chatbot - Pháp luật Ma Tuý (Agentic)", page_icon="⚖️", layout="wide")

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src.agents import SupervisorAgent
except Exception as e:
    st.error(f"❌ Không thể import cấu trúc Agent từ thư mục src: {str(e)}")
    st.stop()

st.title("⚖️ Agentic Chatbot Tư Vấn Pháp Luật Phòng Chống Ma Tuý")
st.markdown("### Kiến trúc Multi-Agent Pattern (Supervisor-Workers) với Trực quan hóa quy trình ngầm")
st.divider()

if "supervisor" not in st.session_state:
    st.session_state.supervisor = SupervisorAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "workflow_steps" in message:
            with st.expander("🔍 Xem lại dấu vết tư duy của hệ thống Agent cho câu hỏi này"):
                for step in message["workflow_steps"]:
                    st.markdown(f"**{step['title']}**")
                    st.code(step["log"], language="text")

# Form tương tác nhập liệu
if prompt := st.chat_input("Nhập câu hỏi của bạn (Ví dụ: Mức phạt tàng trữ ma túy, hoặc vụ án diễn viên Hữu Tín)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # ---- ĐOẠN HIỂN THỊ LUỒNG CHẠY NGẦM CỦA WORKER & TOOL ----
        with st.status("🕵️ Supervisor Agent đang phân rã và điều phối tác vụ...", expanded=True) as status:
            history = st.session_state.messages[:-1]
            
            # Gọi Agent thực thi xử lý
            result = st.session_state.supervisor.route_and_execute(query=prompt, chat_history=history)
            
            # Quét qua từng bước quy trình ngầm của Agent để in lên màn hình
            for step in result.get("workflow_steps", []):
                st.write(step["title"])
                st.caption(f"_{step['log'].replace('\n', ' | ')}_")
            
            status.update(label="✅ Toàn bộ các Worker đã hoàn thành xuất sắc nhiệm vụ!", state="complete", expanded=False)
        # --------------------------------------------------------

        # Hiển thị đáp án cuối cùng
        answer = result.get("answer")
        sources = result.get("sources", [])
        retrieval_source = result.get("retrieval_source")

        st.markdown(answer)
        
        # Hiển thị nguồn trích dẫn dữ liệu sạch (Sources)
        if sources:
            with st.expander(f"📚 Xem chi tiết tài liệu trích dẫn ngữ cảnh ({len(sources)} sources | {retrieval_source})"):
                for i, doc in enumerate(sources, 1):
                    meta = doc.get("metadata", {})
                    source_name = meta.get("source", f"Tài liệu {i}")
                    doc_type = meta.get("doc_type", "unknown")
                    st.markdown(f"**[{i}] Nguồn: {source_name}** | Phân loại dữ liệu: `{doc_type.upper()}`")
                    st.info(doc.get("content"))

        # Lưu vào lịch sử chat kèm theo vết tư duy để xem lại
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "workflow_steps": result.get("workflow_steps", [])
        })