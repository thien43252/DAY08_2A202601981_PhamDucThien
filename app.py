"""
RAG Chatbot — University Services UI
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_utils import load_project_env

load_project_env()

# Thêm project root vào sys.path để import các task từ src/

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title="University Services RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown(
    """
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    
    /* Header card styling */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .header-title {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Source item styling inside expander */
    .source-box {
        background-color: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #38bdf8;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 12px;
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 600;
        color: #f1f5f9;
        font-size: 0.9rem;
        margin-bottom: 6px;
    }
    .source-badge {
        background-color: #0284c7;
        color: #ffffff;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .score-badge {
        background-color: #059669;
        color: #ffffff;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .source-content {
        color: #cbd5e1;
        font-size: 0.85rem;
        font-family: monospace;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px 12px;
        border-radius: 4px;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* Sidebar button enhancement */
    .stButton button {
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
    }
    .stButton button:hover {
        border-color: #38bdf8;
        color: #38bdf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# MOCK DATA GENERATOR (Dùng khi test UI hoặc chưa kết nối backend)
# =============================================================================

def get_mock_response(query: str, top_k: int = 5) -> dict:
    """
    Giả lập câu trả lời từ RAG Pipeline (Task 9 + Task 10) để test UI.
    """
    time.sleep(0.8)  # Giả lập latency sinh câu trả lời
    query_lower = query.lower()

    if "học phí" in query_lower or "rmit" in query_lower or "tiền" in query_lower:
        answer = (
            "Dựa trên quy định tài chính của nhà trường:\n\n"
            "1. **Học phí trung bình**: Khoảng 300,000,000 VNĐ - 320,000,000 VNĐ/năm tùy theo ngành học [Quy_Dinh_Hoc_Phi_2026.pdf].\n"
            "2. **Hình thức nộp**: Sinh viên có thể nộp theo từng học kỳ (3 học kỳ/năm) qua cổng thanh toán myRMIT hoặc chuyển khoản ngân hàng [Huong_Dan_Nop_Hoc_Phi.md].\n"
            "3. **Chính sách đóng chậm**: Cần nộp đơn xin gia hạn trước deadline đóng học phí ít nhất 7 ngày làm việc [Quy_Dinh_Hoc_Phi_2026.pdf].\n\n"
            "*(Lưu ý: Học phí có thể điều chỉnh không quá 10% mỗi năm theo quy định Bộ GD&ĐT)*"
        )
        all_sources = [
            {
                "content": "Quy định học phí năm học 2025-2026: Học phí chương trình cử nhân dao động từ 100,000,000 VNĐ đến 110,000,000 VNĐ mỗi học kỳ. Hạn nộp học phí là tuần thứ 2 của học kỳ.",
                "metadata": {"source": "Quy_Dinh_Hoc_Phi_2026.pdf", "type": "legal_doc", "page": 3},
                "score": 0.9412,
            },
            {
                "content": "Hướng dẫn thanh toán trực tuyến: Sinh viên truy cập myRMIT -> Financial Services -> Pay Tuition Fee. Chấp nhận thẻ ATM nội địa, Visa/Mastercard và VNPay.",
                "metadata": {"source": "Huong_Dan_Nop_Hoc_Phi.md", "type": "guide", "section": "Thanh toan"},
                "score": 0.8850,
            },
            {
                "content": "Đơn xin gia hạn đóng học phí: Sinh viên có hoàn cảnh khó khăn nộp đơn xin gia hạn tối đa 30 ngày tại Phòng Dịch vụ Sinh viên (Student Service Center).",
                "metadata": {"source": "Chinh_Sach_Mien_Giam.pdf", "type": "legal_doc", "page": 7},
                "score": 0.8120,
            },
            {
                "content": "Mức phạt nộp chậm học phí: Trường hợp quá hạn không có lý do chính đáng sẽ chịu mức phí phạt 0.05%/ngày trên tổng số tiền nộp chậm.",
                "metadata": {"source": "Quy_Dinh_Tai_Chinh.pdf", "type": "legal_doc", "page": 12},
                "score": 0.7654,
            },
            {
                "content": "Thông báo điều chỉnh học phí định kỳ: Mức tăng học phí hằng năm không vượt quá 10% và sẽ thông báo trước 3 tháng cho sinh viên.",
                "metadata": {"source": "Thong_Bao_Hoc_Phi_2026.md", "type": "news", "date": "2026-01-15"},
                "score": 0.7210,
            },
        ]

    elif "thư viện" in query_lower or "phòng học" in query_lower or "đặt" in query_lower:
        answer = (
            "Hướng dẫn dịch vụ thư viện và đặt phòng học nhóm:\n\n"
            "1. **Quy định đặt phòng**: Sinh viên đặt phòng học nhóm trực tuyến qua hệ thống Library Resource Booking trước tối đa 7 ngày [Nội_Quy_Thu_Vien.pdf].\n"
            "2. **Thời gian sử dụng**: Mỗi nhóm sinh viên (tối thiểu 3 người) được đặt tối đa 2 giờ/ngày [Quy_Dinh_Su_Dung_Phong.md].\n"
            "3. **Mượn trả sách**: Thẻ sinh viên tích hợp thẻ thư viện. Hạn mượn giáo trình là 14 ngày, sách tham khảo là 7 ngày [Noi_Quy_Thu_Vien.pdf]."
        )
        all_sources = [
            {
                "content": "Nội quy Thư viện trung tâm: Phòng học nhóm dành cho nhóm từ 3 đến 8 sinh viên. Đặt phòng trực tuyến tại website library.university.edu.vn.",
                "metadata": {"source": "Noi_Quy_Thu_Vien.pdf", "type": "legal_doc", "page": 5},
                "score": 0.9230,
            },
            {
                "content": "Quy định sử dụng phòng chức năng: Sinh viên cần check-in phòng học nhóm trong vòng 15 phút đầu tiên, nếu không phòng sẽ tự động hủy booking.",
                "metadata": {"source": "Quy_Dinh_Su_Dung_Phong.md", "type": "guide", "section": "Check-in"},
                "score": 0.8640,
            },
            {
                "content": "Thời gian mở cửa thư viện: Thứ 2 - Thứ 6: 7:30 - 21:00 | Thứ 7 - Chủ nhật: 8:00 - 17:00. Mở cửa 24/7 trong 2 tuần thi học kỳ.",
                "metadata": {"source": "Lich_Hoat_Dong_Thu_Vien.md", "type": "news", "date": "2026-02-01"},
                "score": 0.7980,
            },
            {
                "content": "Mượn tài liệu số và cơ sở dữ liệu điện tử: Sinh viên đăng nhập tài khoản portal để truy cập IEEE Xplore, ScienceDirect và SpringerLink.",
                "metadata": {"source": "Huong_Dan_Thu_Vien_So.pdf", "type": "guide", "page": 2},
                "score": 0.7412,
            },
            {
                "content": "Xử phạt quá hạn sách: Phí phạt mượn quá hạn 5,000 VNĐ/cuốn/ngày. Quá 30 ngày sẽ tính là làm mất sách.",
                "metadata": {"source": "Quy_Dinh_Xu_Phat.pdf", "type": "legal_doc", "page": 9},
                "score": 0.6950,
            },
        ]

    elif "học bổng" in query_lower or "academic" in query_lower:
        answer = (
            "Thông tin về Học bổng Academic Achievement & Khuyến khích học tập:\n\n"
            "1. **Điều kiện ứng tuyển**: Điểm GPA học kỳ gần nhất ≥ 3.6/4.0, không nợ môn, và điểm rèn luyện đạt loại Xuất sắc [Chinh_Sach_Hoc_Bong_2026.pdf].\n"
            "2. **Giá trị học bổng**: Mức 100% học phí (GPA ≥ 3.8) hoặc 50% học phí (GPA từ 3.6 - 3.79) [Chinh_Sach_Hoc_Bong_2026.pdf].\n"
            "3. **Hồ sơ cần nộp**: Bảng điểm chính thức, bài luận cá nhân (500 từ), và các chứng nhận hoạt động ngoại khóa [Huong_Dan_Nop_Ho_So_Hoc_Bong.md]."
        )
        all_sources = [
            {
                "content": "Chính sách Học bổng Academic Achievement 2026: Xét thưởng hằng năm cho 5% sinh viên có thành tích xuất sắc nhất toàn khóa.",
                "metadata": {"source": "Chinh_Sach_Hoc_Bong_2026.pdf", "type": "legal_doc", "page": 1},
                "score": 0.9520,
            },
            {
                "content": "Hướng dẫn nộp hồ sơ xin học bổng: Nộp trực tuyến qua cổng Student Portal trước 17:00 ngày 15/10 hằng năm.",
                "metadata": {"source": "Huong_Dan_Nop_Ho_So_Hoc_Bong.md", "type": "guide", "section": "Ho so"},
                "score": 0.8910,
            },
            {
                "content": "Quy chế điểm rèn luyện: Điểm rèn luyện từ 90 điểm trở lên xếp loại Xuất sắc. Sinh viên vi phạm kỷ luật không được xét học bổng.",
                "metadata": {"source": "Quy_Che_Ren_Luyen.pdf", "type": "legal_doc", "page": 4},
                "score": 0.8250,
            },
        ]

    else:
        answer = (
            f"Dựa trên các tài liệu chính sách và dịch vụ đại học sẵn có:\n\n"
            f"Về câu hỏi **'{query}'**, hệ thống đã tra cứu các nguồn tài liệu liên quan trong cơ sở dữ liệu [Dai_Hoc_Handbook_2026.pdf].\n"
            f"Nếu bạn cần thêm thông tin chi tiết về thủ tục hoặc quy trình cụ thể, vui lòng liên hệ **Phòng Dịch vụ Sinh viên (Student Service Center)** qua email `services@university.edu.vn`."
        )
        all_sources = [
            {
                "content": f"Trích đoạn quy định liên quan đến '{query}': Sinh viên làm theo hướng dẫn tại Sổ tay sinh viên năm học 2025-2026.",
                "metadata": {"source": "Dai_Hoc_Handbook_2026.pdf", "type": "legal_doc", "page": 15},
                "score": 0.8540,
            },
            {
                "content": "Thông tin liên hệ các phòng ban: Phòng Dịch vụ Sinh viên (Ô 102, Tòa nhà Admin). Hotline: (028) 3776 1300.",
                "metadata": {"source": "Directory_Lien_He.md", "type": "guide", "section": "Lien he"},
                "score": 0.7810,
            },
        ]

    return {
        "answer": answer,
        "sources": all_sources[:top_k],
        "retrieval_source": "mock_hybrid_retrieval",
    }


# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("🎓 University RAG")
    st.caption("Trợ lý AI tra cứu chính sách & dịch vụ đại học")

    st.divider()

    # Mode Selector (Cho phép switch giữa Mock Data để demo UI & Real Pipeline)
    st.subheader("⚙️ Cấu hình Pipeline")
    run_mode = st.radio(
        "Chế độ chạy:",
        ["🧪 Mock Data (Demo UI)", "🚀 Real RAG (Task 9 & 10)"],
        index=0,
        help="Chế độ Mock Data giúp thử nghiệm giao diện nhanh mà không cần API key.",
    )

    # Slider top_k (Theo yêu cầu đề bài - Bước 1)
    top_k = st.slider(
        "Số chunks retrieval (top_k)",
        min_value=1,
        max_value=10,
        value=5,
        help="Số lượng văn bản liên quan nhất được trích xuất để đưa vào LLM Context",
    )

    temperature = st.slider(
        "Độ sáng tạo (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="RAG khuyến nghị temperature thấp (0.0 - 0.3) để tránh bịa đặt thông tin.",
    )

    st.divider()

    # Gợi ý câu hỏi (Theo yêu cầu đề bài - Bước 1)
    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Điều kiện xin học bổng Academic Achievement?",
        "Dịch vụ hỗ trợ chỗ ở cho sinh viên như thế nào?",
        "Cách đăng ký học phần qua myRMIT?",
    ]

    for idx, s in enumerate(suggestions):
        if st.button(s, use_container_width=True, key=f"sug_btn_{idx}"):
            st.session_state["pending_query"] = s

    st.divider()

    # Nút xóa lịch sử trò chuyện
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.divider()
    st.markdown("### 🏗️ Kiến trúc RAG")
    st.caption(
        "**Hybrid Retrieval** (Semantic + BM25) ➔ **RRF Rerank** ➔ **PageIndex Fallback** ➔ **LLM Generation với Citations**"
    )

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Xin chào! Tôi là Trợ lý AI hỗ trợ giải đáp chính sách và dịch vụ đại học (Học phí, Học bổng, Thư viện, Ký túc xá). Bạn cần hỗ trợ thông tin gì hôm nay?",
            "sources": [],
        }
    ]

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# =============================================================================
# MAIN CHAT AREA
# =============================================================================

# Banner Header đẹp mắt
st.markdown(
    """
    <div class="header-card">
        <div class="header-title">
            🎓 University Services RAG Chatbot
        </div>
        <div class="header-subtitle">
            Hệ thống hỏi đáp thông minh kết hợp Hybrid Retrieval & Trích dẫn nguồn tài liệu (Citations)
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Hiển thị lịch sử chat (Theo yêu cầu đề bài - Bước 2: st.chat_message)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Hiển thị nguồn tham khảo kèm score (Theo yêu cầu đề bài - Bước 3: st.expander)
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            sources = msg["sources"]
            with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)", expanded=False):
                for i, src in enumerate(sources, 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source") or meta.get("file_name") or "Tài liệu chính sách"
                    doc_type = meta.get("type", "document")
                    score = src.get("score", 0.0)
                    content = src.get("content", "")

                    st.markdown(
                        f"""
                        <div class="source-box">
                            <div class="source-header">
                                <span><b>[{i}] {source_name}</b></span>
                                <div>
                                    <span class="source-badge">{doc_type}</span>
                                    <span class="score-badge">Score: {score:.4f}</span>
                                </div>
                            </div>
                            <div class="source-content">{content}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/dịch vụ đại học...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # 1. Thêm & hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Xử lý sinh câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang truy xuất tài liệu và tổng hợp câu trả lời..."):
            if "Mock Data" in run_mode:
                # Chạy bằng Mock Data
                response = get_mock_response(query, top_k=top_k)
                answer = response["answer"]
                sources = response.get("sources", [])
            else:
                # Chạy bằng Real RAG Pipeline (Task 9 & Task 10)
                try:
                    # TODO (Học viên / Team): Ghép nối chính thức với Task 10
                    # from src.task10_generation import generate_with_citation
                    # response = generate_with_citation(query, top_k=top_k)
                    
                    from src.task10_generation import generate_with_citation
                    
                    # Truyền history nếu muốn hỗ trợ follow-up conversation memory
                    response = generate_with_citation(query, top_k=top_k)
                    answer = response.get("answer", "Chưa có phản hồi từ mô hình.")
                    sources = response.get("sources", [])

                except NotImplementedError:
                    st.warning("⚠️ Task 10 chưa được implement đầy đủ. Đang chuyển sang Mock Data để hiển thị UI...")
                    response = get_mock_response(query, top_k=top_k)
                    answer = response["answer"]
                    sources = response.get("sources", [])
                except Exception as e:
                    st.error(f"❌ Lỗi khi thực thi RAG Pipeline: {e}")
                    st.info("💡 Bạn có thể chuyển sang chế độ '🧪 Mock Data (Demo UI)' ở sidebar để test giao diện.")
                    answer = f"**Lỗi thực thi RAG Pipeline:** {e}\n\n*Vui lòng kiểm tra lại API Key (.env) hoặc code tại src/task10_generation.py.*"
                    sources = []

            # Display answer
            st.markdown(answer)

            # Display citations expander immediately for new response
            if sources:
                with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)", expanded=False):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        source_name = meta.get("source") or meta.get("file_name") or "Tài liệu chính sách"
                        doc_type = meta.get("type", "document")
                        score = src.get("score", 0.0)
                        content = src.get("content", "")

                        st.markdown(
                            f"""
                            <div class="source-box">
                                <div class="source-header">
                                    <span><b>[{i}] {source_name}</b></span>
                                    <div>
                                        <span class="source-badge">{doc_type}</span>
                                        <span class="score-badge">Score: {score:.4f}</span>
                                    </div>
                                </div>
                                <div class="source-content">{content}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
