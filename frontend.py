
import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:8000"

# Page configuration
st.set_page_config(
    page_title="MirAI Student Policy Advisor",
    page_icon="🎓",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🎓 MirAI Student Policy Advisor</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Ask questions about MirAI policies</div>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("📄 Policy Document")
    st.write("Upload the MirAI policy PDF to get started.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if st.button("Upload and Process PDF", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please select a PDF file first.")
        else:
            try:
                with st.spinner("Processing policy PDF..."):
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    }

                    response = requests.post(
                        f"{API_URL}/ingest",
                        files=files,
                        timeout=180
                    )

                if response.ok:
                    data = response.json()
                    st.success(data.get("message", "PDF processed!"))
                    st.write(f"Pages: {data.get('pages', 'N/A')}")
                    st.write(f"Chunks: {data.get('chunks', 'N/A')}")
                    st.session_state.pdf_ready = True
                else:
                    st.error(f"Upload failed: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Cannot connect to backend: {e}")

    st.divider()
    st.caption("Powered by FastAPI, LangChain and Gemini")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

# Welcome message
if not st.session_state.messages:
    st.info(
        "👋 Welcome! Ask me anything about attendance, "
        "examinations, clubs, or other MirAI policies."
    )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input("Ask a question about MirAI policies...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching policy and generating answer..."):
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question},
                    timeout=180
                )

            if response.ok:
                data = response.json()
                answer = data.get("answer", "No answer returned.")
                st.markdown(answer)
            else:
                answer = (
                    "I couldn't answer that question. "
                    "Please upload and process the policy PDF "
                    "from the sidebar first.\n\n"
                    f"Details: {response.text}"
                )
                st.error(answer)

        except requests.exceptions.RequestException as e:
            answer = f"Cannot connect to the backend. Details: {e}"
            st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
