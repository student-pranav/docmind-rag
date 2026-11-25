import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="DocMind RAG Assistant", page_icon="📄", layout="centered")
st.title("📄 DocMind RAG Assistant")
st.caption("Upload a PDF, ask questions, and get grounded answers with page citations.")

with st.sidebar:
    st.header("Document Controls")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None and st.button("Upload & Ingest PDF"):
        with st.spinner("Uploading and indexing PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            up_res = requests.post(f"{API_URL}/upload", files=files, timeout=60)
            if up_res.status_code == 200:
                ing_res = requests.post(f"{API_URL}/ingest", timeout=120)
                if ing_res.status_code == 200:
                    st.success(f"Indexed {ing_res.json()['chunks_processed']} chunks from {uploaded_file.name}!")
                else:
                    st.error(f"Ingest error: {ing_res.text}")
            else:
                st.error(f"Upload error: {up_res.text}")

    st.divider()
    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=3)
    if st.button("Re-ingest current PDF"):
        with st.spinner("Ingesting PDF into ChromaDB..."):
            res = requests.post(f"{API_URL}/ingest", timeout=120)
            if res.status_code == 200:
                st.success(f"Ingested {res.json()['chunks_processed']} chunks!")
            else:
                st.error(res.text)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("View Citations"):
                for c in msg["citations"]:
                    st.markdown(f"**{c['source']} (Page {c['page']})**")
                    st.caption(c["snippet"])

if prompt := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            res = requests.post(
                f"{API_URL}/chat",
                json={"question": prompt, "top_k": top_k},
                timeout=60,
            )
            if res.status_code == 200:
                data = res.json()
                answer = data["answer"]
                citations = data.get("citations", [])
                st.markdown(answer)
                if citations:
                    with st.expander("View Citations"):
                        for c in citations:
                            st.markdown(f"**{c['source']} (Page {c['page']})**")
                            st.caption(c["snippet"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "citations": citations}
                )
            else:
                st.error(f"API Error ({res.status_code}): {res.text}")
