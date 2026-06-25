import streamlit as st
from src.rag_engine import ComplaintRAGEngine

st.set_page_config(page_title="CrediTrust AI Engine", layout="wide", page_icon="🛡️")

st.title("🛡️ CrediTrust Financial — Intelligent Complaint Analysis System")
st.caption("RAG-Powered AI Engine to Extract Actionable Strategic Insights from Customer Feedback")

@st.cache_resource
def load_engine():
    return ComplaintRAGEngine()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Failed loading backend processing files: {str(e)}")
    st.info("Ensure you have run the embedding pipeline and your FAISS database assets are inside vector_store/")
    st.stop()

st.sidebar.header("Query Ingestion Filters")
product_options = [
    "All Products",
    "Credit card or prepaid card",
    "Checking or savings account",
    "Money transfer, virtual currency, or money service",
    "Payday loan, title loan, or personal consumer loan"
]
selected_product = st.sidebar.selectbox("Filter by Product Class Category", product_options)
product_filter = None if selected_product == "All Products" else selected_product

st.sidebar.markdown("""
### Analytical System KPIs
* **Proactive Ingestion**: Parses historical text strings into real-time analytical metrics.
* **Turnkey Intelligence**: Minimizes dependency on engineering teams for data retrieval.
""")

user_query = st.text_input(
    "Enter your operational query:", 
    placeholder="e.g., Why are customers unhappy with credit cards or experiencing money transfer delays?"
)

if user_query:
    with st.spinner("Searching cluster nodes, extracting contexts, and parsing response frames..."):
        result = engine.generate_answer(question=user_query, product_filter=product_filter)
        
    st.subheader("📝 Synthesized Insights & Actionable Feedback Summary")
    st.markdown(result["answer"])
    
    with st.expander("🔍 View Raw Retrieved Context Clustered Evidence Chunks"):
        st.text(result["context"])






