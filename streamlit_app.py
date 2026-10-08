import os
import streamlit as st

# Streamlit Cloud secrets -> environment variables (used by stock_agent.get_llm)
try:
    for k in ("LLM_PROVIDER", "LLM_MODEL", "GROQ_API_KEY", "GOOGLE_API_KEY",
              "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if k in st.secrets and not os.getenv(k):
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass  # running locally without secrets.toml: use normal env vars

from stock_agent import run_agent_safe, format_report, plot_price

st.set_page_config(page_title="AI Stock Market Assistant", page_icon="📈", layout="wide")
st.title("📈 AI Stock Market Assistant")
st.caption("Multi-agent (LangGraph) · RAG over financial news · simulated BUY / SELL / HOLD")

c1, c2, c3 = st.columns([3, 1, 1])
ticker = c1.text_input("Ticker (NSE stocks need .NS, e.g. RELIANCE.NS, TCS.NS; US: AAPL)", "RELIANCE.NS")
period = c2.selectbox("Period", ["6mo", "1y", "2y"], index=1)
go = c3.button("Analyze", type="primary", use_container_width=True)

if go:
    with st.spinner("Agents are working: data -> news (RAG) -> analysis -> recommendation..."):
        try:
            s = run_agent_safe(ticker.strip(), period)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.markdown(format_report(s))
    if not s.get("error"):
        st.pyplot(plot_price(s["ticker"], period))
        t1, t2, t3 = st.tabs(["News summary (RAG)", "Analysis", "Technical summary"])
        t1.markdown(s["news_summary"])
        t2.markdown(s["analysis"])
        t3.markdown(s["technical_summary"])
        st.caption(f"News fetch rounds: {s.get('retry_count')} | analysis confidence: {s.get('confidence'):.0%}")
