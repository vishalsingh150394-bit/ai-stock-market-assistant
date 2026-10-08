# %%
import os
import re
import json
import html
from typing import TypedDict, List, Literal
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END

DISCLAIMER = ("Simulated output for educational purposes only. "
              "This is NOT financial advice.")


def get_llm(temperature: float = 0.2):
    """Return a LangChain chat model based on env vars LLM_PROVIDER / LLM_MODEL."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL", "")
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-3.8-flash",
                                      temperature=temperature)      # GOOGLE_API_KEY
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or "llama-3.3-70b-versatile",
                        temperature=temperature)                    # GROQ_API_KEY
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini",
                          temperature=temperature)                  # OPENAI_API_KEY
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or "claude-haiku-4-5-20251001",
                             temperature=temperature)               # ANTHROPIC_API_KEY
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


_EMB = None
def get_embeddings():
    """Free, local embeddings (no API key needed)."""
    global _EMB
    if _EMB is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _EMB = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _EMB

# %% [markdown]
# ## 2. Data tools (LangChain `@tool`)
# Numbers (RSI, SMA, MACD...) are computed in **Python**, not by the LLM. The LLM only interprets them.

# %%
def fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period, auto_adjust=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


def _r(x, n=2):
    return round(float(x), n) if pd.notna(x) else None


@tool
def get_price_snapshot(ticker: str, period: str = "1y") -> dict:
    """Get latest price, returns, SMA20/50, RSI14, MACD, volatility and 52-week range for a stock ticker (e.g. TCS.NS, AAPL)."""
    try:
        df = fetch_history(ticker, period)
    except Exception as e:
        return {"error": f"Could not fetch data: {e}"}
    if df is None or df.empty or len(df) < 30:
        return {"error": f"Not enough price data for '{ticker}'. Check the ticker (NSE stocks need .NS)."}
    df = add_indicators(df)
    last = df.iloc[-1]
    close = float(last["Close"])

    def ret(n):
        return _r((close / float(df["Close"].iloc[-n - 1]) - 1) * 100) if len(df) > n else None

    vol = df["Close"].pct_change().rolling(30).std().iloc[-1] * np.sqrt(252) * 100
    snap = {
        "ticker": ticker,
        "last_close": _r(close),
        "return_1m_pct": ret(21),
        "return_3m_pct": ret(63),
        "return_6m_pct": ret(126),
        "sma20": _r(last["SMA20"]),
        "sma50": _r(last["SMA50"]),
        "rsi14": _r(last["RSI14"]),
        "macd": _r(last["MACD"], 3),
        "macd_signal": _r(last["MACD_SIGNAL"], 3),
        "annualised_volatility_pct": _r(vol),
        "high_52w": _r(df["Close"].tail(252).max()),
        "low_52w": _r(df["Close"].tail(252).min()),
        "as_of": str(df.index[-1].date()),
    }
    signals = []
    if snap["sma50"]:
        signals.append("Price above SMA50 (uptrend)" if close > snap["sma50"] else "Price below SMA50 (downtrend)")
    if snap["rsi14"] is not None:
        r = snap["rsi14"]
        signals.append("RSI overbought (>70)" if r > 70 else "RSI oversold (<30)" if r < 30 else "RSI neutral")
    if snap["macd"] is not None and snap["macd_signal"] is not None:
        signals.append("MACD above signal (bullish momentum)" if snap["macd"] > snap["macd_signal"]
                       else "MACD below signal (bearish momentum)")
    snap["rule_based_signals"] = signals
    return snap


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(text or ""))).strip()


def _yf_news(ticker: str, n: int = 10) -> list:
    """Best-effort Yahoo Finance news (format changes often, so wrapped in try/except)."""
    out = []
    try:
        for item in (yf.Ticker(ticker).news or [])[:n]:
            c = item.get("content", item)
            title = c.get("title", "")
            if title:
                out.append({"title": title,
                            "summary": _clean(c.get("summary") or c.get("description") or ""),
                            "published": c.get("pubDate", "") or "",
                            "link": ""})
    except Exception:
        pass
    return out


@tool
def fetch_news(query: str, ticker: str = "", max_items: int = 15) -> list:
    """Fetch recent financial news articles for a company from Google News RSS (+ Yahoo Finance if available)."""
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    items = [{"title": _clean(e.get("title", "")),
              "summary": _clean(e.get("summary", "")),
              "published": e.get("published", ""),
              "link": e.get("link", "")} for e in feed.entries[:max_items]]
    if ticker:
        items += _yf_news(ticker)
    seen, unique = set(), []
    for a in items:
        k = a["title"].lower()
        if k and k not in seen:
            seen.add(k)
            unique.append(a)
    return unique

# %% [markdown]
# ## 3. RAG pipeline for news
# articles -> chunks -> embeddings -> vector store -> retriever -> prompt -> LLM (LCEL chain)

# %%
def build_retriever(articles: list, k: int = 5):
    docs = [Document(page_content=f"{a['title']}. {a['summary']}".strip(),
                     metadata={"published": a.get("published", ""), "link": a.get("link", "")})
            for a in articles]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    store = InMemoryVectorStore.from_documents(chunks, get_embeddings())
    return store.as_retriever(search_kwargs={"k": k})


def format_docs(docs) -> str:
    return "\n\n".join(f"[{d.metadata.get('published', '')[:16]}] {d.page_content}" for d in docs)


NEWS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial news analyst. Use ONLY the news context provided. "
     "If the context is thin or irrelevant, say so clearly. Do not invent facts."),
    ("human",
     "{question}\n\nNews context:\n{context}\n\n"
     "Give: (1) overall sentiment: Positive / Negative / Neutral / Mixed, "
     "(2) 3-5 key developments, (3) key risks or concerns mentioned."),
])


def build_rag_chain(retriever):
    return ({"context": retriever | format_docs, "question": RunnablePassthrough()}
            | NEWS_PROMPT | get_llm() | StrOutputParser())

# %% [markdown]
# ## 4. Prompt templates and structured outputs (Pydantic)

# %%
class Analysis(BaseModel):
    summary: str = Field(description="4-6 sentence integrated analysis of technicals and news")
    bullish_factors: List[str] = Field(description="Key positives")
    bearish_factors: List[str] = Field(description="Key negatives")
    confidence: float = Field(description="Confidence in the analysis from 0.0 to 1.0")


class Recommendation(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(description="0.0 to 1.0")
    justification: str = Field(description="3-5 sentences citing specific numbers and news")
    key_risks: List[str]
    time_horizon: str = Field(description="e.g. 'short term (1-3 months)'")


TECH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a technical analyst. Interpret the metrics given. Do not make up numbers."),
    ("human", "Ticker: {ticker}\nMetrics (JSON):\n{metrics}\n\n"
              "Write a concise technical summary (trend, momentum, volatility, support/resistance from 52w range)."),
])

ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior equity research analyst. Combine the technical view and the news view into one analysis. "
     "Be evidence-based. Set confidence LOWER than 0.5 if the news is sparse/irrelevant, "
     "if technical and news signals strongly conflict, or if data is limited."),
    ("human", "Company: {company} ({ticker})\n\nTECHNICAL SUMMARY:\n{technical}\n\n"
              "RULE-BASED SIGNALS: {signals}\n\nNEWS SUMMARY:\n{news}"),
])

RECO_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a portfolio strategist producing a SIMULATED recommendation for an educational project. "
     "Choose BUY, SELL or HOLD. Choose HOLD when evidence is mixed. "
     "Use ONLY numbers that appear in METRICS or the analysis; never invent figures. "
     "Reference concrete numbers (RSI, SMA, returns) and news themes in the justification."),
    ("human", "Company: {company} ({ticker})\n\nANALYSIS:\n{analysis}\n\n"
              "BULLISH: {bull}\nBEARISH: {bear}\nAnalysis confidence: {confidence}\n\nMETRICS (computed in Python):\n{metrics}"),
])

# %% [markdown]
# ## 5. LangGraph: state, nodes, edges (with conditional edges)
# Flow: `data_node -> news_node (RAG) -> analyst_node -> recommend_node`
# - **Conditional edge 1** after `data_node`: bad ticker / no data -> `fallback_node`
# - **Conditional edge 2** after `analyst_node`: low confidence -> go back to `news_node` with a different query (max 2 retries), else `recommend_node`

# %%
class AgentState(TypedDict, total=False):
    ticker: str
    period: str
    company: str
    price_data: dict
    technical_summary: str
    news_articles: list
    news_summary: str
    analysis: str
    bullish: list
    bearish: list
    confidence: float
    retry_count: int
    error: str
    recommendation: dict


QUERY_VARIANTS = ["{c} stock news",
                  "{c} share price latest news",
                  "{c} quarterly results earnings outlook"]


def _company_name(ticker: str) -> str:
    try:
        name = yf.Ticker(ticker).info.get("longName")
        if name:
            return name
    except Exception:
        pass
    return ticker.split(".")[0]


def data_node(state: AgentState) -> dict:
    ticker = state["ticker"].strip().upper()
    snap = get_price_snapshot.invoke({"ticker": ticker, "period": state.get("period", "1y")})
    if "error" in snap:
        return {"ticker": ticker, "price_data": snap, "error": snap["error"]}
    company = _company_name(ticker)
    summary = (TECH_PROMPT | get_llm() | StrOutputParser()).invoke(
        {"ticker": ticker, "metrics": json.dumps(snap, indent=2)})
    return {"ticker": ticker, "company": company, "price_data": snap,
            "technical_summary": summary, "error": "", "retry_count": 0, "news_articles": []}


def news_node(state: AgentState) -> dict:
    attempt = state.get("retry_count", 0)
    company = state.get("company", state["ticker"])
    query = QUERY_VARIANTS[min(attempt, len(QUERY_VARIANTS) - 1)].format(c=company)
    new = fetch_news.invoke({"query": query, "ticker": state["ticker"]})
    seen = {a["title"].lower() for a in state.get("news_articles", [])}
    articles = state.get("news_articles", []) + [a for a in new if a["title"].lower() not in seen]
    if not articles:
        return {"news_articles": [], "news_summary": "No recent news found.", "retry_count": attempt + 1}
    retriever = build_retriever(articles)
    chain = build_rag_chain(retriever)
    summary = chain.invoke(f"What is the recent news sentiment and key developments for {company} ({state['ticker']})?")
    return {"news_articles": articles, "news_summary": summary, "retry_count": attempt + 1}


def analyst_node(state: AgentState) -> dict:
    llm = get_llm().with_structured_output(Analysis)
    res = (ANALYST_PROMPT | llm).invoke({
        "company": state.get("company", ""), "ticker": state["ticker"],
        "technical": state["technical_summary"],
        "signals": "; ".join(state["price_data"].get("rule_based_signals", [])),
        "news": state["news_summary"]})
    conf = float(max(0.0, min(1.0, res.confidence)))
    return {"analysis": res.summary, "bullish": res.bullish_factors,
            "bearish": res.bearish_factors, "confidence": conf}


def recommend_node(state: AgentState) -> dict:
    llm = get_llm().with_structured_output(Recommendation)
    res = (RECO_PROMPT | llm).invoke({
        "company": state.get("company", ""), "ticker": state["ticker"],
        "analysis": state["analysis"],
        "bull": "; ".join(state["bullish"]), "bear": "; ".join(state["bearish"]),
        "confidence": state["confidence"],
        "metrics": json.dumps(state["price_data"], indent=2)})
    return {"recommendation": res.model_dump()}


def fallback_node(state: AgentState) -> dict:
    return {"recommendation": {
        "action": "HOLD", "confidence": 0.0,
        "justification": f"Could not complete analysis: {state.get('error', 'insufficient data')}",
        "key_risks": ["Insufficient data"], "time_horizon": "n/a"}}


# ---- conditional edge functions ----
def route_after_data(state: AgentState) -> str:
    return "fallback_node" if state.get("error") else "news_node"


def route_after_analysis(state: AgentState) -> str:
    if state.get("confidence", 0) < 0.5 and state.get("retry_count", 0) < 3:
        return "news_node"          # loop back: fetch more / different news
    return "recommend_node"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("data_node", data_node)
    g.add_node("news_node", news_node)
    g.add_node("analyst_node", analyst_node)
    g.add_node("recommend_node", recommend_node)
    g.add_node("fallback_node", fallback_node)

    g.add_edge(START, "data_node")
    g.add_conditional_edges("data_node", route_after_data,
                            {"news_node": "news_node", "fallback_node": "fallback_node"})
    g.add_edge("news_node", "analyst_node")
    g.add_conditional_edges("analyst_node", route_after_analysis,
                            {"news_node": "news_node", "recommend_node": "recommend_node"})
    g.add_edge("recommend_node", END)
    g.add_edge("fallback_node", END)
    return g.compile()


agent_app = build_graph()

# %% [markdown]
# ## 6. Helper functions: run, report, chart

# %%
def run_agent(ticker: str, period: str = "1y", verbose: bool = False) -> dict:
    state = {"ticker": ticker, "period": period}
    if verbose:
        for step in agent_app.stream(state):
            node = list(step.keys())[0]
            print(f"✔ finished: {node}")
            state.update(step[node])
        return state
    return agent_app.invoke(state)


def run_agent_safe(ticker: str, period: str = "1y", tries: int = 4, wait: int = 20, verbose: bool = False) -> dict:
    """run_agent with automatic retry for temporary API errors (503 high demand, 429 rate limit)."""
    import time
    for i in range(1, tries + 1):
        try:
            return run_agent(ticker, period, verbose=verbose)
        except Exception as e:
            msg = str(e)
            temporary = any(x in msg for x in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "rate_limit", "overloaded"))
            if temporary and i < tries:
                print(f"⚠ Temporary API error (attempt {i}/{tries}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def format_report(s: dict) -> str:
    r = s["recommendation"]
    icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[r["action"]]
    return (f"## {icon} {r['action']}  (confidence {r['confidence']:.0%})\n"
            f"**{s.get('company', s['ticker'])} ({s['ticker']})**  |  Horizon: {r['time_horizon']}\n\n"
            f"**Justification:** {r['justification']}\n\n"
            f"**Key risks:** " + "; ".join(r["key_risks"]) + f"\n\n_{DISCLAIMER}_")


def plot_price(ticker: str, period: str = "1y"):
    df = add_indicators(fetch_history(ticker, period))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df.index, df["Close"], label="Close")
    ax.plot(df.index, df["SMA20"], label="SMA20", alpha=0.8)
    ax.plot(df.index, df["SMA50"], label="SMA50", alpha=0.8)
    ax.set_title(f"{ticker} - price with moving averages")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
