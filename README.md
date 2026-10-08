# AI Stock Market Assistant (LangChain + LangGraph)

Multi-agent stock assistant: price data + technical indicators, RAG over financial news, LLM reasoning, simulated BUY / SELL / HOLD.

## Run locally (VS Code)
```bash
pip install -r requirements.txt
# requirements.txt already includes Groq, Gemini, OpenAI and Anthropic packages
export LLM_PROVIDER=groq            # groq | gemini | openai | anthropic
export GROQ_API_KEY=your_key        # or GOOGLE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
export LLM_MODEL=openai/gpt-oss-120b
python app.py
```
(Windows PowerShell: `$env:GOOGLE_API_KEY="your_key"`)

## Deploy on Hugging Face Spaces
1. New Space -> SDK: Gradio.
2. Upload `app.py`, `stock_agent.py`, `requirements.txt`, `README.md`.
3. Settings -> Variables and secrets: add `LLM_PROVIDER` (groq), `LLM_MODEL` (openai/gpt-oss-20b) and `GROQ_API_KEY` as Secrets.

## Workflow
`data_node -> news_node (RAG) -> analyst_node -> recommend_node`, with conditional edges:
bad ticker -> `fallback_node`; low confidence -> loop back to `news_node` (max 3 fetches).

Disclaimer: simulated output for education, not financial advice.

## Deploy on Streamlit Community Cloud (free)
1. Push these files to a public GitHub repo (`streamlit_app.py`, `stock_agent.py`, `requirements.txt`, `README.md`, notebook).
2. share.streamlit.io -> New app -> pick repo, branch `main`, main file `streamlit_app.py`.
3. Advanced settings -> Secrets:
```
LLM_PROVIDER = "groq"
LLM_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY = "your_key"
```
