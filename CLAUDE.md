# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Enterprise knowledge base Q&A agent built on LangGraph 1.x + LangChain 1.x. Supports local document retrieval (txt/pdf/docx), online web search fallback via Tavily, and summary export. Uses Ollama for local LLM inference and embeddings.

## Commands

```bash
# Install dependencies (use python -m pip, NOT bare pip — Python/pip paths differ on this machine)
python -m pip install -r requirements.txt

# Run the agent (interactive console)
python main.py

# Run any single module
python -m agent.graph_builder
```

## Architecture

```
main.py                  # Console entry point, invokes agent_app
settings.py              # All config: paths, model params, chunk sizes, retrieval params
prompts.py               # System prompt + judgment prompt template (decoupled from code)
log_config.py            # Global logger ("KB-Agent"), writes to ./logs/run.log + stdout
utils.py                 # format_retrieve_docs helper

document/
  loader.py              # Multi-format loader: .txt (TextLoader), .pdf (PyPDFLoader), .docx (Docx2txtLoader)
  splitter.py            # Two splitters: abstract_splitter (80-char, for FAISS coarse) + detail_splitter (500-char, for fine retrieval)
  vector_store.py        # FAISS IndexFlatL2 + index↔fulltext mappings. Auto-builds on first run, caches to disk (first_faiss.index + index_mapping.json)

retriever.py             # Two-tier hybrid RAG: (1) FAISS abstract search → (2) Chroma vector + BM25 keyword ensemble on detail chunks

tools/agent_tools.py     # Three @tool functions: search_knowledge_base, search_online (Tavily), save_summary_to_txt

agent/
  state.py               # AgentState TypedDict with Annotated[Sequence[BaseMessage], operator.add] for message accumulation
  nodes.py               # LangGraph nodes: agent_think_node (LLM decides tool), tool_execute_node (runs tool), judge_check_node (validates result)
  routes.py              # Conditional edges: tool_route_func (has tool_calls? → execute : END), judge_route_func (ng? → retry think : END)
  graph_builder.py       # StateGraph assembly: think → (tool? → execute → judge → (ng? → think : END) : END). Exports agent_app
```

## Key design decisions

- **LangChain version**: LangChain **1.x** (not 0.x). Core APIs live in `langchain_core`, not `langchain` directly — use `from langchain_core.globals import set_llm_cache`, `from langchain_core.caches import InMemoryCache`, etc.
- **Two-tier RAG**: FAISS searches document abstracts first (cheap coarse filter), then Chroma+BM25 EnsembleRetriever searches detail chunks within matched documents. `retriever.py` is the only module the agent tools call — they never touch `document/` directly.
- **LLM + Embeddings**: Both run locally via Ollama. LLM = `qwen2:7b`, embeddings = `all-minilm`. Configured in `settings.py`.
- **Judgment loop**: After every tool execution, `judge_check_node` validates result relevance. If it returns "ng", the graph loops back to `agent_think_node` for retry. This prevents hallucinated or irrelevant answers.
- **FAISS cache**: `document/vector_store.py` auto-scans the project root for supported files on first run and builds the FAISS index. Subsequent runs load from disk (`first_faiss.index` + `index_mapping.json`). Delete both files to force a rebuild when documents change.
- **`.env` for secrets**: `TAVILY_API_KEY` is loaded from `.env` via `python-dotenv`. The file is `.gitignore`'d.
