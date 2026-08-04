# Ai Fnance Portfolio WarRoom

An agentic, multi-agent portfolio decision-support system. Given a question
like *"should I rebalance out of tech this week?"*, it stages a structured,
evidence-grounded debate between specialist agents, backs that debate with
real quantitative analysis of your actual holdings, stress-tests the
resulting decision with simulation, and explains the outcome in plain
English.

This is **not a chatbot**. It's a decision-support pipeline that outputs a
structured report — a debate transcript, a grounded rationale, simulation
results, and a plain-language summary with charts.

> ⚠️ **Not financial advice.** This is a portfolio project demonstrating
> agentic AI, RAG, and data analysis techniques. It is not a substitute for
> professional financial advice.

## Why this project exists

Built to demonstrate hands-on application of concepts from:
- IBM RAG and Agentic AI Professional Certificate
- AWS Generative AI and AI Agents with Amazon Bedrock Professional Certificate
- IBM Generative AI Engineering Professional Certificate

The goal was to go beyond a simple RAG chatbot and build a system that
actually exhibits **agentic** behavior — dynamic planning, self-correction,
and adaptive tool use — not just a fixed pipeline of LLM calls.

## Architecture

```
User query
    ↓
Orchestrator agent          — plans which agents to run for this query
    ↓
Data analyst agent (pandas) — calls quantitative tools as needed
    ↓
   ┌────────────┬────────────┬──────────────────┐
Bull agent   Bear agent   Risk-parity agent      — each retrieves evidence
   └────────────┴────────────┴──────────────────┘   until confident (ReAct)
    ↓
Judge agent                 — weighs arguments by grounding quality,
    ↓                          can send weak arguments back for revision
Monte Carlo simulation      — stress-tests the decision quantitatively
    ↓
Plain-English explainer     — translates results into a simple summary
    ↓
Decision + memory log       — logged; read back by the orchestrator
                               on future runs
```

### Why it's agentic, not just multi-agent

| Feature | What it means here |
|---|---|
| **Dynamic planning** | The orchestrator decides which agents to invoke per query — a "what's my portfolio worth" question skips the debate entirely; a rebalancing question triggers the full pipeline. |
| **Adaptive tool use** | The data analyst agent chooses which analyses are relevant (e.g. only computes correlation if multiple holdings overlap) instead of always running a fixed script. |
| **ReAct retrieval loops** | Debate agents reason → retrieve → check sufficiency → retrieve again, rather than retrieving once and stopping. |
| **Self-correction** | The judge can reject a weak argument and send it back for revision (capped at ~2 rounds) before finalizing a decision. |
| **Persistent memory** | Past decisions and outcomes are logged and actually read by the orchestrator on future runs, not just stored for a dashboard. |

## Key components

- **Multi-agent debate** — bull, bear, and risk-parity agents argue distinct
  positions, each grounded in retrieved evidence rather than free opinion.
- **Grounded judging** — the judge weighs arguments by citation quality, not
  confidence, directly addressing the hallucination problem that's core to
  production RAG systems.
- **Quantitative backing (pandas)** — real portfolio metrics (returns,
  volatility, drawdown, correlation) feed the debate as evidence, not just
  narrative context.
- **Monte Carlo simulation** — the qualitative decision is checked against
  simulated future scenarios before being finalized.
- **Plain-English explainer** — technical output is translated into a
  jargon-free summary, making the tool usable by a non-expert.
- **MCP tool integration** — external capabilities (pandas analysis, market
  data, retrieval corpus) are exposed as MCP servers, keeping integrations
  swappable rather than hardwired.
- **Evaluation loop** — every decision is logged with its rationale so
  retrieval groundedness and outcome accuracy can be measured over time.

## Tech stack

- **Orchestration**: LangGraph (or CrewAI) — handles the conditional
  routing and revision loops the architecture needs
- **Retrieval**: Chroma / FAISS vector store over investing literature and
  historical case studies
- **Data analysis**: pandas, exposed via a custom MCP server
- **Simulation**: numpy-based Monte Carlo
- **Front end**: Streamlit or Gradio for the report view
- **LLM**: Claude / GPT-family models via API

## Project status

🚧 In development — architecture designed, implementation in progress.

## Roadmap

- [ ] Scaffold repo structure and agent interfaces
- [ ] Build pandas MCP server (returns, volatility, drawdown, correlation)
- [ ] Implement retrieval corpus + vector store
- [ ] Implement debate agents with ReAct retrieval loop
- [ ] Implement judge agent with revision loop
- [ ] Implement Monte Carlo simulation layer
- [ ] Implement plain-English explainer agent
- [ ] Implement orchestrator planning logic
- [ ] Implement decision/memory log + feedback into orchestrator
- [ ] Build report front end (Streamlit/Gradio)
- [ ] Add evaluation metrics (retrieval groundedness, outcome accuracy)

## Disclaimer

This tool is for educational and demonstration purposes only. It does not
constitute financial advice. Simulated results are based on historical and
statistical assumptions and are not guarantees of future performance.