import streamlit as st
from agents.orchestrator import Orchestrator

st.set_page_config(page_title="Portfolio War Room", layout="centered")


@st.cache_resource
def get_orchestrator():
    """Built once per server session, not once per click — same reasoning
    as the module-level embeddings cache in retriever.py."""
    return Orchestrator()


st.title("📊 Portfolio War Room")
st.caption("⚠️ Not financial advice — a decision-support demo project.")

query = st.text_input(
    "Ask a question about your portfolio:",
    placeholder="e.g. should I rebalance out of tech this week?",
)

if st.button("Analyze") and query:
    orchestrator = get_orchestrator()
    with st.spinner("Running analysis..."):
        result = orchestrator.run(query)

    if result["plan"] == "quick_lookup":
        st.subheader("Result")
        st.json(result["result"])

    else:
        st.subheader("Bottom line")
        st.write(result["plain_summary"])

        st.subheader("Simulation outlook")
        sim = result["simulation_summary"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Low estimate (p5)", f"{sim['p5']:.1f}")
        col2.metric("Median outcome", f"{sim['p50']:.1f}", delta=f"{sim['p50'] - 100:.1f} vs today")
        col3.metric("High estimate (p95)", f"{sim['p95']:.1f}")
        st.caption(f"Chance of ending below today's value: {sim['probability_of_loss'] * 100:.0f}%")

        st.subheader("Debate breakdown")
        for arg in result["scored_arguments"]:
            with st.expander(f"{arg['stance']} — score: {arg['score']}/10"):
                st.write(arg["argument"])
                st.caption(f"Judge's note: {arg['justification']}")