"""
Streamlit dashboard for RAG eval results.

Usage:
    streamlit run src/eval/dashboard.py
"""
import json
import os

import pandas as pd
import streamlit as st

RESULTS_PATH = "data/eval_set/results.json"

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")
st.title("RAG Evaluation Dashboard")

if not os.path.exists(RESULTS_PATH):
    st.warning(
        f"No results found at `{RESULTS_PATH}`. "
        "Run `python -m src.eval.run_eval` first."
    )
    st.stop()

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

df = pd.DataFrame([{
    "id": r["id"],
    "question": r["question"],
    "recall_at_k": r["recall_at_k"],
    "reciprocal_rank": r["reciprocal_rank"],
    "faithfulness": r["faithfulness"]["score"],
    "answer_relevancy": r["answer_relevancy"],
} for r in results])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Recall@k", f"{df['recall_at_k'].mean():.2f}")
col2.metric("MRR", f"{df['reciprocal_rank'].mean():.2f}")
col3.metric("Avg Faithfulness", f"{df['faithfulness'].mean():.2f}")
col4.metric("Avg Answer Relevancy", f"{df['answer_relevancy'].mean():.2f}")

st.subheader("Per-query results")
st.dataframe(df, use_container_width=True)

st.subheader("Flagged low-faithfulness answers")
low_faith = [r for r in results if r["faithfulness"]["score"] < 0.7]
if not low_faith:
    st.success("No low-faithfulness answers below threshold 0.7.")
else:
    for r in low_faith:
        with st.expander(f"⚠️ {r['id']}: {r['question']}"):
            st.write(f"**Answer:** {r['generated_answer']}")
            st.write("**Claim breakdown:**")
            for c in r["faithfulness"]["claims"]:
                icon = "✅" if c["entailed"] else "❌"
                st.write(f"{icon} {c['claim']} (entailment: {c['entailment_score']})")
