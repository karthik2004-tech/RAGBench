"""
Simple Streamlit dashboard for RAGBench evaluation results.

Usage:
    streamlit run src/eval/dashboard.py
"""

import json
import os

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_PATH = "data/eval_set/results.json"

st.set_page_config(
    page_title="RAGBench Dashboard",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("🧠 RAGBench")
st.subheader("RAG Evaluation Dashboard")

st.write(
    "Evaluate retrieval quality, faithfulness, and answer relevancy "
    "of your Retrieval-Augmented Generation system."
)

st.divider()


# ============================================================
# CHECK RESULTS FILE
# ============================================================

if not os.path.exists(RESULTS_PATH):

    st.warning(
        f"No results found at `{RESULTS_PATH}`."
    )

    st.info(
        "Run the evaluation first:"
    )

    st.code(
        "python -m src.eval.run_eval "
        "--eval-set data/eval_set/eval_set.json"
    )

    st.stop()


# ============================================================
# LOAD RESULTS
# ============================================================

with open(
    RESULTS_PATH,
    "r",
    encoding="utf-8",
) as f:

    results = json.load(f)


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    [
        {
            "ID": r["id"],
            "Question": r["question"],
            "Recall@K": r["recall_at_k"],
            "Reciprocal Rank": r["reciprocal_rank"],
            "Faithfulness": r["faithfulness"]["score"],
            "Answer Relevancy": r["answer_relevancy"],
        }
        for r in results
    ]
)


# ============================================================
# OVERALL METRICS
# ============================================================

st.header("📊 Overall Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Recall@K",
    f"{df['Recall@K'].mean():.2f}",
)

col2.metric(
    "MRR",
    f"{df['Reciprocal Rank'].mean():.2f}",
)

col3.metric(
    "Average Faithfulness",
    f"{df['Faithfulness'].mean():.2f}",
)

col4.metric(
    "Answer Relevancy",
    f"{df['Answer Relevancy'].mean():.2f}",
)


st.divider()


# ============================================================
# PERFORMANCE CHART
# ============================================================

st.header("📈 Metric Comparison")

chart_df = pd.DataFrame(
    {
        "Metric": [
            "Recall@K",
            "MRR",
            "Faithfulness",
            "Answer Relevancy",
        ],
        "Score": [
            df["Recall@K"].mean(),
            df["Reciprocal Rank"].mean(),
            df["Faithfulness"].mean(),
            df["Answer Relevancy"].mean(),
        ],
    }
)

st.bar_chart(
    chart_df.set_index("Metric")
)


st.divider()


# ============================================================
# PER QUERY RESULTS
# ============================================================

st.header("🔎 Per-Query Results")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# ============================================================
# QUERY DETAILS
# ============================================================

st.header("🧪 Query Details")

for r in results:

    with st.expander(
        f"{r['id']} — {r['question']}"
    ):

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Recall@K",
            f"{r['recall_at_k']:.2f}",
        )

        col2.metric(
            "Faithfulness",
            f"{r['faithfulness']['score']:.2f}",
        )

        col3.metric(
            "Answer Relevancy",
            f"{r['answer_relevancy']:.2f}",
        )

        st.markdown("### Generated Answer")

        st.write(
            r["generated_answer"]
        )

        st.markdown("### Faithfulness Analysis")

        claims = r["faithfulness"]["claims"]

        if not claims:

            st.info(
                "No claims detected."
            )

        else:

            for claim in claims:

                if claim["entailed"]:

                    st.success(
                        f"✅ {claim['claim']} "
                        f"(score: {claim['entailment_score']:.3f})"
                    )

                else:

                    st.error(
                        f"❌ {claim['claim']} "
                        f"(score: {claim['entailment_score']:.3f})"
                    )


st.divider()


# ============================================================
# LOW FAITHFULNESS
# ============================================================

st.header("⚠️ Low Faithfulness Answers")

low_faith = [
    r
    for r in results
    if r["faithfulness"]["score"] < 0.7
]


if not low_faith:

    st.success(
        "No answers have a faithfulness score below 0.70."
    )

else:

    for r in low_faith:

        st.warning(
            f"{r['id']} — "
            f"Faithfulness: "
            f"{r['faithfulness']['score']:.2f}"
        )

        st.write(
            r["question"]
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 RAGBench")

    st.write(
        "RAG Evaluation & Optimization Framework"
    )

    st.divider()

    st.subheader("System")

    st.write("**Embedding:**")
    st.code("all-MiniLM-L6-v2")

    st.write("**Vector Database:**")
    st.code("FAISS")

    st.write("**LLM:**")
    st.code("Qwen 2.5 1.5B")

    st.write("**LLM Runtime:**")
    st.code("Ollama")

    st.write("**Evaluation:**")
    st.code("NLI + Embeddings")

    st.divider()

    st.subheader("Dataset")

    st.write(
        f"Evaluation queries: **{len(results)}**"
    )

    st.write(
        f"Low-faithfulness queries: "
        f"**{len(low_faith)}**"
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "RAGBench — Measure. Understand. Improve."
)