import streamlit as st
import pandas as pd
import numpy as np
import faiss
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sentence_transformers import SentenceTransformer
import re
import requests
import json

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Conversational Data Analyst using RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Dark background ── */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #13151c !important;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 50%, #131824 100%);
    border: 1px solid #2a3050;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #63b3ed, #a78bfa, #f687b3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    color: #8892a4;
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: #13151c;
    border: 1px solid #1e2538;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #63b3ed, #a78bfa);
    border-radius: 0 0 12px 12px;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5a6478;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e8eaf0;
    line-height: 1;
}

/* ── Answer box ── */
.answer-box {
    background: linear-gradient(135deg, #131d2e 0%, #111827 100%);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #63b3ed;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
    font-size: 1.1rem;
    line-height: 1.7;
    color: #cbd5e1;
}
.answer-icon { font-size: 1.4rem; margin-right: 0.5rem; }

/* ── Section headers ── */
.section-header {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a5568;
    margin: 2rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2538;
}

/* ── Chat history ── */
.chat-user {
    background: #1a1f2e;
    border: 1px solid #2a3050;
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    margin-left: 15%;
    color: #a5b4fc;
    font-size: 0.95rem;
}
.chat-ai {
    background: #13151c;
    border: 1px solid #1e2538;
    border-radius: 12px 12px 12px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    margin-right: 15%;
    color: #94a3b8;
    font-size: 0.95rem;
}

/* ── Input styling ── */
.stTextInput input {
    background: #13151c !important;
    border: 1px solid #2a3050 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput input:focus {
    border-color: #63b3ed !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #13151c;
    border: 2px dashed #2a3050 !important;
    border-radius: 12px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2538 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Success / info / warning ── */
.stSuccess, .stInfo, .stWarning {
    border-radius: 10px !important;
}

/* ── RAG context expander ── */
.streamlit-expanderHeader {
    background: #13151c !important;
    border: 1px solid #1e2538 !important;
    border-radius: 8px !important;
    color: #5a6478 !important;
    font-size: 0.82rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD EMBEDDING MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models…")
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()


# ─────────────────────────────────────────────
#  RAG PIPELINE
# ─────────────────────────────────────────────
def build_row_documents(df: pd.DataFrame) -> list[str]:
    """Convert each row into a readable text document for RAG."""
    docs = []
    for _, row in df.iterrows():
        parts = [f"{col}: {val}" for col, val in row.items()]
        docs.append(" | ".join(parts))
    return docs


@st.cache_data(show_spinner=False)
def build_faiss_index(df_hash: str, _df: pd.DataFrame):
    """Build FAISS index from dataframe rows. Cached by df hash."""
    documents = build_row_documents(_df)
    vectors = np.array([
        embed_model.encode(doc) for doc in documents
    ]).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, documents


def retrieve_relevant_rows(question: str, index, documents: list[str],
                            df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    RAG retrieval: embed the question, search FAISS index,
    return the top-k most relevant rows as a DataFrame.
    """
    q_vec = embed_model.encode(question).astype("float32").reshape(1, -1)
    distances, indices = index.search(q_vec, min(top_k, len(documents)))
    retrieved_idx = [i for i in indices[0] if i < len(df)]
    return df.iloc[retrieved_idx].copy(), retrieved_idx


# ─────────────────────────────────────────────
#  RULE-BASED QUERY PARSER
#  (replaces tiny distilgpt2 — much more reliable)
# ─────────────────────────────────────────────
def detect_intent(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["summary", "describe", "overview", "about"]):
        return "summary"
    if any(w in q for w in ["highest", "most", "top", "best", "maximum", "max", "largest"]):
        return "max"
    if any(w in q for w in ["lowest", "least", "worst", "minimum", "min", "smallest"]):
        return "min"
    if any(w in q for w in ["total", "sum", "overall", "aggregate"]):
        return "sum"
    if any(w in q for w in ["average", "mean", "avg", "typical"]):
        return "avg"
    if any(w in q for w in ["count", "how many", "number of"]):
        return "count"
    if any(w in q for w in ["compare", "versus", "vs", "difference", "breakdown"]):
        return "compare"
    if any(w in q for w in ["trend", "over time", "by month", "by year", "by date"]):
        return "trend"
    if any(w in q for w in ["correlation", "relationship", "related", "affect"]):
        return "correlation"
    if any(w in q for w in ["distribution", "spread", "range", "histogram"]):
        return "distribution"
    return "general"


def find_best_column(question: str, df: pd.DataFrame, prefer_numeric: bool = True) -> str | None:
    """
    Find the most relevant VALUE column — ALWAYS numeric.
    Never returns a categorical/text column as the value column.
    """
    q_lower = question.lower()

    if prefer_numeric:
        # Start with pandas-detected numeric columns
        candidate_cols = df.select_dtypes(include="number").columns.tolist()
        # Also include columns that look numeric but were read as strings
        for col in df.columns:
            if col not in candidate_cols:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > len(df) * 0.5:
                    candidate_cols.append(col)
    else:
        candidate_cols = df.select_dtypes(include="object").columns.tolist()

    if not candidate_cols:
        return None

    # 1. Direct name match within numeric candidates only
    for col in candidate_cols:
        if col.lower() in q_lower:
            return col

    # 2. Word-level partial match within numeric candidates only
    for col in candidate_cols:
        for part in col.lower().replace("_", " ").split():
            if len(part) > 2 and part in q_lower:
                return col

    # 3. Fallback: first numeric column
    return candidate_cols[0]


def find_group_column(question: str, df: pd.DataFrame) -> str | None:
    """
    Find the GROUP BY column — ALWAYS categorical/object dtype.
    Never returns a numeric column as the group column.
    """
    q_lower = question.lower()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if not cat_cols:
        return None

    # 1. Direct name match
    for col in cat_cols:
        if col.lower() in q_lower:
            return col

    # 2. Word-level partial match
    for col in cat_cols:
        for part in col.lower().replace("_", " ").split():
            if len(part) > 2 and part in q_lower:
                return col

    # 3. Fallback: first categorical column
    return cat_cols[0]


# ─────────────────────────────────────────────
#  SAFE NUMBER FORMATTER
# ─────────────────────────────────────────────
def fmt(val) -> str:
    """Safely format a value as a number string, handling non-numeric types."""
    try:
        f = float(val)
        if f == int(f):
            return f"{int(f):,}"
        return f"{f:,.2f}"
    except (TypeError, ValueError):
        return str(val)


def to_numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Force-convert a column to numeric, coercing errors to NaN."""
    return pd.to_numeric(df[col], errors="coerce")




# ─────────────────────────────────────────────
#  CLAUDE API — REAL LLM GENERATION
# ─────────────────────────────────────────────
def call_claude(api_key: str, system_prompt: str, user_prompt: str) -> str:
    """
    Sends retrieved context + question to Claude API.
    This is the 'Augmented Generation' step of RAG.
    """
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 200,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=15,
        )
        if response.status_code == 200:
            return response.json()["content"][0]["text"].strip()
        elif response.status_code == 401:
            return "⚠️ Invalid API key. Please check your Anthropic API key in the sidebar."
        elif response.status_code == 429:
            return "⚠️ Rate limit hit. Please wait a moment and try again."
        else:
            return f"⚠️ API error {response.status_code}. Using pandas answer as fallback."
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Check your internet connection."
    except Exception as e:
        return f"⚠️ Could not reach Claude API: {str(e)}"


def build_rag_prompt(question: str, retrieved_df: pd.DataFrame,
                     pandas_answer: str, df: pd.DataFrame) -> tuple[str, str]:
    """
    Builds the system + user prompt by injecting:
    - Retrieved rows (RAG context)
    - Pre-computed pandas stats (for accuracy)
    - The user question
    """
    # Summarise full dataset stats for context
    num_cols = df.select_dtypes(include="number").columns.tolist()
    stats_lines = []
    for col in num_cols[:4]:  # limit to 4 cols to stay within token budget
        s = pd.to_numeric(df[col], errors="coerce")
        stats_lines.append(
            f"  {col}: total={fmt(s.sum())}, avg={fmt(s.mean())}, "
            f"max={fmt(s.max())}, min={fmt(s.min())}"
        )
    stats_text = "\n".join(stats_lines) if stats_lines else "No numeric columns."

    # Convert retrieved rows to readable text
    retrieved_text = retrieved_df.to_string(index=False) if not retrieved_df.empty else "No rows retrieved."

    system_prompt = (
        "You are a concise data analyst assistant. "
        "You are given dataset statistics, retrieved relevant rows, and a pre-computed answer. "
        "Give a SHORT, direct 1-2 sentence response using the data provided. "
        "Do not make up numbers — only use the figures given to you. "
        "Do not use markdown bold or bullet points."
    )

    user_prompt = f"""Question: {question}

Dataset stats:
{stats_text}

Retrieved relevant rows (RAG context):
{retrieved_text}

Pre-computed answer from data:
{pandas_answer}

Give a short, direct 1-2 sentence answer using the above data."""

    return system_prompt, user_prompt

# ─────────────────────────────────────────────
#  ANSWER GENERATION  (uses retrieved RAG rows)
# ─────────────────────────────────────────────
def generate_answer(question: str, df: pd.DataFrame,
                    retrieved_df: pd.DataFrame) -> tuple[str, str]:
    """
    Returns (answer_text, intent).
    Uses the FULL df for aggregation but acknowledges RAG context.
    """
    intent = detect_intent(question)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # ── SUMMARY ──────────────────────────────
    if intent == "summary":
        return (
            f"Your dataset has **{len(df):,} rows** and **{len(df.columns)} columns**.\n\n"
            f"📐 **Numeric columns** ({len(num_cols)}): {', '.join(num_cols) or '—'}\n\n"
            f"🏷️ **Categorical columns** ({len(cat_cols)}): {', '.join(cat_cols) or '—'}\n\n"
            f"🔍 **Missing values**: {df.isnull().sum().sum()} total across all columns.",
            intent,
        )

    val_col = find_best_column(question, df, prefer_numeric=True)
    grp_col = find_group_column(question, df)

    if not val_col or val_col not in df.columns:
        return "⚠️ I couldn't identify a relevant numeric column for your question. Try rephrasing.", intent

    # Force the value column to numeric to avoid 'f' format errors on strings
    series = to_numeric_col(df, val_col).dropna()
    if series.empty:
        return f"⚠️ Column **{val_col}** doesn't appear to contain numeric data.", intent

    # ── MAX ───────────────────────────────────
    if intent == "max":
        if grp_col and grp_col in df.columns and grp_col != val_col:
            grouped = df.groupby(grp_col)[val_col].apply(
                lambda x: pd.to_numeric(x, errors="coerce").sum()
            )
            winner = grouped.idxmax()
            return (
                f"🏆 **{winner}** has the highest **{val_col}** with a total of "
                f"**{fmt(grouped.max())}** (out of {len(grouped)} groups).",
                intent,
            )
        return f"📈 The maximum **{val_col}** is **{fmt(series.max())}**.", intent

    # ── MIN ───────────────────────────────────
    if intent == "min":
        if grp_col and grp_col in df.columns and grp_col != val_col:
            grouped = df.groupby(grp_col)[val_col].apply(
                lambda x: pd.to_numeric(x, errors="coerce").sum()
            )
            loser = grouped.idxmin()
            return (
                f"📉 **{loser}** has the lowest **{val_col}** with a total of "
                f"**{fmt(grouped.min())}** (out of {len(grouped)} groups).",
                intent,
            )
        return f"📉 The minimum **{val_col}** is **{fmt(series.min())}**.", intent

    # ── SUM ───────────────────────────────────
    if intent == "sum":
        total = series.sum()
        if grp_col and grp_col in df.columns and grp_col != val_col:
            grouped = df.groupby(grp_col)[val_col].apply(
                lambda x: pd.to_numeric(x, errors="coerce").sum()
            )
            return (
                f"➕ Total **{val_col}** is **{fmt(total)}**.\n\n"
                f"Top contributor: **{grouped.idxmax()}** ({fmt(grouped.max())})",
                intent,
            )
        return f"➕ Total **{val_col}** across all records: **{fmt(total)}**.", intent

    # ── AVG ───────────────────────────────────
    if intent == "avg":
        avg = series.mean()
        if grp_col and grp_col in df.columns and grp_col != val_col:
            grouped = df.groupby(grp_col)[val_col].apply(
                lambda x: pd.to_numeric(x, errors="coerce").mean()
            )
            return (
                f"📊 Overall average **{val_col}** is **{fmt(avg)}**.\n\n"
                f"Highest average: **{grouped.idxmax()}** ({fmt(grouped.max())}) | "
                f"Lowest: **{grouped.idxmin()}** ({fmt(grouped.min())})",
                intent,
            )
        return f"📊 Average **{val_col}**: **{fmt(avg)}** (median: {fmt(series.median())})", intent

    # ── COUNT ─────────────────────────────────
    if intent == "count":
        if grp_col and grp_col in df.columns:
            counts = df[grp_col].value_counts()
            return (
                f"🔢 Total records: **{len(df):,}**.\n\n"
                f"Most frequent **{grp_col}**: **{counts.index[0]}** ({counts.iloc[0]} times)",
                intent,
            )
        return f"🔢 Dataset contains **{len(df):,}** records.", intent

    # ── COMPARE / TREND / CORRELATION / DISTRIBUTION / GENERAL ──
    if intent in ("compare", "trend", "correlation", "distribution", "general"):
        if grp_col and val_col and grp_col in df.columns:
            grouped = df.groupby(grp_col)[val_col].apply(
                lambda x: pd.to_numeric(x, errors="coerce").sum()
            ).sort_values(ascending=False)
            top3 = grouped.head(3)
            lines = "\n".join([f"  • **{k}**: {fmt(v)}" for k, v in top3.items()])
            return (
                f"📊 **{val_col}** breakdown by **{grp_col}** (top 3):\n{lines}",
                intent,
            )
        return f"Here is the distribution of **{val_col}** in your dataset.", intent

    return "🤔 I couldn't interpret this question. Try asking about totals, averages, highest, lowest, or comparisons.", intent


# ─────────────────────────────────────────────
#  CHART GENERATOR  (always produces a chart)
# ─────────────────────────────────────────────
CHART_BG = "#0d0f14"
CHART_FG = "#e8eaf0"
GRID_CLR = "#1e2538"
PALETTE = ["#63b3ed", "#a78bfa", "#f687b3", "#68d391", "#fbd38d",
           "#fc8181", "#4fd1c5", "#f6ad55", "#76e4f7", "#b794f4"]


def style_ax(ax, title: str = ""):
    ax.set_facecolor(CHART_BG)
    ax.figure.set_facecolor(CHART_BG)
    ax.tick_params(colors=CHART_FG, labelsize=9)
    ax.xaxis.label.set_color(CHART_FG)
    ax.yaxis.label.set_color(CHART_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}" if x >= 1000 else f"{x:,.2f}"
    ))
    if title:
        ax.set_title(title, color=CHART_FG, fontsize=12, fontweight="bold",
                     pad=14, fontfamily="monospace")
    ax.grid(axis="y", color=GRID_CLR, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def generate_charts(df: pd.DataFrame, question: str, intent: str) -> list:
    """
    Always returns at least one relevant matplotlib figure.
    Returns a list of (fig, caption) tuples.
    """
    charts = []
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    val_col = find_best_column(question, df, prefer_numeric=True)
    grp_col = find_group_column(question, df)

    if not val_col:
        return charts

    # ── SUMMARY: overview grid ────────────────────────────────────────
    if intent == "summary":
        if len(num_cols) >= 2:
            fig, axes = plt.subplots(1, min(len(num_cols), 3),
                                     figsize=(14, 4),
                                     facecolor=CHART_BG)
            if len(num_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, num_cols[:3]):
                ax.hist(df[col].dropna(), bins=20, color=PALETTE[0], edgecolor=CHART_BG, alpha=0.9)
                style_ax(ax, f"{col} Distribution")
                ax.set_xlabel(col)
            plt.tight_layout()
            charts.append((fig, "📊 Numeric column distributions"))
        return charts

    # ── BAR: max / min / sum / compare ───────────────────────────────
    if intent in ("max", "min", "sum", "compare") and grp_col and grp_col in df.columns and grp_col != val_col:
        grouped = df.groupby(grp_col)[val_col].sum().sort_values(ascending=(intent == "min"))
        n = min(len(grouped), 15)
        grouped = grouped.head(n)

        fig, ax = plt.subplots(figsize=(10, 5), facecolor=CHART_BG)
        colors = [PALETTE[i % len(PALETTE)] for i in range(n)]
        bars = ax.bar(grouped.index, grouped.values, color=colors,
                      edgecolor=CHART_BG, linewidth=0.5)

        # Highlight winner
        if intent in ("max", "compare"):
            bars[0].set_edgecolor("#f6e05e")
            bars[0].set_linewidth(2)

        style_ax(ax, f"{val_col} by {grp_col}")
        ax.set_xlabel(grp_col)
        ax.set_ylabel(val_col)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        charts.append((fig, f"Bar chart — {val_col} grouped by {grp_col}"))

        # Also pie chart if ≤8 groups
        if n <= 8:
            fig2, ax2 = plt.subplots(figsize=(7, 5), facecolor=CHART_BG)
            wedges, texts, autotexts = ax2.pie(
                grouped.values,
                labels=grouped.index,
                colors=PALETTE[:n],
                autopct="%1.1f%%",
                startangle=140,
                pctdistance=0.78,
            )
            for t in texts + autotexts:
                t.set_color(CHART_FG)
                t.set_fontsize(9)
            ax2.set_facecolor(CHART_BG)
            ax2.figure.set_facecolor(CHART_BG)
            ax2.set_title(f"Share of {val_col}", color=CHART_FG,
                          fontsize=12, fontweight="bold", fontfamily="monospace")
            charts.append((fig2, "Pie chart — proportional share"))

        return charts

    # ── AVG: grouped mean bar ─────────────────────────────────────────
    if intent == "avg" and grp_col and grp_col in df.columns and grp_col != val_col:
        grouped = df.groupby(grp_col)[val_col].mean().sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=CHART_BG)
        ax.barh(grouped.index, grouped.values,
                color=PALETTE[2], edgecolor=CHART_BG, alpha=0.9)
        style_ax(ax, f"Average {val_col} by {grp_col}")
        ax.set_xlabel(f"Mean {val_col}")
        ax.invert_yaxis()
        plt.tight_layout()
        charts.append((fig, f"Horizontal bar — average {val_col} by {grp_col}"))
        return charts

    # ── COUNT ─────────────────────────────────────────────────────────
    if intent == "count" and cat_cols:
        col = find_best_column(question, df, prefer_numeric=False) or cat_cols[0]
        if col in df.columns:
            vc = df[col].value_counts().head(12)
            fig, ax = plt.subplots(figsize=(10, 5), facecolor=CHART_BG)
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(vc))]
            ax.bar(vc.index, vc.values, color=colors, edgecolor=CHART_BG)
            style_ax(ax, f"Count of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            charts.append((fig, f"Count per {col}"))
        return charts

    # ── DISTRIBUTION: histogram + boxplot ────────────────────────────
    if intent == "distribution":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor=CHART_BG)
        ax1.hist(df[val_col].dropna(), bins=25, color=PALETTE[0],
                 edgecolor=CHART_BG, alpha=0.9)
        style_ax(ax1, f"{val_col} Histogram")
        bp = ax2.boxplot(df[val_col].dropna(), patch_artist=True,
                         medianprops={"color": "#f6e05e", "linewidth": 2})
        bp["boxes"][0].set_facecolor(PALETTE[1])
        style_ax(ax2, f"{val_col} Boxplot")
        ax2.set_xticks([])
        plt.tight_layout()
        charts.append((fig, f"Distribution analysis of {val_col}"))
        return charts

    # ── CORRELATION: heatmap ──────────────────────────────────────────
    if intent == "correlation" and len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=CHART_BG)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                    linewidths=0.5, linecolor=CHART_BG,
                    ax=ax, cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 9, "color": CHART_FG})
        ax.set_facecolor(CHART_BG)
        ax.figure.set_facecolor(CHART_BG)
        ax.tick_params(colors=CHART_FG)
        ax.set_title("Correlation Matrix", color=CHART_FG, fontsize=12,
                     fontweight="bold", fontfamily="monospace")
        plt.tight_layout()
        charts.append((fig, "Correlation heatmap"))
        return charts

    # ── TREND: line chart (if a date-like or ordinal col exists) ───────
    date_cols = [c for c in df.columns
                 if any(k in c.lower() for k in ["date", "month", "year", "time", "day"])]
    if intent == "trend" and date_cols:
        dcol = date_cols[0]
        try:
            tmp = df.copy()
            tmp[dcol] = pd.to_datetime(tmp[dcol])
            tmp = tmp.sort_values(dcol)
            fig, ax = plt.subplots(figsize=(12, 5), facecolor=CHART_BG)
            ax.plot(tmp[dcol], tmp[val_col], color=PALETTE[0],
                    linewidth=2, alpha=0.9)
            ax.fill_between(tmp[dcol], tmp[val_col], alpha=0.1, color=PALETTE[0])
            style_ax(ax, f"{val_col} Over Time")
            ax.set_xlabel(dcol)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            charts.append((fig, f"Trend — {val_col} over {dcol}"))
            return charts
        except Exception:
            pass

    # ── FALLBACK: bar chart ───────────────────────────────────────────
    if grp_col and grp_col in df.columns and grp_col != val_col:
        grouped = df.groupby(grp_col)[val_col].sum().sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=CHART_BG)
        ax.bar(grouped.index,
               grouped.values,
               color=[PALETTE[i % len(PALETTE)] for i in range(len(grouped))],
               edgecolor=CHART_BG)
        style_ax(ax, f"{val_col} by {grp_col}")
        ax.set_xlabel(grp_col)
        ax.set_ylabel(val_col)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        charts.append((fig, f"{val_col} grouped by {grp_col}"))
    else:
        # Absolute fallback: histogram of the value column
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=CHART_BG)
        ax.hist(df[val_col].dropna(), bins=20, color=PALETTE[0],
                edgecolor=CHART_BG, alpha=0.9)
        style_ax(ax, f"{val_col} Distribution")
        ax.set_xlabel(val_col)
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        charts.append((fig, f"Distribution of {val_col}"))

    return charts


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.5rem;'>🧠</div>
        <div style='font-size:0.95rem; font-weight:700; color:#e8eaf0; line-height:1.4;'>Conversational Data Analyst</div>
        <div style='font-size:0.72rem; color:#4a5568; margin-top:4px; letter-spacing:0.05em;'>using RAG</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Sample questions")
    sample_qs = [
        "Which product has the highest sales?",
        "Show me the average revenue by region",
        "What is the total purchase amount?",
        "Which country generates the most revenue?",
        "Compare clicks across platforms",
        "Give me a summary of the dataset",
    ]
    for q in sample_qs:
        st.markdown(f"<div style='color:#4a6fa5; font-size:0.8rem; padding:3px 0;'>→ {q}</div>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔑 Claude API Key")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get a free key at console.anthropic.com",
        label_visibility="collapsed",
        key="api_key_input",
    )
    if api_key:
        st.markdown("""
        <div style='color:#68d391; font-size:0.78rem; padding: 4px 0;'>
        ✅ API key set — Claude will generate answers
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='color:#f6ad55; font-size:0.78rem; padding: 4px 0;'>
        ⚠️ No key — using pandas answers only<br>
        Get free key: console.anthropic.com
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='color:#2d3748; font-size:0.75rem; text-align:center;'>
    Powered by Claude + SentenceTransformer + FAISS
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">Conversational Data Analyst</p>
    <p class="hero-sub">RAG-powered analysis · Ask in plain English · Instant insights + charts</p>
</div>
""", unsafe_allow_html=True)

# ── Session state for chat history ──
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── File upload ──
uploaded = st.file_uploader("Upload your CSV dataset", type=["csv"],
                             help="Upload any CSV file to begin analysis")

if uploaded:
    df = pd.read_csv(uploaded)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # ── Dataset metric cards ──
    cards_html = f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Total Rows</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Columns</div>
            <div class="metric-value">{len(df.columns)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Numeric Cols</div>
            <div class="metric-value">{len(num_cols)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Categorical Cols</div>
            <div class="metric-value">{len(cat_cols)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Missing Values</div>
            <div class="metric-value">{df.isnull().sum().sum()}</div>
        </div>
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Preview ──
    st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    # ── Build FAISS index ──
    df_hash = str(pd.util.hash_pandas_object(df).sum())
    with st.spinner("🔍 Building RAG index…"):
        faiss_index, documents = build_faiss_index(df_hash, df)

    # ── Question input ──
    st.markdown('<div class="section-header">💬 Ask a Question</div>', unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_input(
            "question",
            placeholder="e.g. Which product has the highest sales?",
            label_visibility="collapsed",
            key="question_input",
        )
    with col_btn:
        ask_btn = st.button("Analyze", use_container_width=True)

    if ask_btn and question.strip():
        # ── STEP 1: RAG Retrieval — find relevant rows via FAISS ──
        retrieved_df, retrieved_idx = retrieve_relevant_rows(
            question, faiss_index, documents, df, top_k=5
        )

        # ── STEP 2: Pandas computes accurate stats ──
        pandas_answer, intent = generate_answer(question, df, retrieved_df)

        # ── STEP 3: Feed context + retrieved rows to Claude (real RAG generation) ──
        api_key = st.session_state.get("api_key_input", "")
        if api_key and api_key.strip().startswith("sk-ant"):
            with st.spinner("🤖 Claude is reading your data…"):
                system_prompt, user_prompt = build_rag_prompt(
                    question, retrieved_df, pandas_answer, df
                )
                llm_answer = call_claude(api_key.strip(), system_prompt, user_prompt)
            # If Claude returned an error, fall back to pandas answer
            final_answer = llm_answer if not llm_answer.startswith("⚠️") else pandas_answer
            answer_source = "Claude (RAG)" if not llm_answer.startswith("⚠️") else "Pandas (fallback)"
        else:
            final_answer = pandas_answer
            answer_source = "Pandas"

        # ── Save to history ──
        st.session_state.chat_history.append({
            "question": question,
            "answer": final_answer,
            "intent": intent,
            "source": answer_source,
        })

        # ── Display answer ──
        source_badge = (
            "<span style='background:#1a3a2a;color:#68d391;font-size:0.7rem;"
            "padding:2px 8px;border-radius:20px;font-weight:600;'>"
            f"✦ {answer_source}</span>"
        )
        st.markdown('<div class="section-header">🤖 AI Answer</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="answer-box">{source_badge}<br><br>{final_answer}</div>',
            unsafe_allow_html=True,
        )

        # ── Show RAG context (collapsible) ──
        with st.expander("🔍 RAG Context — Retrieved rows used for this answer", expanded=False):
            st.caption(f"Top {len(retrieved_df)} semantically relevant rows retrieved via FAISS:")
            st.dataframe(retrieved_df, use_container_width=True)

        # ── Charts ──
        st.markdown('<div class="section-header">📈 Visual Analysis</div>', unsafe_allow_html=True)
        charts = generate_charts(df, question, intent)

        if charts:
            if len(charts) == 1:
                fig, caption = charts[0]
                st.pyplot(fig, use_container_width=True)
                st.caption(caption)
            else:
                chart_cols = st.columns(len(charts))
                for (fig, caption), col in zip(charts, chart_cols):
                    with col:
                        st.pyplot(fig, use_container_width=True)
                        st.caption(caption)
        else:
            st.info("No chart applicable for this question type.")

        # ── Summary stats (if summary intent) ──
        if intent == "summary":
            st.markdown('<div class="section-header">📊 Detailed Statistics</div>',
                        unsafe_allow_html=True)
            st.dataframe(df.describe().T, use_container_width=True)

    # ── Chat History ──
    if st.session_state.chat_history:
        st.markdown('<div class="section-header">🗂️ Conversation History</div>',
                    unsafe_allow_html=True)
        for entry in reversed(st.session_state.chat_history[-6:]):
            st.markdown(
                f'<div class="chat-user">❓ {entry["question"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="chat-ai">🤖 {entry["answer"]}</div>',
                unsafe_allow_html=True,
            )

        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # ── Empty state ──
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem; color: #2d3748;'>
        <div style='font-size:4rem; margin-bottom:1rem;'>📂</div>
        <div style='font-size:1.2rem; font-weight:600; color:#3d4a5c; margin-bottom:0.5rem;'>
            No dataset loaded
        </div>
        <div style='font-size:0.9rem;'>
            Upload a CSV file above to start your analysis
        </div>
    </div>
    """, unsafe_allow_html=True)