### Lucie's chatter with Lucie ###

### starting from Dill version on GitHub ###


## sparse-safe with removed (2,3) option

## I gave Copilot my old functions and code bits from Jupyter notebooks that I had successfully been using to
## label and comment on clusters using chatGPT API - Howeover, Copilot decided to ignore my strategy and implemented a chatGPT
## strategy which was overengineered and didn't work AT ALL. 
### I forced him to use my old functions and code bits from my old Jupyter notebooks which now seems to work

###  _apple version was a copy of copilots outputted stClusterProcessor_notebook_mode.py which was based on _garlic_patched.py

### this _banana version is a copy of manually edited/patched version of copilot-outputted stClusterProcessor_apple_persist.py 
### which featured persisting labels
#### 
### the only new feature in _banana as compared to stClusterProcessor_apple_persist.py
### is the implementation of download button on the 2D plot chart
####################
### NEXT important to-dos in no particular order:
## - highest priority: accomodate multiple clusterings
## - modularize prompts - yaml
## - let user select unclustered
## - export results packaged
## - improve sampling
## - add columns to chart - partially done
##
####

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPWORDS
import numpy as np

import io
import os
import json
import time
import re
import hashlib

# OpenAI client (optional; only needed for GPT labeling strategy)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


#st.write("compute_data_id exists?", "compute_data_id" in globals())

import hashlib

# --------------------
# Label persistence helpers (session_state)
# --------------------

def compute_data_id(file_bytes: bytes, filename: str) -> str:
    """Lightweight fingerprint for the uploaded dataset."""
    sample = file_bytes[:1_000_000] if file_bytes else b""
    h = hashlib.md5()
    h.update((filename or "").encode("utf-8", errors="ignore"))
    h.update(str(len(file_bytes) if file_bytes is not None else 0).encode("utf-8"))
    h.update(sample)
    return h.hexdigest()

def init_label_store(data_id: str):
    """Initialize (or reset) label store if a new dataset is uploaded."""
    store = st.session_state.get("label_store")
    if (store is None) or (store.get("data_id") != data_id):
        st.session_state["label_store"] = {"data_id": data_id, "by_cluster_col": {}}

def _ensure_cluster_bucket(cluster_col: str) -> dict:
    store = st.session_state.get("label_store", {})
    store.setdefault("by_cluster_col", {})
    store["by_cluster_col"].setdefault(cluster_col, {})
    st.session_state["label_store"] = store
    return store["by_cluster_col"][cluster_col]

def store_ctfidf_labels(cluster_col: str, keyword_map: dict):
    bucket = _ensure_cluster_bucket(cluster_col)
    bucket["ctfidf"] = {str(k): ("" if v is None else str(v)) for k, v in (keyword_map or {}).items()}

def store_gpt_labels(cluster_col: str, labels_df: pd.DataFrame):
    bucket = _ensure_cluster_bucket(cluster_col)
    if labels_df is None or labels_df.empty or "Cluster" not in labels_df.columns:
        return

    tmp = labels_df.copy()
    tmp["Cluster"] = tmp["Cluster"].astype(str)

    def _col_to_map(colname: str) -> dict:
        if colname not in tmp.columns:
            return {}
        return (
            tmp.set_index("Cluster")[colname]
            .astype(str)
            .replace({"<NA>": "", "nan": "", "None": ""})
            .to_dict()
        )

    bucket["gpt_summary"] = _col_to_map("Summary label")
    bucket["gpt_keywords"] = _col_to_map("Keywords")
    bucket["gpt_homogeneity"] = _col_to_map("Homogeneity/Diversity")
    bucket["gpt_subclusters"] = _col_to_map("Subclusters")
    bucket["gpt_df"] = tmp

def get_persisted_label_columns(cluster_col: str) -> list:
    """Which persisted label/enrichment columns are currently available?"""
    store = st.session_state.get("label_store", {})
    bucket = store.get("by_cluster_col", {}).get(cluster_col, {})
    cols = []
    if bucket.get("ctfidf"):
        cols.append("cTF-IDF keywords")
    if bucket.get("gpt_summary"):
        cols.extend(["Summary label", "Keywords", "Homogeneity/Diversity", "Subclusters"])
    # unique preserving order
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def enrich_df_with_labels(df_in: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """Append persisted labeling outputs as columns onto df_in."""
    if df_in is None or df_in.empty or cluster_col not in df_in.columns:
        return df_in

    store = st.session_state.get("label_store", {})
    bucket = store.get("by_cluster_col", {}).get(cluster_col, {})
    if not bucket:
        return df_in

    df = df_in.copy()
    cid = df[cluster_col].astype(str)

    if bucket.get("ctfidf"):
        df["cTF-IDF keywords"] = cid.map(bucket.get("ctfidf", {})).fillna("")

    if bucket.get("gpt_summary"):
        df["Summary label"] = cid.map(bucket.get("gpt_summary", {})).fillna("")
        df["Keywords"] = cid.map(bucket.get("gpt_keywords", {})).fillna("")
        df["Homogeneity/Diversity"] = cid.map(bucket.get("gpt_homogeneity", {})).fillna("")
        df["Subclusters"] = cid.map(bucket.get("gpt_subclusters", {})).fillna("")

    return df

def _shorten_text(val: str, max_len: int) -> str:
    s = "" if val is None else str(val)
    s = re.sub(r"\s+", " ", s).strip()
    if max_len and len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s

def add_enriched_legend_column(df_in: pd.DataFrame, cluster_col: str, legend_style: str, max_len: int = 80) -> pd.DataFrame:
    """Create __legend__ column combining cluster id + persisted labels."""
    if df_in is None or df_in.empty or cluster_col not in df_in.columns:
        return df_in

    store = st.session_state.get("label_store", {})
    bucket = store.get("by_cluster_col", {}).get(cluster_col, {})
    ctfidf_map = bucket.get("ctfidf", {})
    gpt_sum_map = bucket.get("gpt_summary", {})

    uniq = df_in[cluster_col].astype(str).dropna().unique().tolist()
    legend_map = {}

    for cid in uniq:
        parts = []
        if "cTF-IDF" in legend_style and ctfidf_map:
            kw = _shorten_text(ctfidf_map.get(str(cid), ""), max_len)
            if kw:
                parts.append(f"cTF-IDF: {kw}")
        if "GPT" in legend_style and gpt_sum_map:
            sm = _shorten_text(gpt_sum_map.get(str(cid), ""), max_len)
            if sm:
                parts.append(f"GPT: {sm}")

        legend_map[str(cid)] = (str(cid) + "<br>" + "<br>".join(parts)) if parts else str(cid)

    df = df_in.copy()
    df["__legend__"] = df[cluster_col].astype(str).map(legend_map).fillna(df[cluster_col].astype(str))
    return df


# --------------------
# Page config
# --------------------
st.set_page_config(
    page_title="Cluster Labeler",
    layout="wide"
)
st.write("compute_data_id exists?", "compute_data_id" in globals())
st.title("📚 Cluster Labeling & Enrichment")

# --------------------
# Cached data loader
# --------------------

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str):
    if filename.endswith(".csv"):
        return pd.read_csv(
            io.BytesIO(file_bytes),
            dtype="string",
            low_memory=False
        )
    else:
        return pd.read_excel(
            io.BytesIO(file_bytes),
            dtype="string",
            sheet_name=0
        )

#    df = load_data(uploaded_file.getvalue(), uploaded_file.name)

#    # ✅ Normalize Excel junk to real NA
#    df = df.replace(
#        to_replace=[
#            "", " ", "  ", "None", "NONE", "none",
#            "NA", "N/A", "n/a"
#        ],
#        value=pd.NA
#    )

#    # ✅ Now this does what we actually want
#    df = df.dropna(how="all")

    # Initialize / reset persisted labels for this uploaded dataset
    data_id = compute_data_id(file_bytes, uploaded_file.name)
    init_label_store(data_id)

#    return df



# --------------------
# Dynamic filter helper
# --------------------
def apply_dynamic_filter(df, col, mode):
    if col is None:
        return df

    if mode == "Numeric":
      
        s = pd.to_numeric(df[col].str.replace(",", "", regex=False), errors="coerce")   
            #This makes the numeric parsing explicitly robust for:
                    # 2,015
                    # 1,234.5

        s_nonnull = s.dropna()

        if s_nonnull.empty:
            st.warning(f"Column '{col}' cannot be interpreted as numeric.")
            return df

        min_val = float(s_nonnull.min())
        max_val = float(s_nonnull.max())

        ### histogram preview
        
        # ---- histogram preview ----
        with st.container():
            st.caption(f"Distribution of **{col}**")
            hist_df = pd.DataFrame({col: s_nonnull})
            fig = px.histogram(
                hist_df,
                x=col,
                nbins=min(30, s_nonnull.nunique()),
                height=150
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title=None,
                yaxis_title=None
            )
            
            st.plotly_chart(fig, use_container_width=True)


        # ✅ handle single-value numeric columns
        if min_val == max_val:
            st.info(f"Column '{col}' has a single value ({int(min_val)}). No range filter applied.")
            return df

        selected = st.slider(
         f"{col} range",
         min_value=min_val,
         max_value=max_val,
         value=(min_val, max_val)
        )

        return df[s.between(selected[0], selected[1])]

    elif mode == "Categorical":
        values = sorted(df[col].dropna().unique())

        selected = st.multiselect(
            f"Select {col}",
            values,
            default=values
        )

        if not selected:
            return df.iloc[0:0]

        return df[df[col].isin(selected)]

    return df


#### c-TF-IDF ###############
def compute_ctfidf(
    df,
    cluster_col,
    text_cols,
    top_n=10,
    ngram_range=(1, 2),
    extra_stopwords=None
):
    """Compute c-TF-IDF keywords per cluster (BERTopic-style).

    We build one *meta-document* per cluster by concatenating **all** documents
    (rows) belonging to that cluster (and all selected text columns per row).

    Returns dict: {cluster_id: "kw1, kw2, ..."}
    """

    if extra_stopwords is None:
        extra_stopwords = []

    # Normalize stopwords to lowercase strings
    extra_stopwords = [str(w).strip().lower() for w in extra_stopwords if str(w).strip()]
    stopwords = list(SPACY_STOPWORDS.union(set(extra_stopwords)))

    # Work on a copy with only the required columns
    work = df[[cluster_col] + list(text_cols)].copy()
    work = work.dropna(subset=[cluster_col])

    # Build one text per ROW by concatenating selected text columns
    # (prevents bugs where only the first row/first column gets used)
    row_text = (
        work[text_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    work["__row_text__"] = row_text.str.replace(r"\s+", " ", regex=True).str.strip()

    # Concatenate ALL rows per cluster into a single meta-document
    grouped = (
        work.groupby(cluster_col)["__row_text__"]
        .apply(lambda s: " ".join(t for t in s if t))
        .reset_index()
    )

    clusters = grouped[cluster_col].tolist()
    cluster_docs = grouped["__row_text__"].tolist()

    # Edge case: all clusters empty after cleaning
    if not any(doc.strip() for doc in cluster_docs):
        return {cid: "" for cid in clusters}

    vectorizer = CountVectorizer(
        stop_words=stopwords,
        max_features=5000,
        ngram_range=ngram_range,
    )

    X = vectorizer.fit_transform(cluster_docs)
    transformer = TfidfTransformer(norm=None)
    ctfidf = transformer.fit_transform(X)
    terms = np.array(vectorizer.get_feature_names_out())

    cluster_keywords = {}
    for idx, cid in enumerate(clusters):
        row = ctfidf.getrow(idx)
        if row.nnz == 0:
            cluster_keywords[cid] = ""
            continue

        # row.data are the non-zero TF-IDF scores, row.indices are their term indices
        top_local = np.argsort(row.data)[::-1][:top_n]
        top_idx = row.indices[top_local]
        cluster_keywords[cid] = ", ".join(terms[top_idx])

    return cluster_keywords


#### GPT labeling (structured JSON output) ###############

def _safe_json_loads(s: str):
    """Parse JSON robustly.

    Handles common model quirks:
      - code fences (```json ... ```)
      - leading/trailing commentary
      - doubled quotes ""like this"" (often accidental CSV/Excel-style escaping)

    Returns (obj, err_str).
    """
    if not isinstance(s, str):
        return None, "not a string"

    s2 = s.strip()

    # Strip code fences
    s2 = re.sub(r"^```(?:json)?\s*", "", s2, flags=re.IGNORECASE)
    s2 = re.sub(r"\s*```\s*$", "", s2)

    # Collapse pervasive doubled quotes (invalid JSON)
    if '""' in s2 and '\\"' not in s2 and s2.count('""') > 10:
        s2 = s2.replace('""', '"')

    # First attempt: direct parse
    try:
        return json.loads(s2), ""
    except Exception as e1:
        # Try extracting first JSON object substring
        i = s2.find('{')
        j = s2.rfind('}')
        if i != -1 and j != -1 and j > i:
            sub = s2[i:j+1]
            if '""' in sub and '\\"' not in sub and sub.count('""') > 10:
                sub = sub.replace('""', '"')
            try:
                return json.loads(sub), ""
            except Exception as e2:
                return None, f"json decode failed: {e1} | recovered-substring failed: {e2}"
        return None, f"json decode failed: {e1}"


def _parse_notebook_style_label(raw: str):
    """Parse the 1/2/3 sectioned label text from the notebook-style prompt."""
    if not isinstance(raw, str):
        return {"summary_label": "", "keywords": [], "homogeneity": "", "raw": ""}

    txt = raw.strip()

    def _grab(pattern):
        mm = re.search(pattern, txt, flags=re.IGNORECASE | re.DOTALL)
        return mm.group(1).strip() if mm else ""

    summary = _grab(r"1\s*\.\s*Summary\s*label\s*:\s*(.*?)(?:\n\s*2\s*\.|\Z)")
    keywords = _grab(r"2\s*\.\s*Keywords\s*:\s*(.*?)(?:\n\s*3\s*\.|\Z)")
    homog = _grab(r"3\s*\.\s*Homogeneity\s*/\s*Diversity\s*:\s*(.*)\Z")

    # Fallbacks if numbering is missing or varied
    if not summary:
        summary = _grab(r"Summary\s*label\s*:\s*(.*?)(?:\n|\Z)")
    if not keywords:
        keywords = _grab(r"Keywords\s*:\s*(.*?)(?:\n|\Z)")
    if not homog:
        homog = _grab(r"Homogeneity\s*/\s*Diversity\s*:\s*(.*)\Z")

    # Normalize keywords -> list
    kw_list = []
    if keywords:
        parts = re.split(r"[,;\n]+", keywords)
        kw_list = [p.strip() for p in parts if p.strip()]
        kw_list = kw_list[:5]

    return {"summary_label": summary, "keywords": kw_list, "homogeneity": homog, "raw": txt}


def gpt_label_cluster_structured(texts, client, model: str, temperature: float = 0.3, max_tokens: int = 100, retries: int = 0):
    """Notebook-style cluster labeling (matches Lucie's Jupyter workflow).

    Uses the original numbered prompt and returns a dict for the app table.
    """

    items = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not items:
        return {"summary_label": "", "keywords": [], "homogeneity": "", "subclusters": [], "raw_json": {}, "raw_text": ""}

    prompt = (
        "Given the following scientific articles, provide:\n"
        "1. A summary label (max 10 words) that best describes the whole cluster.\n"
        "2. Five keywords or keyphrases (each max 3 words) that represent the cluster topics.\n"
        "3. A brief comment on the homogeneity or diversity of topics within the cluster and if potentially it could be clustered further and how. If it appears homogenous, simply briefly confirm. If not, say it appears to be heterogenous and list potential subclusters labeled very briefly by 3-5 keywords or keyphrases.\n"
        "Format your answer as:\n"
        "1. Summary label: ...\n"
        "2. Keywords: ...\n"
        "3. Homogeneity/Diversity: ...\n\n"
        + "\n".join(items)
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )

    raw = (resp.choices[0].message.content or "").strip()
    parsed = _parse_notebook_style_label(raw)

    return {
        "summary_label": parsed.get("summary_label", ""),
        "keywords": parsed.get("keywords", []),
        "homogeneity": parsed.get("homogeneity", ""),
        "subclusters": [],
        "raw_json": {},
        "raw_text": parsed.get("raw", raw),
    }



def build_cluster_text_samples(df, cluster_col, text_cols, max_docs_per_cluster=25, max_chars_per_doc=1200,
                               skip_cluster_value='-1', sampling='first', max_clusters=None):
    """Create a dict cluster_id -> list[text] with bounded size to control token usage/cost.

    - sampling: 'first' or 'random' (random uses a fixed seed for reproducibility).
    - skip_cluster_value: cluster id to skip (common for HDBSCAN noise). Pass None to label it too.
    - max_clusters: if provided, label only top-N clusters by size (largest first).

    Returns: (cluster_to_texts, cluster_sizes_df)
    """

    work = df[[cluster_col] + list(text_cols)].copy()
    work = work.dropna(subset=[cluster_col])

    # Build one text per row
    row_text = (
        work[text_cols]
        .fillna('')
        .astype(str)
        .agg(' '.join, axis=1)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    work['__row_text__'] = row_text

    # Cluster sizes
    size_series = work[cluster_col].value_counts(dropna=True)
    # pandas version differences: reset_index can yield columns ['index', <series_name>] OR [<index_name>, 'count']
    size_df = size_series.reset_index()
    # Force stable column names
    if size_df.shape[1] >= 2:
        size_df.columns = ['Cluster', 'n_docs'] + list(size_df.columns[2:])
    else:
        size_df.columns = ['Cluster']
        size_df['n_docs'] = 0

    # Optionally skip noise/unclustered
    if skip_cluster_value is not None:
        size_df = size_df[size_df['Cluster'].astype(str) != str(skip_cluster_value)]

    # Pick top-N clusters
    if max_clusters is not None and max_clusters > 0:
        size_df = size_df.head(int(max_clusters))

    clusters = size_df['Cluster'].tolist()

    # Deterministic sampling
    rng = np.random.default_rng(42)

    cluster_to_texts = {}
    for cid in clusters:
        sub = work[work[cluster_col].astype(str) == str(cid)]
        if sub.empty:
            cluster_to_texts[cid] = []
            continue

        if sampling == 'random' and len(sub) > max_docs_per_cluster:
            take_idx = rng.choice(sub.index.to_numpy(), size=max_docs_per_cluster, replace=False)
            sub2 = sub.loc[take_idx]
        else:
            sub2 = sub.head(max_docs_per_cluster)

        texts = []
        for t in sub2['__row_text__'].tolist():
            t = (t or '').strip()
            if not t:
                continue
            if max_chars_per_doc is not None and max_chars_per_doc > 0:
                t = t[:int(max_chars_per_doc)]
            texts.append(t)

        cluster_to_texts[cid] = texts

    return cluster_to_texts, size_df
# --------------------
# Sidebar
# --------------------
st.sidebar.header("Inputs")

st.sidebar.caption(
    "ℹ️ Removing a file only clears the uploader. "
    "Use **Reset session** to fully flush cached data."
)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = st.sidebar.file_uploader(
    "Upload data (CSV or XLSX)",
    type=["csv", "xlsx"],
    key=f"uploader_{st.session_state.uploader_key}"
)




labeling_strategy = st.sidebar.selectbox(
    "Labeling strategy",
    [
        "Manual",
        "Import existing labels",
        "cTF-IDF",
        "GPT (summary + keywords + homogeneity)",
        "Placeholder 2",
        "Placeholder 3",
    ]
)

st.sidebar.divider()

if st.sidebar.button("Reset session (flush everything)"):
    st.cache_data.clear()
    st.session_state.clear()
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
    st.experimental_rerun()


st.sidebar.button("Save labels (placeholder)")

# --------------------
# Load data
# --------------------
if uploaded_file is None:
    st.info("Upload a file to begin.")
    st.stop()

try:
    #df = load_data(uploaded_file)
    file_bytes = uploaded_file.getvalue()
    df = load_data(file_bytes, uploaded_file.name)

    # ✅ Normalize Excel junk to real NA
    df = df.replace(
        to_replace=[
            "", " ", "  ", "None", "NONE", "none",
            "NA", "N/A", "n/a"
        ],
        value=pd.NA
    )

    # ✅ Now this does what we actually want
    df = df.dropna(how="all")

    # Initialize / reset persisted labels for this uploaded dataset
    data_id = compute_data_id(file_bytes, uploaded_file.name)
    init_label_store(data_id)

except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

all_columns = df.columns.tolist()

# --------------------
# Sidebar: cluster + coords
# --------------------
cluster_columns = [c for c in all_columns if "cluster" in c.lower()]
coord_candidates = all_columns

cluster_col = st.sidebar.selectbox(
    "Cluster column",
    options=cluster_columns if cluster_columns else all_columns
)

x_col = st.sidebar.selectbox("2D coord: X", coord_candidates)
y_col = st.sidebar.selectbox("2D coord: Y", coord_candidates)

# --------------------
# Sidebar: Hover columns for Plotly
# --------------------
st.sidebar.divider()
st.sidebar.subheader("Hover columns (scatter)")

# Default: show ONLY the cluster column in hover.
# Allow user to add additional columns.
# Persisted label columns (if you already ran a labeling strategy for this cluster column)
persisted_label_cols = get_persisted_label_columns(cluster_col)
if persisted_label_cols:
    st.sidebar.caption("Available persisted labels: " + ", ".join(persisted_label_cols))

# Extend hover candidates with persisted label columns
hover_candidates = [c for c in list(dict.fromkeys(all_columns + persisted_label_cols)) if c not in {x_col, y_col}]
default_extra_hover = [c for c in ["cTF-IDF keywords", "Summary label"] if c in hover_candidates]

extra_hover_cols = st.sidebar.multiselect(
    "Show these columns in hover (in addition to cluster)",
    options=hover_candidates,
    default=default_extra_hover,
    key="extra_hover_cols",
    help="Tip: selecting many columns (or long text columns) can cause memory issues."
)

# Optional safety cap (prevents accidental 'select everything')
MAX_EXTRA_HOVER = 12
if len(extra_hover_cols) > MAX_EXTRA_HOVER:
    st.sidebar.warning(
        f"Showing only the first {MAX_EXTRA_HOVER} extra hover columns to keep the plot responsive."
    )
    extra_hover_cols = extra_hover_cols[:MAX_EXTRA_HOVER]

st.sidebar.divider()
st.sidebar.subheader("Legend (scatter)")
show_legend = st.sidebar.checkbox("Show legend", value=True, key="show_legend")
use_enriched_legend = st.sidebar.checkbox(
    "Use enriched legend labels (cluster + labels)",
    value=True,
    key="use_enriched_legend",
    help="If labels exist, show cTF-IDF and/or GPT summary in legend entries."
)
legend_style = st.sidebar.selectbox(
    "Legend label content",
    options=["Cluster only", "Cluster + cTF-IDF", "Cluster + GPT summary", "Cluster + cTF-IDF + GPT summary"],
    index=3,
    key="legend_style"
)
legend_max_len = st.sidebar.slider(
    "Max characters per label part (legend)",
    min_value=20,
    max_value=200,
    value=80,
    step=10,
    key="legend_max_len"
)

# --------------------
# Sidebar: Dynamic filters
# --------------------
st.sidebar.divider()
st.sidebar.subheader("Dynamic filters")

filter_cols = [None] + all_columns
filter_modes = ["Numeric", "Categorical"]

# ---- Filter 1 ----
st.sidebar.markdown("**Filter 1**")
f1_col = st.sidebar.selectbox("Column", filter_cols, key="f1_col")
f1_mode = st.sidebar.selectbox("Type", filter_modes, key="f1_mode")

# ---- Filter 2 ----
st.sidebar.markdown("**Filter 2**")
f2_col = st.sidebar.selectbox("Column ", filter_cols, key="f2_col")
f2_mode = st.sidebar.selectbox("Type ", filter_modes, key="f2_mode")

# --------------------
# Apply filters
# --------------------
df_filt = df.copy()
df_filt = apply_dynamic_filter(df_filt, f1_col, f1_mode)
df_filt = apply_dynamic_filter(df_filt, f2_col, f2_mode)

# --------------------
# Tabs
# --------------------
tabs = st.tabs([
    "📄 File preview",
    "📈 2D Scatter",
    "🗂 Cluster browser",
    "🏷 Labeling results",
    "✅ Validation / Comparison",
    "⬇ Exports"
])

# --------------------
# Tab 1: File preview
# --------------------
with tabs[0]:
    st.subheader("Data preview")
    st.write(f"Rows: {len(df_filt):,} | Columns: {len(df_filt.columns)}")
    st.dataframe(df_filt.head(1000), use_container_width=True)

# --------------------
# Tab 2: 2D scatter
# --------------------
with tabs[1]:
    st.subheader("2D scatterplot")

    with st.spinner("Updating scatter plot… this can take really long for large files"):
        # --- everything below is slow ---
        hover_cols = [cluster_col]
        hover_cols += [c for c in extra_hover_cols if c != cluster_col]

        max_points = st.sidebar.number_input(
            "Max points to plot", 1_000, 500_000, 50_000, step=1_000
        )

        plot_df = df_filt
        if len(plot_df) > max_points:
            plot_df = plot_df.sample(n=int(max_points), random_state=42)

        # Append persisted labeling outputs (if available) so they can be used in hover/legend
        plot_df = enrich_df_with_labels(plot_df, cluster_col)

        # Keep only hover columns that actually exist
        hover_cols = [c for c in hover_cols if c in plot_df.columns]

        # Build enriched legend labels (optional)
        color_col = cluster_col
        if use_enriched_legend and legend_style != "Cluster only":
            if get_persisted_label_columns(cluster_col):
                plot_df = add_enriched_legend_column(plot_df, cluster_col, legend_style, max_len=int(legend_max_len))
                color_col = "__legend__"

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=color_col,
            hover_data=hover_cols,
            render_mode="webgl",
        )
        fig.update_layout(showlegend=bool(show_legend))

        st.plotly_chart(fig, use_container_width=True)

# --------------------
# Download buttons for the current 2D plot
# --------------------
        st.markdown("### Download plot")

# 1) Always available: interactive HTML
        html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
         label="⬇️ Download interactive plot (HTML)",
         data=html_bytes,
         file_name="cluster_scatter.html",
         mime="text/html",
        )



# --------------------
# Tab 3: Cluster browser
# --------------------
with tabs[2]:
    st.subheader("Cluster browser")

    clusters = sorted(df_filt[cluster_col].dropna().unique())
    selected_cluster = st.selectbox("Select cluster", clusters)

    cluster_df = df_filt[df_filt[cluster_col] == selected_cluster]
    st.write(f"Documents in cluster: {len(cluster_df)}")

    text_cols = [c for c in df.columns if c.lower() in ["title", "abstract"]]

    if text_cols:
        st.dataframe(cluster_df[text_cols].head(50), use_container_width=True)
    else:
        st.dataframe(cluster_df.head(50), use_container_width=True)

# --------------------
# Tab 4: Labeling results
# --------------------
with tabs[3]:

    st.subheader("Labeling results")

    # ========== Strategy: cTF-IDF ==========
    if labeling_strategy == "cTF-IDF":

        # ---- UI controls ----
        text_cols = st.multiselect(
            "Text columns to use for labeling",
            options=all_columns,
            default=[c for c in all_columns if c.lower() in ["title", "abstract"]]
        )

        extra_stopwords_input = st.text_area(
            "Additional stopwords (comma-separated)",
            value="results, data, dataset, study, analysis, article"
        )

        col1, col2 = st.columns(2)

        with col1:
            ngram_min, ngram_max = st.selectbox(
                "N-gram range",
                options=[(1,1), (1,2), (1,3)],
                index=1
            )

        with col2:
            top_n = st.number_input(
                "Number of keywords",
                min_value=3,
                max_value=30,
                value=10,
                step=1
            )

        run = st.button("Run cTF-IDF labeling")

        # ---- Execution ----
        if run:
            if not text_cols:
                st.error("Please select at least one text column.")
                st.stop()

            extra_stopwords = [
                w.strip().lower()
                for w in extra_stopwords_input.split(",")
                if w.strip()
            ]

            with st.spinner("Computing cTF-IDF keywords…"):
                keyword_map = compute_ctfidf(
                    df=df_filt,
                    cluster_col=cluster_col,
                    text_cols=text_cols,
                    top_n=top_n,
                    ngram_range=(ngram_min, ngram_max),
                    extra_stopwords=extra_stopwords
                )

            st.session_state["ctfidf_keywords"] = keyword_map

            # Persist for other tabs (hover/legend, exports, etc.)
            store_ctfidf_labels(cluster_col, keyword_map)

        # ---- Display ----
        if "ctfidf_keywords" in st.session_state:
            out_df = (
                pd.DataFrame.from_dict(
                    st.session_state["ctfidf_keywords"],
                    orient="index",
                    columns=["cTF-IDF keywords"]
                )
                .reset_index()
                .rename(columns={"index": "Cluster"})
            )

            st.success("cTF-IDF labeling completed.")
            st.dataframe(out_df, use_container_width=True)
            st.session_state["labels_out_df"] = out_df

    # ========== Strategy: GPT (structured JSON) ==========
    elif labeling_strategy == "GPT (summary + keywords + homogeneity)":

        st.markdown(
            "This strategy sends a *sample* of documents per cluster to an OpenAI model and returns a structured JSON label. "
            "For large datasets, **limit the number of clusters and documents per cluster** to control cost and latency."
        )

        # ---- API key handling ----
        default_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            value=st.session_state.get("openai_api_key", default_key),
            help="Your key is kept only in this browser session (Streamlit session_state)."
        )
        if api_key:
            st.session_state["openai_api_key"] = api_key

        colA, colB, colC = st.columns([1.2, 1, 1])
        with colA:
            model = st.text_input(
                "Model",
                value=st.session_state.get("openai_model", "gpt-4o-mini"),
                help="Any chat-completions model available to your key."
            )
        with colB:
            max_clusters = st.number_input(
                "Max clusters to label (largest first)",
                min_value=1,
                max_value=5000,
                value=50,
                step=10
            )
        with colC:
            docs_per_cluster = st.number_input(
                "Docs per cluster (sample)",
                min_value=10,
                max_value=200,
                value=25,
                step=5
            )

        st.session_state["openai_model"] = model

        text_cols = st.multiselect(
            "Text columns sent to the model (concatenated per row)",
            options=all_columns,
            default=[c for c in all_columns if c.lower() in ["title", "abstract"]]
        )

        colD, colE, colF = st.columns(3)
        with colD:
            sampling = st.selectbox("Sampling", options=["first", "random"], index=0)
        with colE:
            max_chars_per_doc = st.number_input(
                "Max characters per document",
                min_value=200,
                max_value=4000,
                value=1200,
                step=200
            )
        with colF:
            temperature = st.slider("Temperature", min_value=0.3, max_value=1.0, value=0.0, step=0.05)

        max_tokens = st.number_input(
            "Max tokens for the response (per cluster)",
            min_value=100,
            max_value=800,
            value=280,
            step=20
        )

        skip_noise = st.checkbox("Skip cluster -1 (noise/unclustered)", value=True)
        noise_value = "-1" if skip_noise else None

        run_gpt = st.button("Run GPT labeling")

        if run_gpt:
            if OpenAI is None:
                st.error("OpenAI library not available. Add 'openai' to requirements.txt on Streamlit Cloud.")
                st.stop()
            if not api_key:
                st.error("Please provide an OpenAI API key.")
                st.stop()
            if not text_cols:
                st.error("Please select at least one text column.")
                st.stop()

            # Build samples per cluster
            with st.spinner("Preparing per-cluster samples…"):
                cluster_to_texts, size_df = build_cluster_text_samples(
                    df=df_filt,
                    cluster_col=cluster_col,
                    text_cols=text_cols,
                    max_docs_per_cluster=int(docs_per_cluster),
                    max_chars_per_doc=int(max_chars_per_doc),
                    skip_cluster_value=noise_value,
                    sampling=sampling,
                    max_clusters=int(max_clusters),
                )

            st.info(f"Labeling {len(cluster_to_texts):,} clusters (largest first).")

            client = OpenAI(api_key=api_key)

            rows = []
            progress = st.progress(0)
            status = st.empty()

            cluster_items = list(cluster_to_texts.items())
            n_total = len(cluster_items)

            for i, (cid, texts) in enumerate(cluster_items, start=1):
                status.write(f"Cluster {cid} ({i}/{n_total})")

                # Safety: if a cluster has no usable text, keep empty label
                if not texts:
                    rows.append({
                        "Cluster": cid,
                        "Summary label": "",
                        "Keywords": "",
                        "Homogeneity/Diversity": "",
                        "Subclusters": "[]",
                        "n_docs_used": 0,
                        "error": "no text",
                    })
                    progress.progress(i / n_total)
                    continue

                try:
                    out = gpt_label_cluster_structured(
                        texts=texts,
                        client=client,
                        model=model,
                        temperature=float(temperature),
                        max_tokens=int(max_tokens),
                    )

                    rows.append({
                        "Cluster": cid,
                        "Summary label": out.get("summary_label", ""),
                        "Keywords": "; ".join(out.get("keywords", [])),
                        "Homogeneity/Diversity": out.get("homogeneity", ""),
                        "Subclusters": json.dumps(out.get("subclusters", []), ensure_ascii=False),
                        "n_docs_used": len(texts),
                        "error": "",
                    })

                except Exception as e:
                    rows.append({
                        "Cluster": cid,
                        "Summary label": "",
                        "Keywords": "",
                        "Homogeneity/Diversity": "",
                        "Subclusters": "[]",
                        "n_docs_used": len(texts),
                        "error": str(e)[:300],
                    })

                    # gentle backoff in case of rate limits
                    time.sleep(0.5)

                progress.progress(i / n_total)

            labels_df = pd.DataFrame(rows)
            st.session_state["gpt_labels_df"] = labels_df

            # Persist for other tabs (hover/legend, exports, etc.)
            store_gpt_labels(cluster_col, labels_df)
            st.session_state["labels_out_df"] = labels_df

        # Display
        if "gpt_labels_df" in st.session_state:
            st.success("GPT labeling completed.")
            st.dataframe(st.session_state["gpt_labels_df"], use_container_width=True)

            n_err = (st.session_state["gpt_labels_df"]["error"].astype(str).str.len() > 0).sum()
            if n_err:
                st.warning(f"{n_err} clusters returned an error (see 'error' column).")

    else:
        st.info("Select a labeling strategy in the sidebar to generate cluster labels.")


# --------------------
# Tab 5: Validation / comparison

# --------------------
with tabs[4]:
    st.subheader("Validation / comparison")
    st.container()

# --------------------
# Tab 6: Exports
# --------------------
with tabs[5]:
    st.subheader("Exports")

    if "labels_out_df" not in st.session_state:
        st.info("Run a labeling strategy first. The resulting labels table will appear here for export.")
    else:
        out_df = st.session_state["labels_out_df"].copy()
        st.dataframe(out_df.head(200), use_container_width=True)

        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download labels (CSV)",
            data=csv_bytes,
            file_name="cluster_labels.csv",
            mime="text/csv",
        )

        # Excel export (optional)
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
            out_df.to_excel(writer, index=False, sheet_name="labels")
        st.download_button(
            "Download labels (XLSX)",
            data=towrite.getvalue(),
            file_name="cluster_labels.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
