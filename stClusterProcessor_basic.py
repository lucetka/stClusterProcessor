### Lucie's chatter with Lucie ###

### starting from Dill version on GitHub ###


import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPWORDS
import numpy as np

import io


# --------------------
# Page config
# --------------------
st.set_page_config(
    page_title="Cluster Labeler",
    layout="wide"
)

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
        scores = ctfidf[idx].toarray().flatten()

        # Keep only positive-score terms (avoid arbitrary tokens for empty docs)
        pos = np.where(scores > 0)[0]
        if pos.size == 0:
            cluster_keywords[cid] = ""
            continue

        top_idx = pos[np.argsort(scores[pos])[::-1]][:top_n]
        cluster_keywords[cid] = ", ".join(terms[top_idx])

    return cluster_keywords
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
    df = load_data(uploaded_file.getvalue(), uploaded_file.name)

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
hover_candidates = [c for c in all_columns if c not in {x_col, y_col}]
default_extra_hover = []  # no extras by default

extra_hover_cols = st.sidebar.multiselect(
    "Show these columns in hover (in addition to cluster)",
    options=hover_candidates,
    default=default_extra_hover,
    help="Tip: selecting many columns (or long text columns) can cause memory issues."
)

# Optional safety cap (prevents accidental 'select everything')
MAX_EXTRA_HOVER = 12
if len(extra_hover_cols) > MAX_EXTRA_HOVER:
    st.sidebar.warning(
        f"Showing only the first {MAX_EXTRA_HOVER} extra hover columns to keep the plot responsive."
    )
    extra_hover_cols = extra_hover_cols[:MAX_EXTRA_HOVER]

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

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=cluster_col,
            hover_data=hover_cols,
            render_mode="webgl",
        )

        st.plotly_chart(fig, use_container_width=True)


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

    if labeling_strategy != "cTF-IDF":
        st.info("Select **cTF-IDF** to generate keyword-based cluster labels.")
        st.stop()

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
            options=[(1,1), (1,2), (1,3), (2,3)],
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
    st.container()
