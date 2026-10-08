import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
import io
import time
from html import escape
from textwrap import dedent

# ─── Page Config ───
st.set_page_config(page_title="DataPilot AI", page_icon="🚀", layout="wide")

# ─── Premium CSS Design System ───
st.markdown("""
<style>
    :root {
        color-scheme: dark;
        --page: #0b1220;
        --surface: #111b2d;
        --surface-raised: #17243a;
        --surface-soft: #1b2a42;
        --border: rgba(148, 163, 184, .16);
        --border-strong: rgba(148, 163, 184, .26);
        --text: #f1f5f9;
        --muted: #9aacc2;
        --quiet: #71839b;
        --blue: #59c6ff;
        --violet: #a990ff;
        --mint: #48d6b0;
        --amber: #ffc76a;
        --rose: #ff8296;
        --radius: 18px;
        --shadow: 0 16px 38px rgba(1, 7, 18, .24);
        --text-primary: var(--text);
        --text-secondary: var(--muted);
        --text-muted: var(--quiet);
        --accent-emerald: var(--mint);
        --accent-amber: var(--amber);
        --accent-rose: var(--rose);
    }
    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }
    .stApp {
        color: var(--text);
        background: radial-gradient(ellipse at 15% 0%, #15233a 0%, var(--page) 48%, #080e18 100%) !important;
    }
    .main .block-container { max-width: 1920px; padding: 1.5rem clamp(1rem, 2.5vw, 2.4rem) 3rem; }
    footer { visibility: hidden; }
    h1, h2, h3, h4 { color: var(--text); letter-spacing: -.025em; }
    h1 { font-size: clamp(2rem, 4vw, 3.25rem) !important; }
    h2 { font-size: clamp(1.25rem, 2.2vw, 1.65rem) !important; }
    p, li, label { line-height: 1.6; }
    a { color: var(--blue); }
    ::selection { background: rgba(89, 198, 255, .28); }

    /* Welcome page */
    .hero-wrapper { padding: 2.25rem 1rem 1.35rem; text-align: center; }
    .hero-logo { display: block; margin-bottom: .5rem; font-size: 3.25rem; }
    .hero-title {
        margin: 0; color: var(--text); font-size: clamp(2.6rem, 6vw, 4.1rem); font-weight: 800;
        letter-spacing: -.055em; background: linear-gradient(110deg, #8cddff, #a990ff 70%, #e0a8ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-subtitle { margin: .7rem 0 .25rem; color: #c5d3e4; font-size: 1.1rem; }
    .hero-tagline { margin: 0; color: var(--muted); font-size: .95rem; }
    .hero-card { padding: 1.5rem; border: 1px solid var(--border); border-radius: 22px; background: rgba(17, 27, 45, .86); box-shadow: var(--shadow); }
    .feature-card {
        min-height: 150px; height: 100%; padding: 1.2rem; text-align: left;
        border: 1px solid var(--border); border-radius: var(--radius); background: rgba(17, 27, 45, .72);
    }
    .feature-icon { font-size: 1.7rem; }
    .feature-title { margin: .65rem 0 .3rem; color: var(--text); font-weight: 700; }
    .feature-desc { color: var(--muted); font-size: .82rem; line-height: 1.55; }

    /* Shared surfaces */
    .section-header { display:flex; align-items:center; gap:.75rem; margin:1.65rem 0 1rem; flex-wrap:wrap; }
    .section-header .sh-icon { font-size:1.4rem; }
    .section-header .sh-text { margin:0; color:var(--text); font-size:1.35rem; font-weight:750; }
    .section-header .sh-badge { padding:.28rem .62rem; border:1px solid var(--border); border-radius:999px; color:var(--muted); font-size:.66rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    .metric-grid { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; margin:1rem 0 1.6rem; }
    .metric-card {
        min-width:0; min-height:145px; padding:1.15rem .8rem; text-align:center;
        border:1px solid var(--border); border-radius:18px; background:linear-gradient(145deg,rgba(24,37,58,.98),rgba(17,27,45,.98));
        box-shadow:0 8px 22px rgba(1,7,18,.15);
    }
    .metric-card:hover { border-color:var(--border-strong); background:linear-gradient(145deg,#1b2b43,#152238); }
    .metric-card::before { display:none; }
    .metric-card .mc-icon { display:block; margin-bottom:.6rem; font-size:1.45rem; }
    .metric-card .mc-label { margin:0; color:var(--muted); font-size:.68rem; font-weight:700; letter-spacing:.11em; text-transform:uppercase; }
    .metric-card .mc-value { margin:.35rem 0 0; font-size:clamp(1.25rem,1.8vw,1.8rem); font-weight:800; line-height:1.15; overflow-wrap:anywhere; }
    .mc-blue .mc-value { color:var(--blue); background:none; -webkit-text-fill-color:currentColor; }
    .mc-violet .mc-value, .mc-purple .mc-value { color:var(--violet); background:none; -webkit-text-fill-color:currentColor; }
    .mc-emerald .mc-value { color:var(--mint); background:none; -webkit-text-fill-color:currentColor; }
    .mc-amber .mc-value { color:var(--amber); background:none; -webkit-text-fill-color:currentColor; }
    .mc-rose .mc-value { color:var(--rose); background:none; -webkit-text-fill-color:currentColor; }
    .insight-card { margin:.7rem 0; padding:.95rem 1.1rem; border:1px solid var(--border); border-left:3px solid var(--violet); border-radius:12px; background:rgba(17,27,45,.82); color:#d5dfec; line-height:1.6; }
    .insight-success { border-left-color:var(--mint); }
    .insight-warning { border-left-color:var(--amber); }
    .insight-danger { border-left-color:var(--rose); }
    .quality-badge { display:inline-flex; padding:.4rem .75rem; border-radius:999px; font-size:.78rem; font-weight:700; }
    .quality-good { color:var(--mint); background:rgba(72,214,176,.1); border:1px solid rgba(72,214,176,.24); }
    .quality-warning { color:var(--amber); background:rgba(255,199,106,.1); border:1px solid rgba(255,199,106,.24); }
    .quality-bad { color:var(--rose); background:rgba(255,130,150,.1); border:1px solid rgba(255,130,150,.24); }
    .quality-ring { width:112px; height:112px; margin:.3rem auto; display:grid; place-items:center; border-radius:50%; }
    .quality-ring-inner { width:84px; height:84px; display:flex; flex-direction:column; align-items:center; justify-content:center; border-radius:50%; background:var(--surface); }
    .quality-ring-value { font-size:1.7rem; font-weight:800; line-height:1; }
    .quality-ring-label { margin-top:.25rem; color:var(--quiet); font-size:.62rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
    .stat-pill { display:inline-flex; padding:.4rem .7rem; border:1px solid var(--border); border-radius:999px; color:var(--muted); background:var(--surface); font-size:.77rem; }
    .audit-item { display:flex; gap:.7rem; align-items:center; margin:.45rem 0; padding:.7rem .85rem; border:1px solid var(--border); border-radius:12px; background:var(--surface); }
    .audit-num { width:24px; height:24px; display:grid; place-items:center; flex:none; border-radius:50%; color:#091321; background:var(--blue); font-size:.7rem; font-weight:800; }
    .audit-text { color:#c2cfdf; font-size:.86rem; }
    .empty-state { padding:3rem 1.5rem; text-align:center; color:var(--muted); }
    .empty-state .es-icon { display:block; margin-bottom:.75rem; font-size:2.5rem; }
    .empty-state .es-title { color:var(--text); font-size:1.15rem; font-weight:700; }
    .empty-state .es-desc { max-width:450px; margin:.4rem auto; font-size:.9rem; line-height:1.6; }
    .lb-winner { padding:1.3rem; border:1px solid rgba(89,198,255,.25); border-radius:16px; background:linear-gradient(120deg,rgba(89,198,255,.08),rgba(169,144,255,.1)); text-align:center; }
    .lb-winner-title { color:var(--muted); font-size:.68rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .lb-winner-name { margin:.35rem 0; color:#b9eaff; font-size:1.45rem; font-weight:800; }
    .lb-winner-score { color:var(--mint); font-weight:700; }

    /* Controls and data tables */
    section[data-testid="stSidebar"] { background:#0e1727 !important; border-right:1px solid var(--border); }
    section[data-testid="stSidebar"] h3 { color:#b9eaff !important; }
    div[data-testid="stMetric"] { min-height:100px; padding:.95rem 1rem; border:1px solid var(--border); border-radius:15px; background:var(--surface); }
    div[data-testid="stMetric"] label { color:var(--muted) !important; font-size:.72rem !important; font-weight:700 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:var(--text) !important; font-weight:750 !important; }
    .stButton > button, [data-testid="stDownloadButton"] button { min-height:42px; border:1px solid var(--border-strong) !important; border-radius:12px !important; font-weight:650 !important; transition:background .18s ease,border-color .18s ease !important; }
    .stButton > button:hover, [data-testid="stDownloadButton"] button:hover { transform:none !important; border-color:rgba(89,198,255,.55) !important; box-shadow:0 6px 18px rgba(1,7,18,.18) !important; }
    button[kind="primary"] { border-color:rgba(89,198,255,.35) !important; background:linear-gradient(115deg,#227cb0,#5a64b5) !important; color:#fff !important; }
    [data-baseweb="select"] > div, [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stTextArea"] textarea { border-color:var(--border-strong) !important; border-radius:11px !important; background:#101a2b !important; }
    [data-testid="stFileUploaderDropzone"] { border:1px dashed rgba(148,163,184,.3); border-radius:15px; background:rgba(17,27,45,.62); }
    [data-testid="stFileUploaderDropzone"]:hover { border-color:rgba(89,198,255,.55); background:rgba(23,36,58,.78); }
    [data-testid="stDataFrame"], [data-testid="stTable"] { overflow:hidden; border:1px solid var(--border); border-radius:14px; }
    [data-testid="stExpander"] { border:1px solid var(--border); border-radius:14px; background:rgba(17,27,45,.6); }
    hr { border-color:var(--border) !important; }
    [data-testid="stAlert"] { border-radius:13px; }
    [data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,var(--blue),var(--violet)); }

    /* Workspace navigation: keep all nine areas visible as one full-width bar. */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        display:flex; width:100%; gap:.35rem; padding:.4rem; overflow-x:auto;
        border:1px solid var(--border); border-radius:16px;
        background:rgba(13,22,36,.94); scrollbar-width:thin;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        flex:1 1 0; min-width:0; min-height:44px; height:44px; justify-content:center;
        padding:0 .45rem; border:1px solid transparent; border-radius:11px;
        color:var(--muted); white-space:nowrap; font-size:.81rem; font-weight:680;
        transition:background .16s ease,color .16s ease,border-color .16s ease;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] p { overflow:visible; white-space:nowrap; font-size:inherit; }
    [data-testid="stTabs"] [data-baseweb="tab"]:hover { color:var(--text); background:rgba(89,198,255,.07); }
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        color:#eaf8ff; border-color:rgba(89,198,255,.2); background:linear-gradient(135deg,rgba(43,133,177,.27),rgba(117,93,194,.2));
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { height:2px; background:linear-gradient(90deg,var(--blue),var(--violet)); }
    [data-testid="stTabs"] [data-baseweb="tab-border"] { background:transparent; }

    /* Chat workspace */
    .chat-hero { display:flex; align-items:center; gap:1rem; margin:.4rem 0 .8rem; padding:1rem 1.2rem; border:1px solid rgba(169,144,255,.2); border-radius:18px; background:linear-gradient(110deg,rgba(25,39,62,.98),rgba(17,27,45,.95)); }
    .chat-hero-icon { width:48px; height:48px; display:grid; place-items:center; flex:none; border:1px solid rgba(169,144,255,.25); border-radius:15px; background:rgba(169,144,255,.12); font-size:1.45rem; }
    .chat-hero-copy { flex:1; min-width:0; }
    .chat-hero-kicker { margin:0 0 .2rem; color:var(--blue); font-size:.65rem; font-weight:750; letter-spacing:.13em; text-transform:uppercase; }
    .chat-hero-title { margin:0; color:var(--text); font-size:1.25rem; font-weight:750; }
    .chat-hero-subtitle { margin:.25rem 0 0; color:var(--muted); font-size:.84rem; }
    .chat-model-chip { flex:none; padding:.4rem .65rem; border:1px solid rgba(72,214,176,.25); border-radius:999px; color:var(--mint); background:rgba(72,214,176,.08); font-size:.65rem; font-weight:750; letter-spacing:.06em; }
    .chat-status { display:flex; align-items:center; gap:.55rem; margin:0 0 .75rem; padding:.65rem .9rem; border:1px solid var(--border); border-radius:12px; color:#bfccdc; background:rgba(17,27,45,.78); font-size:.83rem; }
    .chat-status-ready { border-color:rgba(72,214,176,.22); }
    .chat-status-warn { border-color:rgba(255,199,106,.23); }
    .chat-status-offline { border-color:rgba(255,130,150,.22); }
    .chat-welcome { display:flex; align-items:center; gap:1rem; margin:.15rem 0 1rem; padding:1.15rem 1.25rem; border:1px solid rgba(169,144,255,.18); border-radius:17px; background:linear-gradient(120deg,rgba(23,36,58,.94),rgba(17,27,45,.94)); color:var(--muted); text-align:left; }
    .chat-welcome-icon { display:grid; place-items:center; width:50px; height:50px; flex:none; border:1px solid rgba(169,144,255,.24); border-radius:16px; background:rgba(169,144,255,.1); font-size:1.55rem; }
    .chat-welcome-copy { min-width:0; }
    .chat-welcome-title { margin:0; color:var(--text); font-size:1rem; font-weight:700; }
    .chat-welcome-desc { margin:.2rem 0 0; color:var(--muted); font-size:.82rem; line-height:1.5; }
    .st-key-chat-history { border-color:var(--border) !important; border-radius:18px !important; background:rgba(10,17,28,.58) !important; }
    .st-key-chat-history [data-testid="stVerticalBlock"] { gap:.65rem; }
    [data-testid="stChatMessage"] { width:min(100%,1120px); margin:.35rem auto; padding:.75rem 1rem; border:1px solid var(--border); border-radius:15px; background:rgba(17,27,45,.82); }
    [data-testid="stChatMessage"] p { line-height:1.7; }
    [data-testid="stChatInput"] { width:100%; max-width:none; margin:.55rem auto 0; }
    [data-testid="stChatInput"] > div { border:1px solid var(--border-strong); border-radius:16px; background:rgba(17,27,45,.98) !important; box-shadow:0 10px 28px rgba(1,7,18,.2); }
    [data-testid="stChatInput"] textarea { min-height:46px !important; border:0 !important; border-radius:15px !important; background:transparent !important; line-height:1.5; }
    [data-testid="stChatInput"] button { border:1px solid rgba(89,198,255,.24) !important; border-radius:11px !important; background:linear-gradient(135deg,#2588b8,#6a69c6) !important; color:white !important; }
    [data-testid="stChatInput"] textarea:focus { border-color:var(--blue) !important; box-shadow:0 0 0 3px rgba(89,198,255,.12) !important; }

    @media (max-width: 1250px) {
        [data-testid="stTabs"] [data-baseweb="tab"] { flex:0 0 112px; min-width:112px; }
    }
    @media (max-width: 1150px) { .metric-grid { grid-template-columns:repeat(3,minmax(0,1fr)); } }
    @media (max-width: 700px) {
        .main .block-container { padding:1rem .75rem 2rem; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { justify-content:flex-start; }
        [data-testid="stTabs"] [data-baseweb="tab"] { flex:0 0 auto; min-width:102px; padding:0 .65rem; }
        .metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); gap:.6rem; }
        .metric-card { min-height:126px; padding:.95rem .45rem; }
        .hero-wrapper { padding:1.5rem .5rem 1rem; }
        .hero-card { padding:1rem; }
        .feature-card { min-height:130px; }
        .chat-hero { align-items:flex-start; flex-wrap:wrap; padding:1rem; }
        .chat-model-chip { margin-left:3.7rem; }
        .chat-hero-title { font-size:1.08rem; }
        .chat-welcome { align-items:flex-start; padding:.9rem; }
        .chat-welcome-icon { width:42px; height:42px; }
        [data-testid="stChatMessage"] { padding:.7rem; }
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───
WORKSPACES = (
    ("📊 Overview", "Overview"),
    ("🧹 Cleaning", "Cleaning"),
    ("📈 EDA", "Explore"),
    ("🧪 Statistics", "Statistics"),
    ("🤖 AutoML", "AutoML"),
    ("🎯 Predict", "Predict"),
    ("🔍 Anomalies", "Anomalies"),
    ("✨ Insights", "Insights"),
    ("💬 AI Chat", "AI Chat"),
)

defaults = {
    "df": None,
    "messages": [],
    "trained_model": None,
    "model_info": {},
    "cleaning_log": [],
    "filename": "Dataset",
    "dataset_profile": None,
    "export_payload": None,
    "loaded_upload_signature": None,
    "workspace_nav": WORKSPACES[0][0],
    "pending_chat_prompt": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        if key == "workspace_nav":
            previous_workspace = st.session_state.get("active_workspace", "Overview")
            val = next(
                (label for label, name in WORKSPACES if name == previous_workspace),
                val,
            )
        st.session_state[key] = val

# ─── Helper Functions ───
def safe_html_text(value):
    """Escape user data while retaining only the simple formatting tags we emit."""
    escaped = escape(str(value))
    for tag in ("b", "strong", "br", "code", "i"):
        escaped = escaped.replace(f"&lt;{tag}&gt;", f"<{tag}>")
        escaped = escaped.replace(f"&lt;/{tag}&gt;", f"</{tag}>")
    return escaped


def set_dataset(dataframe, reset_ui=False):
    """Replace the active dataset and invalidate derived state."""
    st.session_state.df = dataframe
    st.session_state.dataset_profile = None
    st.session_state.export_payload = None
    st.session_state.trained_model = None
    st.session_state.model_info = {}
    st.session_state.messages = []
    if reset_ui:
        for widget_key in (
            "automl_feats", "eda_chart_col", "scatter_multi", "iso_cols",
            "outlier_col", "type_conv_col", "iqr_col", "zscore_col",
        ):
            st.session_state.pop(widget_key, None)
        st.session_state.workspace_nav = WORKSPACES[0][0]


def get_dataset_profile(df):
    """Compute expensive dataset-wide counts once for the current dataframe."""
    cached = st.session_state.get("dataset_profile")
    if cached and cached["data_id"] == id(df):
        return cached

    def build_profile():
        missing_by_column = df.isna().sum()
        missing_total = int(missing_by_column.sum())
        duplicate_scan_skipped = len(df) > 1_000_000
        duplicate_total = None if duplicate_scan_skipped else (int(df.duplicated().sum()) if len(df) else 0)
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (missing_total / total_cells * 100) if total_cells else 0
        duplicate_pct = (duplicate_total / len(df) * 100) if len(df) and duplicate_total is not None else 0
        quality = max(0, min(100, int(100 - missing_pct * 1.5 - duplicate_pct * 2)))
        return {
            "data_id": id(df),
            "missing_by_column": missing_by_column,
            "missing_total": missing_total,
            "duplicate_total": duplicate_total,
            "duplicate_scan_skipped": duplicate_scan_skipped,
            "quality": quality,
            "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
            "categorical_columns": df.select_dtypes(exclude=np.number).columns.tolist(),
            "overview_column_profile": None,
            "eda_describe": None,
        }

    if len(df) >= 100_000:
        with st.spinner("Profiling this large dataset once…"):
            profile = build_profile()
    else:
        profile = build_profile()
    st.session_state.dataset_profile = profile
    return profile


def get_overview_column_profile(df):
    """Cache the overview table so returning to Overview doesn't rescan the dataset."""
    profile = get_dataset_profile(df)
    if profile["overview_column_profile"] is None:
        missing = profile["missing_by_column"]
        unique = df.nunique(dropna=True)
        samples = []
        for column in df.columns:
            if missing[column] == len(df):
                samples.append("—")
            else:
                samples.append(str(df[column].dropna().iloc[0]))

        profile["overview_column_profile"] = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": (len(df) - missing).values,
            "Missing": missing.values,
            "Missing %": (missing / max(len(df), 1) * 100).round(1).values,
            "Unique": unique.values,
            "Sample Value": samples,
        })
    return profile["overview_column_profile"]


def get_eda_describe(df):
    """Keep descriptive statistics across widget reruns for the same dataset."""
    profile = get_dataset_profile(df)
    if profile["eda_describe"] is None:
        profile["eda_describe"] = df.describe(include="all").T
    return profile["eda_describe"]


def get_chat_data_summary(df):
    """Build and reuse a compact summary so chat never scans a large dataset per turn."""
    profile = get_dataset_profile(df)
    if "chat_summary" not in profile:
        is_sampled = len(df) > 100_000
        summary_df = df.sample(n=100_000, random_state=42) if is_sampled else df
        stats_text = []
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    values = summary_df[col].dropna()
                    if len(values):
                        stats_text.append(
                            f"- '{col}' (Numeric): Mean={values.mean():.2f}, "
                            f"Median={values.median():.2f}, Min={values.min():.2f}, "
                            f"Max={values.max():.2f}, Std={values.std():.2f}"
                        )
                else:
                    counts = summary_df[col].value_counts().head(5)
                    top_values = ", ".join(f"{value}: {count}" for value, count in counts.items())
                    stats_text.append(f"- '{col}' (Categorical): Top values: {top_values}")
            except Exception:
                stats_text.append(f"- '{col}': [summary unavailable]")
        missing = profile["missing_by_column"]
        missing_text = ", ".join(f"{col}: {int(count)}" for col, count in missing.items() if count) or "None"
        profile["chat_summary"] = {
            "columns": "\n".join(stats_text),
            "missing": missing_text,
            "sampling_note": (
                f"Column statistics are estimates based on a reproducible 100,000-row sample from {len(df):,} rows."
                if is_sampled else "Column statistics use all rows."
            ),
        }
    return profile["chat_summary"]


def load_data(file):
    """Load data from uploaded file with robust error handling."""
    try:
        signature = (
            file.name,
            getattr(file, "size", None),
            getattr(file, "file_id", None),
        )
        if signature == st.session_state.loaded_upload_signature:
            return
        name = file.name.lower()
        with st.spinner("Loading and preparing your dataset…"):
            if name.endswith('.csv'):
                dataframe = pd.read_csv(file)
            elif name.endswith('.json'):
                dataframe = pd.read_json(file)
            elif name.endswith(('.xlsx', '.xls')):
                dataframe = pd.read_excel(file)
            else:
                st.error(f"Unsupported file format: {file.name}")
                return
        set_dataset(dataframe, reset_ui=True)
        st.session_state.filename = file.name
        st.session_state.loaded_upload_signature = signature
        st.session_state.cleaning_log = []
        st.session_state.trained_model = None
        st.session_state.model_info = {}
        st.session_state.messages = []
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")

def compute_quality_score(df):
    """Return the cached quality score for the active dataset."""
    return get_dataset_profile(df)["quality"]


def queue_chat_prompt(prompt):
    """Send a starter suggestion through the same flow as a typed chat prompt."""
    st.session_state.pending_chat_prompt = prompt


@st.cache_data(ttl=15, show_spinner=False)
def get_local_models(ollama_url):
    """Cache local Ollama availability briefly to avoid a network wait on every rerun."""
    import requests

    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=3)
        if response.status_code == 200:
            return [model["name"] for model in response.json().get("models", [])]
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False)
def load_sample_dataset(name):
    """Load an optional sample dataset only when the user requests it."""
    import seaborn as sns
    return sns.load_dataset(name)

def auto_select_chart(series, col_name):
    """Automatically select the best chart type based on column data."""
    if pd.api.types.is_numeric_dtype(series):
        nunique = series.nunique()
        if nunique <= 10:
            return "bar"
        try:
            skew = series.dropna().skew()
            if abs(skew) > 1:
                return "box"
        except Exception:
            pass
        return "histogram"
    else:
        nunique = series.nunique()
        if nunique <= 15:
            return "bar"
        return "treemap"

def detect_problem_type(y):
    """Auto-detect classification vs regression."""
    if y.dtype == 'object' or str(y.dtype) == 'category':
        return 'classification'
    nunique = y.nunique()
    if nunique <= 20 and nunique / len(y) < 0.05:
        return 'classification'
    return 'regression'

def suggest_target(df):
    """Auto-suggest likely target columns."""
    target_keywords = ['target', 'label', 'class', 'survived', 'outcome', 'result', 'y', 'price', 'salary', 'revenue']
    for col in df.columns:
        if col.lower().strip() in target_keywords:
            return col
    return df.columns[-1]

def safe_numeric_stat(series, stat_func):
    """Safely compute a numeric statistic, returning None on failure."""
    try:
        if pd.api.types.is_numeric_dtype(series):
            val = stat_func(series.dropna())
            if pd.notna(val) and np.isfinite(val):
                return val
    except Exception:
        pass
    return None

def get_df_download(df, fmt="csv"):
    """Convert df to downloadable bytes."""
    buf = io.BytesIO()
    if fmt == "csv":
        df.to_csv(buf, index=False)
    elif fmt == "excel":
        df.to_excel(buf, index=False, engine='openpyxl')
    buf.seek(0)
    return buf.getvalue()

def render_metric_card(icon, label, value, color_class="mc-blue"):
    """Render a styled metric card."""
    return f"""<div class='metric-card {color_class}'>
        <span class='mc-icon'>{safe_html_text(icon)}</span>
        <p class='mc-label'>{safe_html_text(label)}</p>
        <p class='mc-value'>{safe_html_text(value)}</p>
    </div>"""

def render_section_header(icon, text, badge=""):
    """Render a styled section header."""
    badge_html = f"<span class='sh-badge'>{safe_html_text(badge)}</span>" if badge else ""
    st.markdown(f"""<div class='section-header'>
        <span class='sh-icon'>{safe_html_text(icon)}</span>
        <h2 class='sh-text'>{safe_html_text(text)}</h2>
        {badge_html}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 1: LANDING PAGE
# ══════════════════════════════════════════════
if st.session_state.df is None:
    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        st.markdown("""
        <div class='hero-wrapper'>
            <span class='hero-logo'>🚀</span>
            <h1 class='hero-title'>DataPilot AI</h1>
            <p class='hero-subtitle'>Upload · Analyze · Predict · Understand</p>
            <p class='hero-tagline'>Your intelligent, local-first data science copilot</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop your dataset here — CSV, Excel, or JSON",
            type=["csv", "xlsx", "xls", "json"],
            help="Supports files up to 200MB. Data stays local."
        )
        if uploaded_file is not None:
            load_data(uploaded_file)

        st.divider()
        st.markdown("<p style='text-align: center; color: var(--text-muted); font-weight: 600; font-size: 0.9rem;'>Quick start · Load a sample dataset</p>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        datasets = [
            (c1, "🚢", "Titanic", "Classification", 'titanic', "titanic.csv"),
            (c2, "🌸", "Iris", "Multi-class", 'iris', "iris.csv"),
            (c3, "🐧", "Penguins", "Multi-class", 'penguins', "penguins.csv"),
        ]
        for col, emoji, name, task, sns_name, fname in datasets:
            with col:
                if st.button(f"{emoji} {name} ({task})", width="stretch", key=f"sample_{sns_name}"):
                    set_dataset(load_sample_dataset(sns_name), reset_ui=True)
                    st.session_state.filename = fname
                    st.session_state.cleaning_log = []
                    st.session_state.trained_model = None
                    st.session_state.model_info = {}
                    st.rerun()

        c4, c5, c6 = st.columns(3)
        datasets2 = [
            (c4, "💎", "Diamonds", "Regression", 'diamonds', "diamonds.csv"),
            (c5, "🏥", "Breast Cancer", "Binary Class", None, "breast_cancer.csv"),
            (c6, "🏠", "California Housing", "Regression", None, "california_housing.csv"),
        ]
        for col, emoji, name, task, sns_name, fname in datasets2:
            with col:
                if st.button(f"{emoji} {name} ({task})", width="stretch", key=f"sample_{fname}"):
                    if sns_name:
                        set_dataset(load_sample_dataset(sns_name), reset_ui=True)
                    elif fname == "breast_cancer.csv":
                        from sklearn.datasets import load_breast_cancer
                        data = load_breast_cancer(as_frame=True)
                        set_dataset(data.frame, reset_ui=True)
                    elif fname == "california_housing.csv":
                        from sklearn.datasets import fetch_california_housing
                        data = fetch_california_housing(as_frame=True)
                        set_dataset(data.frame, reset_ui=True)
                    st.session_state.filename = fname
                    st.session_state.cleaning_log = []
                    st.session_state.trained_model = None
                    st.session_state.model_info = {}
                    st.rerun()

        # Feature highlights
        st.markdown("<br>", unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)
        features = [
            ("📊", "Smart EDA", "Auto-charts, distributions, correlations & statistical profiling"),
            ("🤖", "AutoML Arena", "Train 6+ models, cross-validate & rank on a live leaderboard"),
            ("🔍", "Explainable AI", "SHAP values & feature importance for transparent decisions"),
            ("💬", "Data Chatbot", "Ask questions in plain English — powered by local LLM"),
        ]
        for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
            col.markdown(f"""<div class='feature-card'>
                <span class='feature-icon'>{icon}</span>
                <div class='feature-title'>{title}</div>
                <div class='feature-desc'>{desc}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 2: ANALYSIS DASHBOARD
# ══════════════════════════════════════════════
else:
    df = st.session_state.df

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 🚀 DataPilot AI")
        st.markdown(f"**📁 {st.session_state.get('filename', 'Dataset')}**")

        # Quality Score
        profile = get_dataset_profile(df)
        quality = profile["quality"]
        if quality >= 80:
            badge_cls = "quality-good"
            badge_icon = "✅"
        elif quality >= 50:
            badge_cls = "quality-warning"
            badge_icon = "⚠️"
        else:
            badge_cls = "quality-bad"
            badge_icon = "🔴"
        st.markdown(f"<span class='quality-badge {badge_cls}'>{badge_icon} Quality: {quality}/100</span>", unsafe_allow_html=True)

        st.caption(f"📐 {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.caption(f"💾 Memory: {profile['memory_mb']:.1f} MB")

        st.divider()

        # Data Export
        st.markdown("##### 📥 Export Data")
        export_format = st.selectbox("File format", ["CSV", "Excel"], key="export_format")
        if st.button("Prepare export", width="stretch", key="prepare_export"):
            fmt = "csv" if export_format == "CSV" else "excel"
            with st.spinner(f"Preparing {export_format} download…"):
                try:
                    st.session_state.export_payload = {
                        "data_id": id(df),
                        "format": export_format,
                        "data": get_df_download(df, fmt),
                    }
                except Exception as exc:
                    st.error(f"Could not create {export_format} export: {exc}")
        export_payload = st.session_state.get("export_payload")
        if export_payload and export_payload["data_id"] == id(df):
            is_csv = export_payload["format"] == "CSV"
            st.download_button(
                f"Download {export_payload['format']}",
                data=export_payload["data"],
                file_name=f"datapilot_export.{'csv' if is_csv else 'xlsx'}",
                mime="text/csv" if is_csv else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
                on_click="ignore",
            )

        st.divider()

        new_file = st.file_uploader("📂 Upload New Dataset", type=["csv", "xlsx", "json"], key="sidebar_upload")
        if new_file is not None:
            load_data(new_file)

        st.divider()
        if st.button("🔄 Reset App", type="secondary", width="stretch"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── Header ──
    st.markdown(f"""<div class='section-header' style='margin-top: 0;'>
        <span class='sh-icon'>📊</span>
        <h2 class='sh-text'>{safe_html_text(st.session_state.get('filename', 'Dataset'))}</h2>
        <span class='sh-badge'>Live Analysis</span>
    </div>""", unsafe_allow_html=True)

    # ── Metric Cards ──
    profile = get_dataset_profile(df)
    num_cols_list = profile["numeric_columns"]
    cat_cols_list = profile["categorical_columns"]
    missing_total = profile["missing_total"]
    dup_total = profile["duplicate_total"]

    metrics_html = f"""<div class='metric-grid'>
        {render_metric_card("📋", "Rows", f"{df.shape[0]:,}", "mc-blue")}
        {render_metric_card("📊", "Columns", str(df.shape[1]), "mc-violet")}
        {render_metric_card("🔢", "Numerical", str(len(num_cols_list)), "mc-emerald")}
        {render_metric_card("🏷️", "Categorical", str(len(cat_cols_list)), "mc-purple")}
        {render_metric_card("⚠️", "Missing", f"{missing_total:,}", "mc-amber" if missing_total > 0 else "mc-emerald")}
        {render_metric_card("🔁", "Duplicates", f"{dup_total:,}" if dup_total is not None else "Skipped", "mc-rose" if dup_total else "mc-amber" if dup_total is None else "mc-emerald")}
    </div>"""
    st.markdown(metrics_html, unsafe_allow_html=True)
    if profile["duplicate_scan_skipped"]:
        st.caption("Automatic duplicate scanning is skipped above 1 million rows to protect memory. You can still remove duplicates explicitly in Cleaning.")

    # ─── Workspace Navigation ───
    workspace_tabs = st.tabs(
        [label for label, _ in WORKSPACES],
        key="workspace_nav",
        on_change="rerun",
        default=WORKSPACES[0][0],
        width="stretch",
    )
    # TAB 1: OVERVIEW & DATA QUALITY
    # ═══════════════════════════════════
    if workspace_tabs[0].open:
        with workspace_tabs[0]:
            render_section_header("🎯", "Dataset Overview & Quality", "Profile")

            # Quality Score Breakdown
            q_score = compute_quality_score(df)
            total_cells = df.shape[0] * df.shape[1]
            missing_pct = (profile["missing_total"] / total_cells) * 100 if total_cells > 0 else 0
            dup_pct = (profile["duplicate_total"] / df.shape[0]) * 100 if df.shape[0] > 0 and profile["duplicate_total"] is not None else 0

            # Quality visual
            if q_score >= 80:
                ring_text_color = "var(--accent-emerald)"
            elif q_score >= 50:
                ring_text_color = "var(--accent-amber)"
            else:
                ring_text_color = "var(--accent-rose)"

            qc1, qc2, qc3, qc4 = st.columns(4)
            with qc1:
                st.markdown(f"""<div style='text-align: center;'>
                    <div class='quality-ring' style='background: conic-gradient({"#34d399" if q_score >= 80 else "#fbbf24" if q_score >= 50 else "#fb7185"} {q_score * 3.6}deg, rgba(30,41,59,0.5) 0deg);'>
                        <div class='quality-ring-inner'>
                            <span class='quality-ring-value' style='color: {ring_text_color}'>{q_score}</span>
                            <span class='quality-ring-label'>Quality</span>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
            with qc2:
                st.metric("⚠️ Missing Values", f"{missing_pct:.1f}%", delta=f"{missing_total:,} cells", delta_color="inverse")
            with qc3:
                duplicate_display = f"{profile['duplicate_total']:,}" if profile["duplicate_total"] is not None else "Not scanned"
                duplicate_delta = f"{dup_pct:.1f}%" if profile["duplicate_total"] is not None else "Large dataset"
                st.metric("🔁 Duplicate Rows", duplicate_display, delta=duplicate_delta, delta_color="inverse")
            with qc4:
                st.metric("📐 Data Types", f"{df.dtypes.nunique()}", delta=f"{len(num_cols_list)} num / {len(cat_cols_list)} cat")

            # Column profile is computed once and reused if users revisit Overview.
            render_section_header("📑", "Column Profile", f"{df.shape[1]} columns")
            col_info = get_overview_column_profile(df)
            st.dataframe(col_info, width="stretch", hide_index=True)

            # Preview
            render_section_header("👁️", "Data Preview", "First 10 rows")
            st.dataframe(df.head(10), width="stretch")

        # ═══════════════════════════════════
        # TAB 2: DATA CLEANING
        # ═══════════════════════════════════
    if workspace_tabs[1].open:
        with workspace_tabs[1]:
            render_section_header("🧹", "Data Preparation & Cleaning", "Transform")

            cl1, cl2, cl3 = st.columns(3)

            with cl1:
                st.markdown("#### 🩹 Missing Values")
                missing_columns = int((profile["missing_by_column"] > 0).sum())
                st.caption(f"{profile['missing_total']:,} missing cells across {missing_columns} columns")
                strategy = st.selectbox("Imputation Strategy", ["Mean", "Median", "Mode", "Drop Rows with Missing"], key="impute_strategy")
                if st.button("✅ Apply Imputation", width="stretch", key="btn_impute"):
                    if strategy == "Drop Rows with Missing":
                        before = len(df)
                        set_dataset(df.dropna())
                        after = len(st.session_state.df)
                        st.session_state.cleaning_log.append(f"Dropped {before - after} rows with missing values")
                    else:
                        # Create a copy to avoid inplace mutation issues
                        df_clean = df.copy()
                        for col in df_clean.columns:
                            if df_clean[col].isnull().sum() > 0:
                                if pd.api.types.is_numeric_dtype(df_clean[col]):
                                    if strategy == "Mean":
                                        val = df_clean[col].mean()
                                    elif strategy == "Median":
                                        val = df_clean[col].median()
                                    else:
                                        val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 0
                                    df_clean[col] = df_clean[col].fillna(val)
                                else:
                                    mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "Unknown"
                                    df_clean[col] = df_clean[col].fillna(mode_val)
                        set_dataset(df_clean)
                        st.session_state.cleaning_log.append(f"Imputed missing values using {strategy}")
                    st.success("✅ Done!")
                    st.rerun()

            with cl2:
                st.markdown("#### 🔁 Duplicates")
                dup_count = profile["duplicate_total"]
                if dup_count is None:
                    st.caption("Automatic scan was skipped for this large dataset.")
                else:
                    st.caption(f"{dup_count:,} duplicate rows found ({dup_count / max(len(df), 1) * 100:.1f}%)")
                if st.button("🗑️ Remove Duplicates", width="stretch", key="btn_dedup", disabled=(dup_count == 0)):
                    before = len(df)
                    set_dataset(df.drop_duplicates())
                    removed = before - len(st.session_state.df)
                    st.session_state.cleaning_log.append(f"Removed {removed} duplicate rows")
                    st.success(f"✅ Removed {removed} duplicates!")
                    st.rerun()

            with cl3:
                st.markdown("#### 📐 Outlier Treatment")
                if len(num_cols_list) > 0:
                    outlier_col = st.selectbox("Column", num_cols_list, key="outlier_col")
                    outlier_method = st.selectbox("Method", ["IQR Capping", "Z-Score Removal"], key="outlier_method")
                    if st.button("⚡ Treat Outliers", width="stretch", key="btn_outlier"):
                        df_out = df.copy()
                        if outlier_method == "IQR Capping":
                            Q1 = df_out[outlier_col].quantile(0.25)
                            Q3 = df_out[outlier_col].quantile(0.75)
                            IQR = Q3 - Q1
                            df_out[outlier_col] = df_out[outlier_col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
                        else:
                            col_data = df_out[outlier_col].dropna()
                            z_scores = np.abs(stats.zscore(col_data))
                            valid_indices = col_data.index[z_scores < 3]
                            df_out = df_out.loc[df_out.index.isin(valid_indices) | df_out[outlier_col].isna()]
                        set_dataset(df_out)
                        st.session_state.cleaning_log.append(f"Outlier treatment ({outlier_method}) on {outlier_col}")
                        st.success("✅ Outliers treated!")
                        st.rerun()
                else:
                    st.info("No numerical columns available for outlier treatment.")

            # Column Operations
            st.divider()
            render_section_header("🛠️", "Column Operations", "Advanced")
            cop1, cop2 = st.columns(2)

            with cop1:
                st.markdown("#### 🗑️ Drop Columns")
                drop_cols = st.multiselect("Select columns to drop", df.columns.tolist(), key="drop_cols")
                if st.button("Drop Selected", width="stretch", key="btn_drop_cols", disabled=(len(drop_cols) == 0)):
                    set_dataset(df.drop(columns=drop_cols))
                    st.session_state.cleaning_log.append(f"Dropped columns: {', '.join(drop_cols)}")
                    st.success(f"✅ Dropped {len(drop_cols)} columns!")
                    st.rerun()

            with cop2:
                st.markdown("#### 🔄 Convert Types")
                type_col = st.selectbox("Column", df.columns.tolist(), key="type_conv_col")
                new_type = st.selectbox("Convert to", ["numeric", "string", "datetime", "category"], key="new_type")
                if st.button("Convert", width="stretch", key="btn_convert"):
                    try:
                        df_conv = df.copy()
                        if new_type == "numeric":
                            df_conv[type_col] = pd.to_numeric(df_conv[type_col], errors='coerce')
                        elif new_type == "string":
                            df_conv[type_col] = df_conv[type_col].astype(str)
                        elif new_type == "datetime":
                            df_conv[type_col] = pd.to_datetime(df_conv[type_col], errors='coerce')
                        elif new_type == "category":
                            df_conv[type_col] = df_conv[type_col].astype('category')
                        set_dataset(df_conv)
                        st.session_state.cleaning_log.append(f"Converted '{type_col}' to {new_type}")
                        st.success(f"✅ Converted '{type_col}' to {new_type}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Conversion failed: {e}")

            # Audit Trail
            if st.session_state.cleaning_log:
                st.divider()
                render_section_header("📝", "Cleaning Audit Trail", f"{len(st.session_state.cleaning_log)} operations")
                for i, log in enumerate(st.session_state.cleaning_log, 1):
                    st.markdown(f"""<div class='audit-item'>
                        <span class='audit-num'>{i}</span>
                        <span class='audit-text'>{safe_html_text(log)}</span>
                    </div>""", unsafe_allow_html=True)

        # ═══════════════════════════════════
        # TAB 3: EDA & AUTO-CHARTS
        # ═══════════════════════════════════
    if workspace_tabs[2].open:
        with workspace_tabs[2]:
            render_section_header("📊", "Exploratory Data Analysis", "Discover")
            plot_df = df.sample(n=10_000, random_state=42) if len(df) > 10_000 else df
            if len(df) > len(plot_df):
                st.caption(f"Charts use a fixed 10,000-row sample for responsiveness. Summary statistics still use all {len(df):,} rows.")

            # Descriptive Stats
            st.markdown("#### 📊 Descriptive Statistics")
            desc = get_eda_describe(df)
            if 'mean' in desc.columns:
                for stat_col in ['mean', 'std', 'min', 'max', '25%', '50%', '75%']:
                    if stat_col in desc.columns:
                        desc[stat_col] = desc[stat_col].apply(lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int, float)) else "—")
            st.dataframe(desc.astype(str), width="stretch")

            # Skewness & Kurtosis
            if len(num_cols_list) > 0:
                with st.expander("📐 Distribution Shape — Skewness & Kurtosis", expanded=False):
                    shape_data = pd.DataFrame({
                        'Column': num_cols_list,
                        'Skewness': [round(df[c].skew(), 3) if pd.notna(df[c].skew()) else 0 for c in num_cols_list],
                        'Kurtosis': [round(df[c].kurtosis(), 3) if pd.notna(df[c].kurtosis()) else 0 for c in num_cols_list],
                        'Shape': [
                            "🔴 Highly Skewed" if abs(df[c].skew()) > 1 else "🟡 Moderate" if abs(df[c].skew()) > 0.5 else "🟢 Normal"
                            for c in num_cols_list
                        ]
                    })
                    st.dataframe(shape_data, width="stretch", hide_index=True)

            # Auto-Charts
            st.divider()
            render_section_header("📈", "Automated Visualizations", "AI-Selected")

            eda_col1, eda_col2 = st.columns([1, 3])
            with eda_col1:
                chart_col = st.selectbox("Select Column", df.columns, key="eda_chart_col")
                chart_type = auto_select_chart(df[chart_col], chart_col)
                st.markdown(f"<span class='stat-pill'>📊 Auto-selected: <b>{chart_type}</b></span>", unsafe_allow_html=True)

                # Quick column stats
                if pd.api.types.is_numeric_dtype(df[chart_col]):
                    st.markdown(f"""
                    <div style='margin-top: 12px; font-size: 0.82rem; color: var(--text-secondary); line-height: 1.8;'>
                        <b>Mean:</b> {df[chart_col].mean():.3f}<br>
                        <b>Median:</b> {df[chart_col].median():.3f}<br>
                        <b>Std:</b> {df[chart_col].std():.3f}<br>
                        <b>Min:</b> {df[chart_col].min():.3f}<br>
                        <b>Max:</b> {df[chart_col].max():.3f}
                    </div>""", unsafe_allow_html=True)
                else:
                    nuniq = df[chart_col].nunique()
                    top_val = df[chart_col].mode().iloc[0] if not df[chart_col].mode().empty else "N/A"
                    st.markdown(f"""
                    <div style='margin-top: 12px; font-size: 0.82rem; color: var(--text-secondary); line-height: 1.8;'>
                        <b>Unique:</b> {nuniq}<br>
                        <b>Top:</b> {safe_html_text(top_val)}<br>
                        <b>Missing:</b> {df[chart_col].isnull().sum()}
                    </div>""", unsafe_allow_html=True)

            with eda_col2:
                ecol1, ecol2 = st.columns(2)
                with ecol1:
                    if pd.api.types.is_numeric_dtype(df[chart_col]):
                        fig = px.histogram(plot_df, x=chart_col, marginal="box",
                            title=f"Distribution of {chart_col}",
                            color_discrete_sequence=["#818cf8"],
                            template="plotly_dark")
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter"),
                            title_font_size=14
                        )
                        st.plotly_chart(fig, width="stretch")
                    else:
                        vc = df[chart_col].value_counts().head(20).reset_index()
                        vc.columns = [chart_col, 'count']
                        fig = px.bar(vc, x=chart_col, y='count',
                            title=f"Frequency of {chart_col}",
                            color_discrete_sequence=["#38bdf8"],
                            template="plotly_dark")
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter"),
                            title_font_size=14
                        )
                        st.plotly_chart(fig, width="stretch")
                with ecol2:
                    if pd.api.types.is_numeric_dtype(df[chart_col]):
                        fig = px.violin(plot_df, y=chart_col, box=True, points="outliers",
                            title=f"Violin Plot — {chart_col}",
                            color_discrete_sequence=["#34d399"],
                            template="plotly_dark")
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter"),
                            title_font_size=14
                        )
                        st.plotly_chart(fig, width="stretch")
                    else:
                        vc = df[chart_col].value_counts().head(10).reset_index()
                        vc.columns = [chart_col, 'count']
                        fig = px.pie(vc, names=chart_col, values='count',
                            title=f"Distribution — {chart_col}",
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            template="plotly_dark")
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter"),
                            title_font_size=14
                        )
                        st.plotly_chart(fig, width="stretch")

            # AI Insight for Column
            insight_parts = []
            if df[chart_col].isnull().sum() > 0:
                insight_parts.append(f"This column has <b>{df[chart_col].isnull().sum()}</b> missing values ({df[chart_col].isnull().sum()/len(df)*100:.1f}%).")
            if pd.api.types.is_numeric_dtype(df[chart_col]):
                try:
                    skew_val = df[chart_col].skew()
                    if skew_val > 1:
                        insight_parts.append(f"It is <b>highly right-skewed</b> (skew={skew_val:.2f}), suggesting a concentration of lower values. Consider a log transformation.")
                    elif skew_val < -1:
                        insight_parts.append(f"It is <b>highly left-skewed</b> (skew={skew_val:.2f}).")
                    elif abs(skew_val) <= 0.5:
                        insight_parts.append(f"It follows an <b>approximately normal distribution</b> (skew={skew_val:.2f}).")
                    else:
                        insight_parts.append(f"It has <b>moderate skewness</b> (skew={skew_val:.2f}).")
                except Exception:
                    pass
            else:
                try:
                    mode_val = df[chart_col].mode().iloc[0] if not df[chart_col].mode().empty else "N/A"
                    top_count = df[chart_col].value_counts().iloc[0] if df[chart_col].value_counts().shape[0] > 0 else 0
                    insight_parts.append(f"The dominant category is '<b>{mode_val}</b>' appearing {top_count} times ({top_count/len(df)*100:.1f}% of data).")
                except Exception:
                    pass

            if insight_parts:
                st.markdown(f"""<div class='insight-card'>
                    <strong>💡 AI Insight for <code>{safe_html_text(chart_col)}</code>:</strong><br>
                    {safe_html_text(" ".join(insight_parts))}
                </div>""", unsafe_allow_html=True)

            # Correlation Heatmap
            if len(num_cols_list) >= 2:
                st.divider()
                render_section_header("🔥", "Correlation Heatmap", f"{len(num_cols_list)} features")

                corr_method = st.radio("Method", ["Pearson", "Spearman", "Kendall"], horizontal=True, key="corr_method")
                corr_df = df[num_cols_list]
                if len(corr_df) > 50_000:
                    corr_df = corr_df.sample(n=50_000, random_state=42)
                    st.caption("Correlation is estimated from a fixed 50,000-row sample for this large dataset.")
                corr = corr_df.corr(method=corr_method.lower())

                fig = px.imshow(corr, text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title=f"{corr_method} Correlation Matrix",
                    aspect="auto",
                    template="plotly_dark")
                fig.update_layout(
                    height=max(400, len(num_cols_list) * 35),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter")
                )
                st.plotly_chart(fig, width="stretch")

                # Strong correlations insight
                strong = []
                for i in range(len(corr.columns)):
                    for j in range(i + 1, len(corr.columns)):
                        val = corr.iloc[i, j]
                        if abs(val) > 0.7:
                            direction = "📈 Positive" if val > 0 else "📉 Negative"
                            strong.append(f"{direction}: <b>{safe_html_text(corr.columns[i])}</b> ↔ <b>{safe_html_text(corr.columns[j])}</b> = {val:.3f}")
                if strong:
                    insight_cls = "insight-warning" if any(abs(corr.iloc[i, j]) > 0.85 for i in range(len(corr.columns)) for j in range(i + 1, len(corr.columns))) else ""
                    st.markdown(f"""<div class='insight-card {insight_cls}'>
                        <strong>🔗 Strong Correlations Detected ({len(strong)}):</strong><br>
                        {"<br>".join(strong)}
                    </div>""", unsafe_allow_html=True)

            # Scatter Matrix
            if len(num_cols_list) >= 2:
                with st.expander("🔗 Pairwise Scatter Plots", expanded=False):
                    scatter_cols = st.multiselect(
                        "Select columns (max 5)",
                        num_cols_list,
                        default=num_cols_list[:min(3, len(num_cols_list))],
                        key="scatter_multi"
                    )
                    if len(scatter_cols) >= 2:
                        fig = px.scatter_matrix(
                            plot_df[scatter_cols[:5]],
                            dimensions=scatter_cols[:5],
                            color_discrete_sequence=["#818cf8"],
                            title="Scatter Matrix",
                            template="plotly_dark"
                        )
                        fig.update_layout(
                            height=600,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter")
                        )
                        st.plotly_chart(fig, width="stretch")

        # ═══════════════════════════════════
        # TAB 4: STATISTICAL TESTS
        # ═══════════════════════════════════
    if workspace_tabs[3].open:
        with workspace_tabs[3]:
            render_section_header("📈", "Statistical Hypothesis Testing", "Inference")

            test_type = st.selectbox("Select Test", [
                "Independent T-Test", "Paired T-Test", "ANOVA (One-Way)",
                "Mann-Whitney U", "Kruskal-Wallis",
                "Pearson Correlation", "Spearman Correlation",
                "Chi-Square Test", "Shapiro-Wilk (Normality)"
            ], key="stat_test_type")

            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

            if test_type == "Independent T-Test":
                if cat_cols and num_cols:
                    cat = st.selectbox("Grouping Variable (Binary)", cat_cols, key="tt_cat")
                    num = st.selectbox("Measurement Variable", num_cols, key="tt_num")
                    if st.button("▶️ Run T-Test", key="btn_ttest"):
                        groups = df[cat].dropna().unique()
                        if len(groups) == 2:
                            g1 = df[df[cat] == groups[0]][num].dropna()
                            g2 = df[df[cat] == groups[1]][num].dropna()
                            t_stat, p_val = stats.ttest_ind(g1, g2)

                            r1, r2, r3 = st.columns(3)
                            r1.metric("T-Statistic", f"{t_stat:.4f}")
                            r2.metric("P-Value", f"{p_val:.4e}")
                            # Cohen's d
                            pooled_std = np.sqrt((g1.std() ** 2 + g2.std() ** 2) / 2)
                            d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0
                            r3.metric("Cohen's d", f"{d:.3f}")

                            if p_val < 0.05:
                                st.success(f"✅ Statistically significant (p < 0.05). Meaningful difference between '{groups[0]}' and '{groups[1]}'.")
                            else:
                                st.warning(f"⚠️ Not significant (p ≥ 0.05). No meaningful difference found.")

                            fig = px.box(df, x=cat, y=num, color=cat, title=f"{num} by {cat}",
                                color_discrete_sequence=["#818cf8", "#f87171"], template="plotly_dark")
                            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"))
                            st.plotly_chart(fig, width="stretch")
                        else:
                            st.error(f"T-Test requires exactly 2 groups, but '{cat}' has {len(groups)}. Use ANOVA for 3+ groups.")
                else:
                    st.info("Need both categorical and numerical columns for this test.")

            elif test_type == "ANOVA (One-Way)":
                if cat_cols and num_cols:
                    cat = st.selectbox("Grouping Variable", cat_cols, key="anova_cat")
                    num = st.selectbox("Measurement Variable", num_cols, key="anova_num")
                    if st.button("▶️ Run ANOVA", key="btn_anova"):
                        groups_data = [group[num].dropna().values for name, group in df.groupby(cat) if len(group[num].dropna()) > 0]
                        if len(groups_data) >= 2:
                            f_stat, p_val = stats.f_oneway(*groups_data)
                            r1, r2 = st.columns(2)
                            r1.metric("F-Statistic", f"{f_stat:.4f}")
                            r2.metric("P-Value", f"{p_val:.4e}")
                            if p_val < 0.05:
                                st.success("✅ Significant difference across groups (p < 0.05).")
                            else:
                                st.warning("⚠️ No significant difference found.")
                            fig = px.box(df, x=cat, y=num, color=cat, title=f"ANOVA: {num} by {cat}", template="plotly_dark")
                            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"))
                            st.plotly_chart(fig, width="stretch")
                        else:
                            st.error("Need at least 2 non-empty groups for ANOVA.")
                else:
                    st.info("Need both categorical and numerical columns for this test.")

            elif test_type == "Pearson Correlation":
                if len(num_cols) >= 2:
                    v1 = st.selectbox("Variable 1", num_cols, key="p_v1")
                    v2 = st.selectbox("Variable 2", num_cols, index=min(1, len(num_cols) - 1), key="p_v2")
                    if st.button("▶️ Compute Pearson", key="btn_pearson"):
                        # Fix: align the two series before computing
                        clean_df = df[[v1, v2]].dropna()
                        if len(clean_df) > 2:
                            r, p = stats.pearsonr(clean_df[v1], clean_df[v2])
                            c1, c2 = st.columns(2)
                            c1.metric("Pearson r", f"{r:.4f}")
                            c2.metric("P-Value", f"{p:.4e}")
                            fig = px.scatter(clean_df, x=v1, y=v2, trendline="ols", title=f"Scatter: {v1} vs {v2}",
                                color_discrete_sequence=["#818cf8"], template="plotly_dark")
                            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"))
                            st.plotly_chart(fig, width="stretch")
                        else:
                            st.error("Not enough data points after removing missing values.")
                else:
                    st.info("Need at least 2 numerical columns.")

            elif test_type == "Spearman Correlation":
                if len(num_cols) >= 2:
                    v1 = st.selectbox("Variable 1", num_cols, key="s_v1")
                    v2 = st.selectbox("Variable 2", num_cols, index=min(1, len(num_cols) - 1), key="s_v2")
                    if st.button("▶️ Compute Spearman", key="btn_spearman"):
                        # Fix: align the two series before computing
                        clean_df = df[[v1, v2]].dropna()
                        if len(clean_df) > 2:
                            r, p = stats.spearmanr(clean_df[v1], clean_df[v2])
                            c1, c2 = st.columns(2)
                            c1.metric("Spearman ρ", f"{r:.4f}")
                            c2.metric("P-Value", f"{p:.4e}")
                            fig = px.scatter(clean_df, x=v1, y=v2, trendline="ols", title=f"Scatter: {v1} vs {v2}",
                                color_discrete_sequence=["#c084fc"], template="plotly_dark")
                            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"))
                            st.plotly_chart(fig, width="stretch")
                        else:
                            st.error("Not enough data points after removing missing values.")
                else:
                    st.info("Need at least 2 numerical columns.")

            elif test_type == "Chi-Square Test":
                if len(cat_cols) >= 2:
                    v1 = st.selectbox("Variable 1", cat_cols, key="chi_v1")
                    v2 = st.selectbox("Variable 2", cat_cols, index=min(1, len(cat_cols) - 1), key="chi_v2")
                    if st.button("▶️ Run Chi-Square", key="btn_chi"):
                        ct = pd.crosstab(df[v1], df[v2])
                        chi2, p, dof, expected = stats.chi2_contingency(ct)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Chi² Statistic", f"{chi2:.4f}")
                        c2.metric("P-Value", f"{p:.4e}")
                        c3.metric("Degrees of Freedom", dof)
                        if p < 0.05:
                            st.success("✅ Variables are significantly associated (p < 0.05).")
                        else:
                            st.warning("⚠️ No significant association found.")

                        # Heatmap of contingency table
                        fig = px.imshow(ct, text_auto=True, title=f"Contingency Table: {v1} vs {v2}",
                            color_continuous_scale="Blues", template="plotly_dark")
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"))
                        st.plotly_chart(fig, width="stretch")
                else:
                    st.info("Need at least 2 categorical columns.")

            elif test_type == "Shapiro-Wilk (Normality)":
                if num_cols:
                    col = st.selectbox("Column", num_cols, key="shapiro_col")
                    if st.button("▶️ Run Shapiro-Wilk", key="btn_shapiro"):
                        sample = df[col].dropna()
                        if len(sample) > 5000:
                            sample = sample.sample(5000, random_state=42)
                        if len(sample) >= 3:
                            stat_val, p = stats.shapiro(sample)
                            c1, c2 = st.columns(2)
                            c1.metric("W-Statistic", f"{stat_val:.4f}")
                            c2.metric("P-Value", f"{p:.4e}")
                            if p > 0.05:
                                st.success(f"✅ '{col}' appears normally distributed (p > 0.05).")
                            else:
                                st.warning(f"⚠️ '{col}' is NOT normally distributed (p < 0.05).")
                            fig = px.histogram(df, x=col, marginal="violin", title=f"Distribution: {col}",
                                color_discrete_sequence=["#818cf8"], template="plotly_dark")
                            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"))
                            st.plotly_chart(fig, width="stretch")
                        else:
                            st.error("Need at least 3 data points for Shapiro-Wilk test.")
                else:
                    st.info("Need numerical columns for this test.")

            elif test_type == "Mann-Whitney U":
                if cat_cols and num_cols:
                    cat = st.selectbox("Grouping Variable (Binary)", cat_cols, key="mw_cat")
                    num = st.selectbox("Measurement Variable", num_cols, key="mw_num")
                    if st.button("▶️ Run Mann-Whitney", key="btn_mw"):
                        groups = df[cat].dropna().unique()
                        if len(groups) == 2:
                            g1 = df[df[cat] == groups[0]][num].dropna()
                            g2 = df[df[cat] == groups[1]][num].dropna()
                            u, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                            c1, c2 = st.columns(2)
                            c1.metric("U-Statistic", f"{u:.1f}")
                            c2.metric("P-Value", f"{p:.4e}")
                            if p < 0.05:
                                st.success("✅ Significant difference between groups.")
                            else:
                                st.warning("⚠️ No significant difference found.")
                        else:
                            st.error(f"Mann-Whitney requires exactly 2 groups, found {len(groups)}.")
                else:
                    st.info("Need both categorical and numerical columns.")

            elif test_type == "Kruskal-Wallis":
                if cat_cols and num_cols:
                    cat = st.selectbox("Grouping Variable", cat_cols, key="kw_cat")
                    num = st.selectbox("Measurement Variable", num_cols, key="kw_num")
                    if st.button("▶️ Run Kruskal-Wallis", key="btn_kw"):
                        groups_data = [group[num].dropna().values for name, group in df.groupby(cat) if len(group[num].dropna()) > 0]
                        if len(groups_data) >= 2:
                            h, p = stats.kruskal(*groups_data)
                            c1, c2 = st.columns(2)
                            c1.metric("H-Statistic", f"{h:.4f}")
                            c2.metric("P-Value", f"{p:.4e}")
                            if p < 0.05:
                                st.success("✅ Significant difference across groups.")
                            else:
                                st.warning("⚠️ No significant difference found.")
                        else:
                            st.error("Need at least 2 non-empty groups.")
                else:
                    st.info("Need both categorical and numerical columns.")

            elif test_type == "Paired T-Test":
                if len(num_cols) >= 2:
                    v1 = st.selectbox("Before / Group 1", num_cols, key="pt_v1")
                    v2 = st.selectbox("After / Group 2", num_cols, index=min(1, len(num_cols) - 1), key="pt_v2")
                    if st.button("▶️ Run Paired T-Test", key="btn_paired"):
                        clean = df[[v1, v2]].dropna()
                        if len(clean) >= 3:
                            t, p = stats.ttest_rel(clean[v1], clean[v2])
                            c1, c2 = st.columns(2)
                            c1.metric("T-Statistic", f"{t:.4f}")
                            c2.metric("P-Value", f"{p:.4e}")
                            if p < 0.05:
                                st.success("✅ Significant difference between paired measurements.")
                            else:
                                st.warning("⚠️ No significant difference.")
                        else:
                            st.error("Not enough paired data points.")
                else:
                    st.info("Need at least 2 numerical columns.")

        # ═══════════════════════════════════
        # TAB 5: AUTOML & LEADERBOARD
        # ═══════════════════════════════════
    if workspace_tabs[4].open:
        with workspace_tabs[4]:
            render_section_header("🤖", "AutoML — Model Training Arena", "ML")

            # Target suggestion
            suggested = suggest_target(df)
            target = st.selectbox("🎯 Target Variable", df.columns,
                index=list(df.columns).index(suggested) if suggested in df.columns else 0, key="automl_target")

            feature_cols = [c for c in df.columns if c != target]
            features = st.multiselect("📊 Features (leave empty for all)", feature_cols, key="automl_feats")
            if not features:
                features = feature_cols

            # Auto-detect problem type
            problem_type = detect_problem_type(df[target])
            problem_override = st.radio("Problem Type", ["Classification", "Regression"],
                index=0 if problem_type == 'classification' else 1, horizontal=True, key="problem_type_radio")
            is_class = problem_override == "Classification"

            opt_col1, opt_col2, opt_col3 = st.columns(3)
            with opt_col1:
                cv_folds = st.slider("Cross-Validation Folds", 2, 5, 3, key="cv_folds")
            with opt_col2:
                test_size = st.slider("Test Set Size (%)", 10, 40, 20, key="test_size")
            with opt_col3:
                max_training_rows = st.selectbox("Training row limit", [5_000, 10_000, 25_000, 50_000], index=2, key="training_row_limit", format_func=lambda value: f"{value:,} rows")
            if not feature_cols:
                st.info("Add at least one feature column besides the target before training a model.")

            if st.button("🚀 Train All Models", width="stretch", type="primary", key="btn_train", disabled=not feature_cols):
                with st.spinner("🔄 Training multiple models with cross-validation..."):
                    ml_df = df[features + [target]].dropna()
                    if len(ml_df) > 10:
                        if len(ml_df) > max_training_rows:
                            ml_df = ml_df.sample(n=max_training_rows, random_state=42)
                            st.info(f"Training uses a reproducible {max_training_rows:,}-row sample to keep runtime and memory predictable. Evaluation metrics describe this sample.")
                        X = ml_df[features].copy()
                        y = ml_df[target].copy()

                        # Encode categoricals
                        label_encoders = {}
                        for col in X.columns:
                            if not pd.api.types.is_numeric_dtype(X[col]):
                                le = LabelEncoder()
                                X[col] = le.fit_transform(X[col].astype(str))
                                label_encoders[col] = le
                        # Ensure all numeric
                        X = X.apply(pd.to_numeric, errors='coerce')
                        X = X.fillna(0)

                        target_le = None
                        if is_class:
                            target_le = LabelEncoder()
                            y = target_le.fit_transform(y.astype(str))
                        else:
                            y = pd.to_numeric(y, errors='coerce')
                            valid_mask = y.notna()
                            X = X[valid_mask]
                            y = y[valid_mask].values

                        stratify = y if is_class and pd.Series(y).value_counts().min() >= 2 else None
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size / 100, random_state=42, stratify=stratify
                            )
                        except ValueError:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size / 100, random_state=42
                            )
                            st.warning("A stratified split was not possible for the selected sample; using a standard random split.")

                        # Models
                        # Optional estimators are imported only when AutoML is run, reducing
                        # the dashboard's initial startup time for users who only explore data.
                        try:
                            import xgboost as xgb
                            HAS_XGB = True
                        except ImportError:
                            HAS_XGB = False

                        if is_class:
                            models = {
                                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                                "K-Nearest Neighbors": KNeighborsClassifier(),
                                "Decision Tree": DecisionTreeClassifier(random_state=42),
                                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                                "Naive Bayes": GaussianNB(),
                            }
                            if HAS_XGB:
                                models["XGBoost"] = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
                            scoring = 'accuracy'
                        else:
                            models = {
                                "Linear Regression": LinearRegression(),
                                "Ridge Regression": Ridge(random_state=42),
                                "Lasso Regression": Lasso(random_state=42),
                                "Decision Tree": DecisionTreeRegressor(random_state=42),
                                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                            }
                            if HAS_XGB:
                                models["XGBoost"] = xgb.XGBRegressor(random_state=42, verbosity=0)
                            scoring = 'r2'

                        # Train & Evaluate
                        results = []
                        best_score = -np.inf
                        best_model = None
                        best_name = ""

                        progress = st.progress(0, text="Training models...")
                        status = st.empty()
                        for i, (name, model) in enumerate(models.items()):
                            status.caption(f"Training {name}...")
                            try:
                                cv_scores = cross_val_score(model, X_train, y_train, cv=min(cv_folds, len(X_train)), scoring=scoring)
                                model.fit(X_train, y_train)
                                preds = model.predict(X_test)

                                if is_class:
                                    acc = accuracy_score(y_test, preds)
                                    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                                    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
                                    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
                                    results.append({
                                        "Model": name, "Accuracy": f"{acc:.4f}",
                                        "Precision": f"{prec:.4f}", "Recall": f"{rec:.4f}",
                                        "F1-Score": f"{f1:.4f}",
                                        "CV Mean": f"{cv_scores.mean():.4f}", "CV Std": f"{cv_scores.std():.4f}",
                                        "_score": acc
                                    })
                                else:
                                    r2 = r2_score(y_test, preds)
                                    mae = mean_absolute_error(y_test, preds)
                                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                                    results.append({
                                        "Model": name, "R²": f"{r2:.4f}",
                                        "MAE": f"{mae:.4f}", "RMSE": f"{rmse:.4f}",
                                        "CV Mean": f"{cv_scores.mean():.4f}", "CV Std": f"{cv_scores.std():.4f}",
                                        "_score": r2
                                    })

                                if results[-1]["_score"] > best_score:
                                    best_score = results[-1]["_score"]
                                    best_model = model
                                    best_name = name
                            except Exception as e:
                                st.warning(f"⚠️ {name} failed: {str(e)[:100]}")

                            progress.progress((i + 1) / len(models), text=f"Trained {i + 1}/{len(models)} models")

                        status.empty()
                        progress.empty()

                        # Leaderboard
                        if results:
                            # Winner announcement
                            metric_name = 'Accuracy' if is_class else 'R²'
                            st.markdown(f"""<div class='lb-winner'>
                                <div class='lb-winner-title'>🏆 Best Model</div>
                                <div class='lb-winner-name'>{best_name}</div>
                                <div class='lb-winner-score'>{metric_name}: {best_score:.4f}</div>
                            </div>""", unsafe_allow_html=True)

                            st.markdown("<br>", unsafe_allow_html=True)

                            render_section_header("📊", "Model Leaderboard", f"{len(results)} models")
                            lb = pd.DataFrame(results).sort_values("_score", ascending=False).drop("_score", axis=1)
                            st.dataframe(lb, width="stretch", hide_index=True)

                            # Comparison chart
                            score_col = "Accuracy" if is_class else "R²"
                            chart_df = lb[["Model", score_col]].copy()
                            chart_df[score_col] = chart_df[score_col].astype(float)
                            fig = px.bar(chart_df, x="Model", y=score_col,
                                title=f"Model Comparison — {score_col}",
                                color=score_col,
                                color_continuous_scale="Viridis",
                                template="plotly_dark")
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(family="Inter"),
                                showlegend=False
                            )
                            st.plotly_chart(fig, width="stretch")

                            # Save best model
                            st.session_state.trained_model = best_model
                            st.session_state.model_info = {
                                "name": best_name, "features": list(X.columns),
                                "label_encoders": label_encoders, "target_le": target_le,
                                "target": target, "is_class": is_class,
                                "X_test": X_test, "y_test": y_test
                            }

                            # Confusion Matrix / Residuals
                            preds = best_model.predict(X_test)
                            if is_class:
                                cm_col1, cm_col2 = st.columns(2)
                                with cm_col1:
                                    st.markdown("#### Confusion Matrix")
                                    cm = confusion_matrix(y_test, preds)
                                    labels = target_le.classes_ if target_le else None
                                    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                        title=f"Confusion Matrix — {best_name}",
                                        x=labels, y=labels, template="plotly_dark")
                                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"))
                                    st.plotly_chart(fig, width="stretch")
                                with cm_col2:
                                    # ROC Curve
                                    try:
                                        if hasattr(best_model, 'predict_proba'):
                                            proba = best_model.predict_proba(X_test)
                                            if proba.shape[1] == 2:
                                                fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
                                                auc = roc_auc_score(y_test, proba[:, 1])
                                                st.markdown("#### ROC Curve")
                                                fig = px.area(x=fpr, y=tpr,
                                                    title=f"ROC Curve (AUC = {auc:.3f})",
                                                    labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                                                    template="plotly_dark")
                                                fig.add_shape(type='line', x0=0, x1=1, y0=0, y1=1,
                                                    line=dict(dash='dash', color='gray'))
                                                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"))
                                                st.plotly_chart(fig, width="stretch")
                                    except Exception as e:
                                        st.caption(f"ROC curve unavailable: {str(e)[:50]}")
                            else:
                                res_col1, res_col2 = st.columns(2)
                                with res_col1:
                                    st.markdown("#### Actual vs Predicted")
                                    fig = px.scatter(x=y_test, y=preds,
                                        labels={'x': 'Actual', 'y': 'Predicted'},
                                        title=f"Actual vs Predicted — {best_name}",
                                        color_discrete_sequence=["#818cf8"],
                                        template="plotly_dark")
                                    fig.add_shape(type='line',
                                        x0=min(y_test), x1=max(y_test),
                                        y0=min(y_test), y1=max(y_test),
                                        line=dict(dash='dash', color='#f87171'))
                                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"))
                                    st.plotly_chart(fig, width="stretch")
                                with res_col2:
                                    st.markdown("#### Residual Distribution")
                                    residuals = y_test - preds
                                    fig = px.histogram(x=residuals, title="Residual Distribution",
                                        labels={'x': 'Residual'},
                                        color_discrete_sequence=["#34d399"],
                                        template="plotly_dark")
                                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"))
                                    st.plotly_chart(fig, width="stretch")

                            # Feature Importance / SHAP
                            render_section_header("🔍", "Feature Importance", "Explainability")
                            try:
                                import shap
                                import matplotlib
                                matplotlib.use('Agg')
                                import matplotlib.pyplot as plt
                                explainer = shap.TreeExplainer(best_model)
                                shap_values = explainer.shap_values(X_test)
                                fig, ax = plt.subplots(figsize=(10, 6))
                                if is_class and isinstance(shap_values, list):
                                    shap.summary_plot(shap_values[1] if len(shap_values) > 1 else shap_values[0], X_test, show=False)
                                else:
                                    shap.summary_plot(shap_values, X_test, show=False)
                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception:
                                # Fallback: built-in feature importance
                                if hasattr(best_model, 'feature_importances_'):
                                    feat_names = list(X.columns)
                                    importances = best_model.feature_importances_
                                    # Ensure lengths match
                                    n = min(len(feat_names), len(importances))
                                    imp = pd.DataFrame({
                                        'Feature': feat_names[:n],
                                        'Importance': importances[:n]
                                    }).sort_values('Importance', ascending=True)
                                    fig = px.bar(imp, x='Importance', y='Feature', orientation='h',
                                        title="Feature Importance",
                                        color='Importance',
                                        color_continuous_scale="Viridis",
                                        template="plotly_dark")
                                    fig.update_layout(
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        font=dict(family="Inter"),
                                        showlegend=False
                                    )
                                    st.plotly_chart(fig, width="stretch")
                                elif hasattr(best_model, 'coef_'):
                                    feat_names = list(X.columns)
                                    coefs = best_model.coef_.flatten() if best_model.coef_.ndim > 1 else best_model.coef_
                                    n = min(len(feat_names), len(coefs))
                                    imp = pd.DataFrame({
                                        'Feature': feat_names[:n],
                                        'Coefficient': np.abs(coefs[:n])
                                    }).sort_values('Coefficient', ascending=True)
                                    fig = px.bar(imp, x='Coefficient', y='Feature', orientation='h',
                                        title="Feature Coefficients (Absolute)",
                                        color='Coefficient',
                                        color_continuous_scale="Viridis",
                                        template="plotly_dark")
                                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"), showlegend=False)
                                    st.plotly_chart(fig, width="stretch")
                                else:
                                    st.info("Feature importance is not available for this model type.")
                    else:
                        st.error("Not enough data (need >10 rows) after dropping missing values. Try cleaning your data first.")

        # ═══════════════════════════════════
        # TAB 6: PREDICTION PLAYGROUND
        # ═══════════════════════════════════
    if workspace_tabs[5].open:
        with workspace_tabs[5]:
            render_section_header("🔮", "Prediction Playground", "Inference")

            if st.session_state.trained_model is None:
                st.markdown("""<div class='empty-state'>
                    <span class='es-icon'>🤖</span>
                    <div class='es-title'>No Model Trained Yet</div>
                    <div class='es-desc'>Head over to the AutoML tab to train models first. Once trained, you can make predictions here with custom inputs!</div>
                </div>""", unsafe_allow_html=True)
            else:
                info = st.session_state.model_info
                st.markdown(f"""<div class='insight-card insight-success'>
                    <strong>✅ Model Ready:</strong> Using <b>{safe_html_text(info['name'])}</b> trained on target <b>{safe_html_text(info['target'])}</b>
                    ({'Classification' if info['is_class'] else 'Regression'})
                </div>""", unsafe_allow_html=True)

                st.markdown("#### Enter Feature Values")
                input_data = {}
                cols_per_row = 3
                feature_list = info['features']
                for i in range(0, len(feature_list), cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for j, col_name in enumerate(feature_list[i:i + cols_per_row]):
                        with row_cols[j]:
                            if col_name in info['label_encoders']:
                                le = info['label_encoders'][col_name]
                                options = list(le.classes_)
                                val = st.selectbox(f"🏷️ {col_name}", options, key=f"pred_{col_name}")
                                input_data[col_name] = le.transform([val])[0]
                            else:
                                default_val = 0.0
                                try:
                                    if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
                                        med = df[col_name].median()
                                        if pd.notna(med) and np.isfinite(med):
                                            default_val = float(med)
                                except Exception:
                                    pass
                                input_data[col_name] = st.number_input(f"🔢 {col_name}", value=default_val, key=f"pred_{col_name}")

                if st.button("🎯 Make Prediction", width="stretch", type="primary", key="btn_predict"):
                    try:
                        input_df = pd.DataFrame([input_data])
                        pred = st.session_state.trained_model.predict(input_df)[0]

                        if info['is_class'] and info.get('target_le') is not None:
                            pred_label = info['target_le'].inverse_transform([int(pred)])[0]
                            st.markdown(f"""<div class='lb-winner'>
                                <div class='lb-winner-title'>Prediction Result</div>
                                <div class='lb-winner-name'>{safe_html_text(pred_label)}</div>
                            </div>""", unsafe_allow_html=True)

                            if hasattr(st.session_state.trained_model, 'predict_proba'):
                                proba = st.session_state.trained_model.predict_proba(input_df)[0]
                                classes = info['target_le'].classes_ if info.get('target_le') else [f"Class {i}" for i in range(len(proba))]
                                prob_df = pd.DataFrame({'Class': classes, 'Probability': proba})
                                prob_df = prob_df.sort_values('Probability', ascending=True)
                                fig = px.bar(prob_df, x='Probability', y='Class', orientation='h',
                                    title="Prediction Confidence",
                                    color='Probability',
                                    color_continuous_scale="Viridis",
                                    template="plotly_dark")
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(family="Inter"),
                                    showlegend=False
                                )
                                st.plotly_chart(fig, width="stretch")
                        elif info['is_class']:
                            # Classification but no label encoder
                            st.markdown(f"""<div class='lb-winner'>
                                <div class='lb-winner-title'>Prediction Result</div>
                                <div class='lb-winner-name'>{safe_html_text(pred)}</div>
                            </div>""", unsafe_allow_html=True)
                        else:
                            # Regression
                            st.markdown(f"""<div class='lb-winner'>
                                <div class='lb-winner-title'>Predicted Value</div>
                                <div class='lb-winner-name'>{pred:.4f}</div>
                            </div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

        # ═══════════════════════════════════
        # TAB 7: ANOMALY DETECTION
        # ═══════════════════════════════════
    if workspace_tabs[6].open:
        with workspace_tabs[6]:
            render_section_header("🔍", "Anomaly & Outlier Detection", "Detect")

            if len(num_cols_list) == 0:
                st.markdown("""<div class='empty-state'>
                    <span class='es-icon'>🔢</span>
                    <div class='es-title'>No Numerical Columns</div>
                    <div class='es-desc'>Anomaly detection requires numerical columns. Try converting some columns to numeric in the Cleaning tab.</div>
                </div>""", unsafe_allow_html=True)
            else:
                method = st.selectbox("Detection Method", [
                    "IQR (Interquartile Range)",
                    "Z-Score",
                    "Isolation Forest (Multivariate)"
                ], key="anomaly_method")

                if method == "IQR (Interquartile Range)":
                    col = st.selectbox("Column", num_cols_list, key="iqr_col")
                    if st.button("🔍 Detect IQR Outliers", key="btn_iqr"):
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR_val = Q3 - Q1
                        outlier_mask = (df[col] < Q1 - 1.5 * IQR_val) | (df[col] > Q3 + 1.5 * IQR_val)
                        outlier_count = int(outlier_mask.sum())

                        st.markdown(f"""<div class='insight-card insight-warning'>
                            <strong>⚠️ Outliers Found:</strong> <b>{outlier_count:,}</b> outliers detected ({outlier_count / max(len(df), 1) * 100:.1f}% of data)
                        </div>""", unsafe_allow_html=True)

                        box_df = df[[col]].sample(n=10_000, random_state=42) if len(df) > 10_000 else df[[col]]
                        fig = px.box(box_df, y=col, title=f"Box Plot — {col} (sampled for display)", points="outliers",
                            color_discrete_sequence=["#fb7185"], template="plotly_dark")
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"))
                        st.plotly_chart(fig, width="stretch")

                        if outlier_count > 0:
                            st.dataframe(df.loc[outlier_mask].head(100), width="stretch")

                elif method == "Z-Score":
                    col = st.selectbox("Column", num_cols_list, key="zscore_col")
                    threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1, key="z_threshold")
                    if st.button("🔍 Detect Z-Score Outliers", key="btn_zscore"):
                        col_data = df[col].dropna()
                        if len(col_data) > 100_000:
                            col_data = col_data.sample(n=100_000, random_state=42)
                            st.info("Z-Score is evaluated on a fixed 100,000-row sample for this large dataset.")
                        z = np.abs(stats.zscore(col_data))
                        outlier_mask = z > threshold
                        outlier_indices = col_data.index[outlier_mask]
                        outliers = df.loc[outlier_indices]

                        st.markdown(f"""<div class='insight-card insight-warning'>
                            <strong>⚠️ Outliers Found:</strong> <b>{len(outliers):,}</b> outliers detected in {len(col_data):,} evaluated rows ({len(outliers) / max(len(col_data), 1) * 100:.1f}% of evaluated data)
                        </div>""", unsafe_allow_html=True)

                        if len(outliers) > 0:
                            st.dataframe(outliers.head(100), width="stretch")

                elif method == "Isolation Forest (Multivariate)":
                    if len(num_cols_list) >= 2:
                        iso_cols = st.multiselect("Select Columns", num_cols_list,
                            default=num_cols_list[:min(4, len(num_cols_list))], key="iso_cols")
                        contamination = st.slider("Contamination Rate", 0.01, 0.15, 0.05, key="iso_contam")
                        if st.button("🔍 Run Isolation Forest", key="btn_isoforest") and len(iso_cols) >= 2:
                            clean_data = df[iso_cols].dropna()
                            if len(clean_data) > 20_000:
                                clean_data = clean_data.sample(n=20_000, random_state=42)
                                st.info("Isolation Forest is fitted on a fixed 20,000-row sample to keep memory and runtime predictable.")
                            if len(clean_data) > 10:
                                iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
                                iso_preds = iso.fit_predict(clean_data)
                                clean_data = clean_data.copy()
                                clean_data['Anomaly'] = ['🔴 Anomaly' if p == -1 else '🟢 Normal' for p in iso_preds]
                                anomalies = clean_data[clean_data['Anomaly'] == '🔴 Anomaly']

                                st.markdown(f"""<div class='insight-card insight-warning'>
                                    <strong>⚠️ Anomalies Found:</strong> <b>{len(anomalies)}</b> anomalies detected ({len(anomalies) / len(clean_data) * 100:.1f}% of data)
                                </div>""", unsafe_allow_html=True)

                                fig = px.scatter(clean_data, x=iso_cols[0], y=iso_cols[1], color='Anomaly',
                                    color_discrete_map={'🟢 Normal': '#34d399', '🔴 Anomaly': '#fb7185'},
                                    title="Anomaly Detection Scatter Plot",
                                    template="plotly_dark")
                                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"))
                                st.plotly_chart(fig, width="stretch")

                                if len(anomalies) > 0:
                                    st.dataframe(anomalies.drop('Anomaly', axis=1).head(100), width="stretch")
                            else:
                                st.error("Not enough data after removing missing values.")
                    else:
                        st.info("Need at least 2 numerical columns for Isolation Forest.")

        # ═══════════════════════════════════
        # TAB 8: AI INSIGHTS
        # ═══════════════════════════════════
    if workspace_tabs[7].open:
        with workspace_tabs[7]:
            render_section_header("💡", "AI-Generated Insights & Recommendations", "Intelligence")

            insights = []

            # Data Quality
            q = compute_quality_score(df)
            if q < 60:
                insights.append(("danger", "🔴", "Critical Data Quality Issue", f"Data quality score is only {q}/100. Heavy cleaning is needed before reliable analysis. Focus on missing values and duplicates first."))
            elif q < 80:
                insights.append(("warning", "🟡", "Moderate Data Quality", f"Data quality score is {q}/100. Some cleaning is recommended for better analysis results."))
            else:
                insights.append(("success", "🟢", "Excellent Data Quality", f"Data quality score is {q}/100. Your dataset is in great shape for analysis!"))

            # Missing value insights
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                worst = df[missing_cols].isnull().sum().idxmax()
                worst_pct = df[worst].isnull().sum() / len(df) * 100
                insights.append(("warning", "⚠️", "Missing Values Detected",
                    f"Column '{worst}' has the most missing values ({worst_pct:.1f}%). "
                    f"{'Consider dropping this column if >50% missing.' if worst_pct > 50 else 'Use mean/median imputation for numerical, mode for categorical.'}"))

            # Correlation insights
            if len(num_cols_list) >= 2:
                corr = df[num_cols_list].corr()
                high_corr_pairs = []
                for i in range(len(corr.columns)):
                    for j in range(i + 1, len(corr.columns)):
                        val = corr.iloc[i, j]
                        if abs(val) > 0.85:
                            high_corr_pairs.append((corr.columns[i], corr.columns[j], val))
                if high_corr_pairs:
                    pairs_str = "; ".join([f"'{a}' ↔ '{b}' (r={v:.2f})" for a, b, v in high_corr_pairs[:3]])
                    insights.append(("warning", "🔗", "High Multicollinearity",
                        f"{len(high_corr_pairs)} highly correlated pairs found: {pairs_str}. Consider removing redundant features to improve model stability."))

            # Skewness
            skewed_cols = []
            for col in num_cols_list:
                try:
                    skew = df[col].skew()
                    if abs(skew) > 2:
                        skewed_cols.append((col, skew))
                except Exception:
                    pass
            if skewed_cols:
                cols_str = ", ".join([f"'{c}' (skew={s:.1f})" for c, s in skewed_cols[:3]])
                insights.append(("", "📐", "Extreme Skewness", f"Columns with extreme skewness: {cols_str}. Apply log, sqrt, or Box-Cox transformation."))

            # Class imbalance
            potential_targets = [c for c in df.columns if 1 < df[c].nunique() <= 10]
            for col in potential_targets[:3]:
                vc = df[col].value_counts(normalize=True)
                if vc.min() < 0.1:
                    insights.append(("warning", "⚖️", "Class Imbalance",
                        f"Column '{col}' has severe class imbalance (minority: {vc.min() * 100:.1f}%). Consider SMOTE, class weights, or stratified sampling."))

            # Constant columns
            const_cols = [c for c in df.columns if df[c].nunique() <= 1]
            if const_cols:
                insights.append(("danger", "🗑️", "Zero-Variance Columns",
                    f"Columns with only 1 unique value: {', '.join(const_cols)}. These provide no information and should be dropped."))

            # High cardinality
            for col in cat_cols_list:
                if df[col].nunique() > 50:
                    insights.append(("", "🏷️", "High Cardinality",
                        f"Column '{col}' has {df[col].nunique()} unique categories. Consider grouping rare categories or using target encoding for ML."))
                    break  # Only show once

            # Dataset size
            if len(df) < 100:
                insights.append(("warning", "📉", "Small Dataset",
                    "Dataset has fewer than 100 rows. Statistical tests and ML models may not be reliable. Collect more data if possible."))
            elif len(df) > 100000:
                insights.append(("", "📈", "Large Dataset",
                    f"Dataset has {len(df):,} rows. Consider sampling for initial EDA. Models will benefit from the large training set."))

            # Display insights
            if insights:
                for cls, icon, title, desc in insights:
                    card_cls = f"insight-{cls}" if cls else ""
                    st.markdown(f"""<div class='insight-card {card_cls}'>
                        <strong>{icon} {safe_html_text(title)}</strong><br>{safe_html_text(desc)}
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("✅ No significant issues detected. Your dataset looks great!")

        # ═══════════════════════════════════
        # TAB 9: AI CHATBOT
        # ═══════════════════════════════════
    if workspace_tabs[8].open:
        with workspace_tabs[8]:
            st.markdown("""<div class='chat-hero'>
                <div class='chat-hero-icon'>✦</div>
                <div class='chat-hero-copy'>
                    <p class='chat-hero-kicker'>Private, local AI</p>
                    <h2 class='chat-hero-title'>Chat with your dataset</h2>
                    <p class='chat-hero-subtitle'>Ask a question in plain language and explore your data with confidence.</p>
                </div>
                <span class='chat-model-chip'>🔒 ON DEVICE</span>
            </div>""", unsafe_allow_html=True)

            import requests as req
            import json

            OLLAMA_URL = "http://localhost:11434"
            MODEL_NAME = "llama3.2"

            # Briefly cache local model discovery so reruns don't repeatedly wait on HTTP.
            models_list = get_local_models(OLLAMA_URL)
            ollama_ok = models_list is not None
            models_list = models_list or []

            if not ollama_ok:
                st.markdown("""<div class='chat-status chat-status-offline'>
                    <strong>Ollama is offline.</strong> Start Ollama to enable local chat.
                </div>""", unsafe_allow_html=True)
                st.code("ollama serve\n# Then in another terminal tab:\nollama pull llama3.2", language="bash")
            elif MODEL_NAME not in [m.split(":")[0] for m in models_list]:
                st.markdown(f"""<div class='chat-status chat-status-warn'>
                    <strong>{MODEL_NAME} is not installed.</strong> Add it with <code>ollama pull {MODEL_NAME}</code>.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='chat-status chat-status-ready'>
                    <strong>Ready to chat</strong> · {MODEL_NAME} is running locally. Your dataset stays on this device.
                </div>""", unsafe_allow_html=True)

                chat_container = st.container(
                    key="chat-history", height=460, border=True, autoscroll=True
                )

                with chat_container:
                    if len(st.session_state.messages) == 0:
                        st.markdown("""<div class='chat-welcome'>
                            <span class='chat-welcome-icon'>🤖</span>
                            <div class='chat-welcome-copy'>
                                <p class='chat-welcome-title'>Ask a question about this dataset</p>
                                <p class='chat-welcome-desc'>Get quick, private answers from the summary statistics on this device.</p>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        st.caption("START WITH A QUESTION")
                        suggestion_columns = st.columns(3)
                        suggestions = (
                            ("Dataset shape", "How many rows and columns are in this dataset?"),
                            ("Missing values", "Which columns contain missing values?"),
                            ("Numeric summary", "Summarize the numeric columns."),
                        )
                        for index, (column, (label, suggestion)) in enumerate(zip(suggestion_columns, suggestions)):
                            with column:
                                st.button(
                                    label,
                                    key=f"chat_suggestion_{index}",
                                    width="stretch",
                                    on_click=queue_chat_prompt,
                                    args=(suggestion,),
                                )
                    for msg in st.session_state.messages:
                        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
                        with st.chat_message(msg["role"], avatar=avatar):
                            st.markdown(msg["content"])

                prompt = st.chat_input(
                    "Ask anything about your dataset...",
                    key="dataset_chat_input",
                    width="stretch",
                )
                if not prompt:
                    prompt = st.session_state.pop("pending_chat_prompt", None)
                if prompt:
                    st.session_state.pending_chat_prompt = None
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with chat_container:
                        with st.chat_message("user", avatar="🧑‍💻"):
                            st.write(prompt)

                        with st.chat_message("assistant", avatar="🤖"):
                            dataset_summary = get_chat_data_summary(df)

                            system = dedent(f"""\
                                You are a strict, factual Data Assistant.
                                You must ONLY use the exact statistics provided below to answer the question.
                                DO NOT perform any calculations. DO NOT guess or infer beyond the data.
                                If the answer is not in the data below, say: "I cannot determine this from the available summary statistics."
                                Keep your answer concise (2-3 sentences max). Use bullet points for multiple items.
                                {dataset_summary["sampling_note"]}

                                DATASET: {st.session_state.get('filename', 'Unknown')}
                                Shape: {df.shape[0]} rows × {df.shape[1]} columns
                                Missing values: {dataset_summary["missing"]}

                                COLUMN STATISTICS:
                                {dataset_summary["columns"]}
                                """)

                            recent_messages = [
                                {"role": message["role"], "content": message["content"]}
                                for message in st.session_state.messages[-12:]
                            ]
                            messages = [{"role": "system", "content": system}, *recent_messages]

                            payload = {
                                "model": MODEL_NAME,
                                "messages": messages,
                                "stream": True,
                                "options": {
                                    "temperature": 0.0,
                                    "top_p": 0.1
                                }
                            }

                            try:
                                container = st.empty()
                                response_parts = []
                                last_render = time.monotonic()
                                with req.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=60) as r:
                                    for line in r.iter_lines():
                                        if line:
                                            chunk = json.loads(line)
                                            token = chunk.get("message", {}).get("content", "")
                                            if token:
                                                response_parts.append(token)
                                            now = time.monotonic()
                                            if now - last_render >= 0.08 and response_parts:
                                                container.markdown("".join(response_parts) + "▌")
                                                last_render = now
                                            if chunk.get("done"):
                                                break
                                response = "".join(response_parts)
                                container.markdown(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                st.session_state.messages = st.session_state.messages[-40:]
                            except req.exceptions.ConnectionError:
                                st.error("Lost connection to Ollama. Is it still running?")
                            except req.exceptions.Timeout:
                                st.error("Request timed out. The model might be loading — try again.")
                            except Exception as e:
                                st.error(f"Error: {str(e)[:200]}")
