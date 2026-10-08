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


# ─── Page Config ───
st.set_page_config(page_title="DataPilot AI", page_icon="🚀", layout="wide")

# ─── Custom CSS for Premium Look ───
st.markdown("""
<style>
    html, body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    
    :root {
        --card-bg: rgba(30, 41, 59, 0.7);
        --card-bg-hover: rgba(30, 41, 59, 0.95);
        --card-border: rgba(255, 255, 255, 0.1);
        --card-border-hover: #38bdf8;
        --card-title: #94a3b8;
        --card-val: #f8fafc;
        --card-shadow: rgba(0,0,0,0.4);
        --card-shadow-hover: rgba(56, 189, 248, 0.25);
        --insight-bg: rgba(30, 41, 59, 0.7);
        --insight-text: #f8fafc;
        --feature-bg: rgba(30, 41, 59, 0.7);
        --feature-title: #e2e8f0;
        --glass-blur: blur(12px);
    }
    
    @media (prefers-color-scheme: light) {
        :root {
            --card-bg: rgba(255, 255, 255, 0.7);
            --card-bg-hover: rgba(255, 255, 255, 0.95);
            --card-border: rgba(0, 0, 0, 0.1);
            --card-border-hover: #38bdf8;
            --card-title: #64748b;
            --card-val: #0f172a;
            --card-shadow: rgba(0,0,0,0.05);
            --card-shadow-hover: rgba(56, 189, 248, 0.15);
            --insight-bg: rgba(255, 255, 255, 0.7);
            --insight-text: #334155;
            --feature-bg: rgba(255, 255, 255, 0.7);
            --feature-title: #0f172a;
        }
    }
    
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .metric-card, .hero-card, .feature-card, .insight-card {
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        animation: fadeUp 0.6s ease-out forwards;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }

    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px var(--card-shadow);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        background: var(--card-bg-hover);
        border-color: var(--card-border-hover);
        box-shadow: 0 10px 30px var(--card-shadow-hover);
    }
    .metric-card h3 { color: var(--card-title); font-size: 0.85rem; margin: 0; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h1 { color: var(--card-val); font-size: 2.2rem; margin: 5px 0 0 0; font-weight: 800; }
    
    .metric-blue h1 { background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-purple h1 { background: linear-gradient(135deg, #c084fc, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-amber h1 { background: linear-gradient(135deg, #fbbf24, #f59e0b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-green h1 { background: linear-gradient(135deg, #34d399, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-red h1 { background: linear-gradient(135deg, #f87171, #ef4444); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    .hero-card {
        background: var(--card-bg);
        padding: 35px;
        border-radius: 24px;
        border: 1px solid var(--card-border);
        box-shadow: 0 20px 40px var(--card-shadow);
    }
    
    .feature-card {
        text-align: center;
        background: var(--feature-bg);
        border-radius: 16px;
        padding: 25px 15px;
        border: 1px solid var(--card-border);
    }
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: var(--card-border-hover);
        background: var(--card-bg-hover);
        box-shadow: 0 10px 30px var(--card-shadow-hover);
    }
    .feature-title { color: var(--feature-title); font-weight: 700; margin: 10px 0 5px 0; font-size: 1.1rem; }
    
    .quality-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .quality-good { background: #065f46; color: #34d399; }
    .quality-warning { background: #78350f; color: #fbbf24; }
    .quality-bad { background: #7f1d1d; color: #f87171; }
    
    .insight-card {
        background: var(--insight-bg);
        color: var(--insight-text);
        border-left: 4px solid #818cf8;
        border-radius: 0 12px 12px 0;
        padding: 15px;
        margin: 8px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 16px; }
    
    div[data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 15px;
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] div {
        color: var(--card-val) !important;
    }
    /* ══════ Refined dashboard and responsive layout ══════ */
    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }
    .stApp {
        background: radial-gradient(ellipse at 12% 0%, #172440 0%, #0d1424 42%, #090e18 100%) !important;
    }
    .main .block-container {
        max-width: 1680px;
        padding: 1.75rem clamp(1rem, 3vw, 2.75rem) 3rem;
    }
    .metric-grid {
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 14px;
        margin: 16px 0 22px;
    }
    .metric-card {
        min-width: 0;
        min-height: 158px;
        padding: 20px 12px;
        border-radius: 20px;
        background: linear-gradient(150deg, rgba(27, 39, 60, .96), rgba(19, 29, 47, .96));
        border-color: rgba(148, 163, 184, .14);
        box-shadow: 0 8px 24px rgba(2, 6, 18, .18);
        animation: none;
        transition: border-color .2s ease, background .2s ease;
    }
    .metric-card:hover {
        transform: none;
        background: linear-gradient(150deg, rgba(32, 47, 72, .98), rgba(22, 34, 55, .98));
        border-color: rgba(96, 165, 250, .35);
        box-shadow: 0 10px 26px rgba(2, 6, 18, .24);
    }
    .metric-card .mc-icon { font-size: 1.65rem; margin-bottom: 10px; }
    .metric-card .mc-label { font-size: .7rem; letter-spacing: 1.2px; }
    .metric-card .mc-value {
        font-size: clamp(1.35rem, 2vw, 1.9rem);
        overflow-wrap: anywhere;
    }
    .section-header { animation: none; margin: 22px 0 14px; flex-wrap: wrap; }
    .section-header .sh-text { font-size: clamp(1.2rem, 2vw, 1.5rem); letter-spacing: -.02em; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        overflow-x: auto;
        scrollbar-width: thin;
        background: rgba(16, 25, 41, .82);
        border-color: rgba(148, 163, 184, .12);
    }
    .stTabs [data-baseweb="tab"] { white-space: nowrap; padding: 9px 14px; }
    .stTabs [aria-selected="true"] {
        background: rgba(96, 165, 250, .16) !important;
        color: #bfdbfe !important;
    }
    .stButton > button, [data-testid="stDownloadButton"] button {
        min-height: 42px;
        border: 1px solid rgba(148, 163, 184, .2) !important;
        transition: background .18s ease, border-color .18s ease !important;
    }
    .stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
        transform: none !important;
        border-color: rgba(96, 165, 250, .55) !important;
        box-shadow: 0 5px 16px rgba(2, 6, 18, .2) !important;
    }
    .stDataFrame, [data-testid="stTable"] {
        border: 1px solid rgba(148, 163, 184, .14);
        border-radius: 16px !important;
        overflow: hidden;
    }

    /* ══════ Chat experience ══════ */
    .chat-hero {
        display: flex;
        align-items: center;
        gap: 18px;
        margin: 8px 0 20px;
        padding: 22px 24px;
        border: 1px solid rgba(129, 140, 248, .18);
        border-radius: 22px;
        background: linear-gradient(110deg, rgba(25, 39, 62, .96), rgba(20, 29, 48, .82));
    }
    .chat-hero-icon {
        flex: 0 0 54px;
        width: 54px;
        height: 54px;
        display: grid;
        place-items: center;
        border: 1px solid rgba(129, 140, 248, .28);
        border-radius: 17px;
        background: linear-gradient(135deg, rgba(96, 165, 250, .2), rgba(192, 132, 252, .2));
        font-size: 1.55rem;
    }
    .chat-hero-copy { flex: 1; min-width: 0; }
    .chat-hero-kicker {
        margin: 0 0 4px;
        color: #8b9bb3;
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .chat-hero-title { margin: 0; color: #f1f5f9; font-size: 1.35rem; font-weight: 750; }
    .chat-hero-subtitle { margin: 4px 0 0; color: #9baec7; font-size: .88rem; }
    .chat-model-chip {
        flex: 0 0 auto;
        padding: 7px 11px;
        border: 1px solid rgba(52, 211, 153, .22);
        border-radius: 999px;
        background: rgba(16, 185, 129, .08);
        color: #6ee7b7;
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .06em;
    }
    .chat-status {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 0 0 16px;
        padding: 12px 15px;
        border: 1px solid rgba(148, 163, 184, .15);
        border-radius: 14px;
        background: rgba(17, 27, 45, .72);
        color: #b7c5d8;
        font-size: .88rem;
    }
    .chat-status-ready { border-color: rgba(52, 211, 153, .2); }
    .chat-status-warn { border-color: rgba(251, 191, 36, .24); }
    .chat-status-offline { border-color: rgba(251, 113, 133, .22); }
    .chat-welcome {
        max-width: 820px;
        margin: 26px auto;
        padding: clamp(28px, 5vw, 54px) 24px;
        background: linear-gradient(145deg, rgba(25, 39, 62, .96), rgba(17, 27, 45, .96));
        border: 1px solid rgba(129, 140, 248, .2);
        border-radius: 24px;
        box-shadow: 0 18px 48px rgba(2, 6, 18, .22);
        color: #a9b8ce;
    }
    .chat-welcome-icon {
        width: 64px;
        height: 64px;
        display: grid;
        place-items: center;
        margin: 0 auto 18px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(96, 165, 250, .18), rgba(192, 132, 252, .18));
        border: 1px solid rgba(129, 140, 248, .25);
        font-size: 2rem;
        animation: none;
    }
    [data-testid="stChatMessage"] {
        max-width: 900px;
        margin: 12px auto;
        padding: 16px 18px;
        border: 1px solid rgba(148, 163, 184, .12);
        border-radius: 18px;
        background: rgba(19, 29, 47, .78);
    }
    [data-testid="stChatMessage"] p { line-height: 1.7; }
    [data-testid="stChatInput"] {
        max-width: 940px;
        margin: 12px auto 0;
    }
    [data-testid="stChatInput"] textarea {
        border: 1px solid rgba(129, 140, 248, .28) !important;
        border-radius: 16px !important;
        background: rgba(17, 27, 45, .96) !important;
        line-height: 1.5;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(96, 165, 250, .75) !important;
        box-shadow: 0 0 0 3px rgba(96, 165, 250, .12) !important;
    }
    @media (max-width: 1250px) {
        .metric-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    @media (max-width: 700px) {
        .main .block-container { padding: 1rem .75rem 2rem; }
        .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
        .metric-card { min-height: 132px; padding: 16px 8px; }
        .stTabs [data-baseweb="tab"] { padding: 8px 10px; font-size: .78rem; }
        [data-testid="stChatMessage"] { padding: 12px; }
        .chat-hero { align-items: flex-start; padding: 17px; gap: 12px; flex-wrap: wrap; }
        .chat-model-chip { margin-left: 66px; }
        .chat-hero-title { font-size: 1.15rem; }
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───
if "df" not in st.session_state:
    st.session_state.df = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "model_info" not in st.session_state:
    st.session_state.model_info = {}
if "cleaning_log" not in st.session_state:
    st.session_state.cleaning_log = []

# ─── Helper Functions ───
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(file)
        elif file.name.endswith('.json'):
            st.session_state.df = pd.read_json(file)
        else:
            st.session_state.df = pd.read_excel(file)
        st.session_state.filename = file.name
        st.session_state.cleaning_log = []
        st.session_state.trained_model = None
        st.session_state.model_info = {}
        st.session_state.messages = []
        st.rerun()
    except Exception as e:
        st.error(f"Error loading file: {e}")

@st.cache_data(show_spinner=False)
def load_sample_dataset(name):
    \"\"\"Load and cache a sample dataset on demand.\"\"\"
    import seaborn as sns
    return sns.load_dataset(name)


def compute_quality_score(df):
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (df.isnull().sum().sum() / total_cells) * 100 if total_cells > 0 else 0
    dup_pct = (df.duplicated().sum() / df.shape[0]) * 100 if df.shape[0] > 0 else 0
    return max(0, min(100, int(100 - (missing_pct * 1.5) - (dup_pct * 2))))

def auto_select_chart(series, col_name):
    """Automatically select the best chart type based on column data."""
    if pd.api.types.is_numeric_dtype(series):
        skew = series.dropna().skew()
        nunique = series.nunique()
        if nunique <= 10:
            return "bar"
        elif abs(skew) > 1:
            return "box"
        else:
            return "histogram"
    else:
        nunique = series.nunique()
        if nunique <= 15:
            return "bar"
        else:
            return "treemap"

def detect_problem_type(y):
    """Section 18: Auto-detect classification vs regression."""
    if y.dtype == 'object' or str(y.dtype) == 'category':
        return 'classification'
    nunique = y.nunique()
    if nunique <= 20 and nunique / len(y) < 0.05:
        return 'classification'
    return 'regression'

def suggest_target(df):
    """Section 17: Auto-suggest likely target columns."""
    target_keywords = ['target', 'label', 'class', 'survived', 'outcome', 'result', 'y', 'price', 'salary', 'revenue']
    for col in df.columns:
        if col.lower().strip() in target_keywords:
            return col
    # Pick the last column as default
    return df.columns[-1]

# ══════════════════════════════════════
# PAGE 1: LANDING PAGE (Section 6)
# ══════════════════════════════════════
if st.session_state.df is None:
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        st.markdown("""
        <div style='text-align: center; padding: 40px 0 20px 0;'>
            <h1 style='font-size: 3.5rem; background: linear-gradient(135deg, #38BDF8, #818CF8, #C084FC); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 5px;'>🚀 DataPilot AI</h1>
            <p style='color: #94A3B8; font-size: 1.2rem; font-weight: 300;'>Upload. Analyze. Predict. Understand.</p>
            <p style='color: #64748B; font-size: 0.9rem;'>Your AI-Powered Data Scientist</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='hero-card'>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload your CSV, Excel, or JSON dataset", type=["csv", "xlsx", "xls", "json"])
        if uploaded_file is not None:
            load_data(uploaded_file)
            
        st.divider()
        st.markdown("<p style='text-align: center; color: #64748B;'>Or try it instantly with popular datasets:</p>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🚢 Titanic (Classification)", use_container_width=True):
                st.session_state.df = load_sample_dataset('titanic')
                st.session_state.filename = "titanic.csv"
                st.session_state.cleaning_log = []
                st.rerun()
        with c2:
            if st.button("🌸 Iris (Clustering/Class)", use_container_width=True):
                st.session_state.df = load_sample_dataset('iris')
                st.session_state.filename = "iris.csv"
                st.session_state.cleaning_log = []
                st.rerun()
        with c3:
            if st.button("🐧 Penguins (Multi-class)", use_container_width=True):
                st.session_state.df = load_sample_dataset('penguins')
                st.session_state.filename = "penguins.csv"
                st.session_state.cleaning_log = []
                st.rerun()
                
        c4, c5, c6 = st.columns(3)
        with c4:
            if st.button("💎 Diamonds (Regression)", use_container_width=True):
                st.session_state.df = load_sample_dataset('diamonds')
                st.session_state.filename = "diamonds.csv"
                st.session_state.cleaning_log = []
                st.rerun()
        with c5:
            if st.button("🏥 Breast Cancer (sklearn)", use_container_width=True):
                from sklearn.datasets import load_breast_cancer
                data = load_breast_cancer(as_frame=True)
                st.session_state.df = data.frame
                st.session_state.filename = "breast_cancer.csv"
                st.session_state.cleaning_log = []
                st.rerun()
        with c6:
            if st.button("🏠 California Housing", use_container_width=True):
                from sklearn.datasets import fetch_california_housing
                data = fetch_california_housing(as_frame=True)
                st.session_state.df = data.frame
                st.session_state.filename = "california_housing.csv"
                st.session_state.cleaning_log = []
                st.rerun()
                
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown("<br>", unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)
        features = [
            ("📊", "Auto EDA", "Smart charts & stats"),
            ("🤖", "AutoML", "Multi-model training"),
            ("🔍", "SHAP", "Explainable AI"),
            ("💬", "AI Chat", "Ask your dataset")
        ]
        for col, (icon, title, desc) in zip([f1,f2,f3,f4], features):
            col.markdown(f"""<div class='feature-card'>
                <div style='font-size:2rem;'>{icon}</div>
                <div class='feature-title'>{title}</div>
                <div style='color:#64748b; font-size:0.8rem;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════
# PAGE 2: ANALYSIS DASHBOARD
# ══════════════════════════════════════
else:
    df = st.session_state.df
    
    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 🚀 DataPilot AI")
        st.success(f"**{st.session_state.get('filename', 'Dataset')}**")
        
        # Quick stats
        quality = compute_quality_score(df)
        if quality >= 80:
            badge = "quality-good"
        elif quality >= 50:
            badge = "quality-warning"
        else:
            badge = "quality-bad"
        st.markdown(f"Data Quality: <span class='quality-badge {badge}'>{quality}/100</span>", unsafe_allow_html=True)
        
        st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
        st.caption(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        st.divider()
        new_file = st.file_uploader("Upload New Dataset", type=["csv", "xlsx", "json"], key="sidebar_upload")
        if new_file is not None:
            load_data(new_file)
        
        st.divider()
        if st.button("🔄 Reset App", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── Header ──
    st.markdown(f"## 📊 {st.session_state.get('filename', 'Dataset')}")
    
    # ── Metric Cards (Section 8) ──
    num_cols_list = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_list = df.select_dtypes(exclude=np.number).columns.tolist()
    missing_total = df.isnull().sum().sum()
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Numerical", len(num_cols_list))
    m4.metric("Categorical", len(cat_cols_list))
    m5.metric("Missing", f"{missing_total:,}")
    
    # ── Main Tabs ──
    tabs = st.tabs([
        "📋 Overview & Quality",
        "🧹 Data Cleaning",
        "📊 EDA & Auto-Charts",
        "📈 Statistical Tests",
        "🤖 AutoML & Leaderboard",
        "🔮 Prediction Playground",
        "🔍 Anomaly Detection",
        "💡 AI Insights",
        "💬 AI Chatbot"
    ])
    
    # ═══════════════════════════════════
    # TAB 1: OVERVIEW & DATA QUALITY (Sections 8-10)
    # ═══════════════════════════════════
    with tabs[0]:
        st.header("Dataset Overview & Data Quality")
        
        # Quality Score Breakdown (Section 9)
        q_score = compute_quality_score(df)
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (df.isnull().sum().sum() / total_cells) * 100 if total_cells > 0 else 0
        dup_pct = (df.duplicated().sum() / df.shape[0]) * 100 if df.shape[0] > 0 else 0
        
        qc1, qc2, qc3 = st.columns(3)
        qc1.metric("🎯 Data Quality Score", f"{q_score}/100")
        qc2.metric("⚠️ Missing Values", f"{missing_pct:.1f}%")
        qc3.metric("🔁 Duplicate Rows", f"{df.duplicated().sum()} ({dup_pct:.1f}%)")
        
        # Column Info Table (Section 8)
        st.subheader("Column Profile")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str).values,
            'Non-Null Count': df.count().values,
            'Missing': df.isnull().sum().values,
            'Missing %': (df.isnull().sum() / len(df) * 100).round(1).values,
            'Unique Values': df.nunique().values,
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        # Preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

    # ═══════════════════════════════════
    # TAB 2: DATA CLEANING (Sections 10-12)
    # ═══════════════════════════════════
    with tabs[1]:
        st.header("Data Preparation & Cleaning")
        
        cl1, cl2, cl3 = st.columns(3)
        with cl1:
            st.subheader("Missing Values")
            strategy = st.selectbox("Imputation Strategy", ["Mean", "Median", "Mode", "Drop Rows with Missing"])
            if st.button("Apply Imputation", use_container_width=True):
                if strategy == "Drop Rows with Missing":
                    before = len(df)
                    st.session_state.df = df.dropna()
                    after = len(st.session_state.df)
                    st.session_state.cleaning_log.append(f"Dropped {before-after} rows with missing values")
                else:
                    for col in df.columns:
                        if df[col].isnull().sum() > 0:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                val = df[col].mean() if strategy == "Mean" else (df[col].median() if strategy == "Median" else df[col].mode()[0])
                                df[col].fillna(val, inplace=True)
                            else:
                                df[col].fillna(df[col].mode()[0], inplace=True)
                    st.session_state.df = df
                    st.session_state.cleaning_log.append(f"Imputed missing values using {strategy}")
                st.success("Done!")
                st.rerun()
                
        with cl2:
            st.subheader("Duplicates")
            st.write(f"**{df.duplicated().sum()}** duplicate rows found")
            if st.button("Remove Duplicates", use_container_width=True):
                before = len(df)
                st.session_state.df = df.drop_duplicates()
                removed = before - len(st.session_state.df)
                st.session_state.cleaning_log.append(f"Removed {removed} duplicate rows")
                st.success(f"Removed {removed} duplicates!")
                st.rerun()

        with cl3:
            st.subheader("Outlier Treatment")
            outlier_col = st.selectbox("Column", num_cols_list, key="outlier_col")
            outlier_method = st.selectbox("Method", ["IQR Capping", "Z-Score Removal"])
            if st.button("Treat Outliers", use_container_width=True):
                if outlier_method == "IQR Capping":
                    Q1 = df[outlier_col].quantile(0.25)
                    Q3 = df[outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    df[outlier_col] = df[outlier_col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
                else:
                    z = np.abs(stats.zscore(df[outlier_col].dropna()))
                    mask = z < 3
                    df = df.loc[df[outlier_col].dropna().index[mask]]
                st.session_state.df = df
                st.session_state.cleaning_log.append(f"Outlier treatment ({outlier_method}) on {outlier_col}")
                st.success("Outliers treated!")
                st.rerun()

        # Audit Trail (Section 12)
        if st.session_state.cleaning_log:
            st.subheader("🔍 Cleaning Audit Trail")
            for i, log in enumerate(st.session_state.cleaning_log, 1):
                st.write(f"{i}. {log}")

    # ═══════════════════════════════════
    # TAB 3: EDA & AUTO-CHARTS (Sections 13-15)
    # ═══════════════════════════════════
    with tabs[2]:
        st.header("Exploratory Data Analysis")
        
        # Descriptive Stats (Section 13)
        st.subheader("📊 Descriptive Statistics")
        desc = df.describe(include='all').T
        if 'mean' in desc.columns:
            for stat_col in ['mean', 'std', 'min', 'max']:
                if stat_col in desc.columns:
                    desc[stat_col] = desc[stat_col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        st.dataframe(desc.astype(str), use_container_width=True)
        
        # Additional Stats: Skewness & Kurtosis (Section 13)
        if len(num_cols_list) > 0:
            st.subheader("📐 Distribution Shape (Skewness & Kurtosis)")
            shape_data = pd.DataFrame({
                'Column': num_cols_list,
                'Skewness': [df[c].skew() for c in num_cols_list],
                'Kurtosis': [df[c].kurtosis() for c in num_cols_list],
                'Interpretation': [
                    "Highly Skewed" if abs(df[c].skew()) > 1 else "Moderately Skewed" if abs(df[c].skew()) > 0.5 else "Approximately Normal"
                    for c in num_cols_list
                ]
            })
            st.dataframe(shape_data, use_container_width=True, hide_index=True)
        
        # Auto-Charts (Section 14)
        st.subheader("📈 Automated Visualizations")
        st.write("Charts are automatically selected based on your column data type and distribution.")
        
        chart_col = st.selectbox("Select Column to Visualize", df.columns, key="eda_chart_col")
        chart_type = auto_select_chart(df[chart_col], chart_col)
        
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            if pd.api.types.is_numeric_dtype(df[chart_col]):
                st.plotly_chart(px.histogram(df, x=chart_col, marginal="box",
                    title=f"Distribution of {chart_col}",
                    color_discrete_sequence=["#818cf8"]), use_container_width=True)
            else:
                vc = df[chart_col].value_counts().head(20).reset_index()
                vc.columns = [chart_col, 'count']
                st.plotly_chart(px.bar(vc, x=chart_col, y='count',
                    title=f"Frequency of {chart_col}",
                    color_discrete_sequence=["#38bdf8"]), use_container_width=True)
        with ecol2:
            if pd.api.types.is_numeric_dtype(df[chart_col]):
                st.plotly_chart(px.box(df, y=chart_col, title=f"Box Plot — {chart_col}",
                    color_discrete_sequence=["#34d399"]), use_container_width=True)
            else:
                vc = df[chart_col].value_counts().head(10).reset_index()
                vc.columns = [chart_col, 'count']
                st.plotly_chart(px.pie(vc, names=chart_col, values='count',
                    title=f"Pie Chart — {chart_col}"), use_container_width=True)
        
        # Smart Insight for Column (Section 15)
        st.markdown(f"""<div class='insight-card'>
            <strong>💡 AI Insight for <code>{chart_col}</code>:</strong><br>
            {"This column has <b>" + str(df[chart_col].isnull().sum()) + "</b> missing values. " if df[chart_col].isnull().sum() > 0 else ""}
            {"It is <b>highly right-skewed</b> (skew=" + f"{df[chart_col].skew():.2f}" + "), suggesting a concentration of lower values with a long right tail. Consider a log transformation." if pd.api.types.is_numeric_dtype(df[chart_col]) and df[chart_col].skew() > 1 else ""}
            {"It is <b>highly left-skewed</b> (skew=" + f"{df[chart_col].skew():.2f}" + ")." if pd.api.types.is_numeric_dtype(df[chart_col]) and df[chart_col].skew() < -1 else ""}
            {"It follows an <b>approximately normal distribution</b>." if pd.api.types.is_numeric_dtype(df[chart_col]) and abs(df[chart_col].skew()) <= 0.5 else ""}
            {"The dominant category is '<b>" + str(df[chart_col].mode()[0]) + "</b>' appearing " + str(df[chart_col].value_counts().iloc[0]) + " times." if not pd.api.types.is_numeric_dtype(df[chart_col]) else ""}
        </div>""", unsafe_allow_html=True)
        
        # Correlation Heatmap (Section 14)
        if len(num_cols_list) >= 2:
            st.subheader("🔥 Correlation Heatmap")
            corr = df[num_cols_list].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                title="Feature Correlation Matrix", aspect="auto")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Strong correlations insight
            strong = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.7:
                        strong.append(f"**{corr.columns[i]}** ↔ **{corr.columns[j]}**: {val:.2f}")
            if strong:
                st.markdown(f"""<div class='insight-card'>
                    <strong>💡 Strong Correlations Detected:</strong><br>
                    {"<br>".join(strong)}
                </div>""", unsafe_allow_html=True)
        
        # Scatter Matrix for selected numerics
        if len(num_cols_list) >= 2:
            st.subheader("🔗 Pairwise Scatter Plots")
            scatter_cols = st.multiselect("Select columns (max 5)", num_cols_list, default=num_cols_list[:min(3, len(num_cols_list))], key="scatter_multi")
            if len(scatter_cols) >= 2:
                fig = px.scatter_matrix(df[scatter_cols[:5]], dimensions=scatter_cols[:5],
                    color_discrete_sequence=["#818cf8"], title="Scatter Matrix")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════
    # TAB 4: STATISTICAL TESTS (Section 16)
    # ═══════════════════════════════════
    with tabs[3]:
        st.header("Statistical Hypothesis Testing")
        
        test_type = st.selectbox("Select Test", [
            "Independent T-Test", "Paired T-Test", "ANOVA (One-Way)",
            "Mann-Whitney U", "Kruskal-Wallis",
            "Pearson Correlation", "Spearman Correlation",
            "Chi-Square Test", "Shapiro-Wilk (Normality)"
        ])
        
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        if test_type == "Independent T-Test":
            if cat_cols and num_cols:
                cat = st.selectbox("Grouping Variable (Binary)", cat_cols, key="tt_cat")
                num = st.selectbox("Measurement Variable", num_cols, key="tt_num")
                if st.button("Run T-Test"):
                    groups = df[cat].dropna().unique()
                    if len(groups) == 2:
                        g1 = df[df[cat]==groups[0]][num].dropna()
                        g2 = df[df[cat]==groups[1]][num].dropna()
                        t_stat, p_val = stats.ttest_ind(g1, g2)
                        
                        r1, r2 = st.columns(2)
                        r1.metric("T-Statistic", f"{t_stat:.4f}")
                        r2.metric("P-Value", f"{p_val:.4e}")
                        
                        # Effect size (Cohen's d)
                        d = (g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)
                        st.metric("Cohen's d (Effect Size)", f"{d:.3f}")
                        
                        if p_val < 0.05:
                            st.success(f"✅ Statistically significant (p < 0.05). There IS a meaningful difference between '{groups[0]}' and '{groups[1]}'.")
                        else:
                            st.warning(f"⚠️ Not significant (p ≥ 0.05). No meaningful difference found.")
                        
                        fig = px.box(df, x=cat, y=num, color=cat, title=f"{num} by {cat}",
                            color_discrete_sequence=["#818cf8", "#f87171"])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("T-Test requires exactly 2 groups. Use ANOVA for 3+ groups.")
        
        elif test_type == "ANOVA (One-Way)":
            if cat_cols and num_cols:
                cat = st.selectbox("Grouping Variable", cat_cols, key="anova_cat")
                num = st.selectbox("Measurement Variable", num_cols, key="anova_num")
                if st.button("Run ANOVA"):
                    groups_data = [group[num].dropna().values for name, group in df.groupby(cat)]
                    if len(groups_data) >= 2:
                        f_stat, p_val = stats.f_oneway(*groups_data)
                        r1, r2 = st.columns(2)
                        r1.metric("F-Statistic", f"{f_stat:.4f}")
                        r2.metric("P-Value", f"{p_val:.4e}")
                        if p_val < 0.05:
                            st.success("✅ Significant difference across groups (p < 0.05).")
                        else:
                            st.warning("⚠️ No significant difference found.")
                        fig = px.box(df, x=cat, y=num, color=cat, title=f"ANOVA: {num} by {cat}")
                        st.plotly_chart(fig, use_container_width=True)
                        
        elif test_type == "Pearson Correlation":
            if len(num_cols) >= 2:
                v1 = st.selectbox("Variable 1", num_cols, key="p_v1")
                v2 = st.selectbox("Variable 2", num_cols, index=1, key="p_v2")
                if st.button("Compute Pearson"):
                    r, p = stats.pearsonr(df[v1].dropna(), df[v2].dropna())
                    c1, c2 = st.columns(2)
                    c1.metric("Pearson r", f"{r:.4f}")
                    c2.metric("P-Value", f"{p:.4e}")
                    fig = px.scatter(df, x=v1, y=v2, trendline="ols", title=f"Scatter: {v1} vs {v2}",
                        color_discrete_sequence=["#818cf8"])
                    st.plotly_chart(fig, use_container_width=True)
                    
        elif test_type == "Spearman Correlation":
            if len(num_cols) >= 2:
                v1 = st.selectbox("Variable 1", num_cols, key="s_v1")
                v2 = st.selectbox("Variable 2", num_cols, index=1, key="s_v2")
                if st.button("Compute Spearman"):
                    r, p = stats.spearmanr(df[v1].dropna(), df[v2].dropna())
                    c1, c2 = st.columns(2)
                    c1.metric("Spearman ρ", f"{r:.4f}")
                    c2.metric("P-Value", f"{p:.4e}")
        
        elif test_type == "Chi-Square Test":
            if len(cat_cols) >= 2:
                v1 = st.selectbox("Variable 1", cat_cols, key="chi_v1")
                v2 = st.selectbox("Variable 2", cat_cols, index=min(1, len(cat_cols)-1), key="chi_v2")
                if st.button("Run Chi-Square"):
                    ct = pd.crosstab(df[v1], df[v2])
                    chi2, p, dof, expected = stats.chi2_contingency(ct)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Chi² Statistic", f"{chi2:.4f}")
                    c2.metric("P-Value", f"{p:.4e}")
                    c3.metric("Degrees of Freedom", dof)
                    if p < 0.05:
                        st.success("✅ Variables are significantly associated.")
                    else:
                        st.warning("⚠️ No significant association.")
                        
        elif test_type == "Shapiro-Wilk (Normality)":
            if num_cols:
                col = st.selectbox("Column", num_cols, key="shapiro_col")
                if st.button("Run Shapiro-Wilk"):
                    sample = df[col].dropna()
                    if len(sample) > 5000:
                        sample = sample.sample(5000, random_state=42)
                    stat, p = stats.shapiro(sample)
                    c1, c2 = st.columns(2)
                    c1.metric("W-Statistic", f"{stat:.4f}")
                    c2.metric("P-Value", f"{p:.4e}")
                    if p > 0.05:
                        st.success(f"✅ {col} appears normally distributed (p > 0.05).")
                    else:
                        st.warning(f"⚠️ {col} is NOT normally distributed (p < 0.05).")
                    fig = px.histogram(df, x=col, marginal="violin", title=f"Distribution: {col}",
                        color_discrete_sequence=["#818cf8"])
                    st.plotly_chart(fig, use_container_width=True)

        elif test_type == "Mann-Whitney U":
            if cat_cols and num_cols:
                cat = st.selectbox("Grouping Variable (Binary)", cat_cols, key="mw_cat")
                num = st.selectbox("Measurement Variable", num_cols, key="mw_num")
                if st.button("Run Mann-Whitney"):
                    groups = df[cat].dropna().unique()
                    if len(groups) == 2:
                        g1 = df[df[cat]==groups[0]][num].dropna()
                        g2 = df[df[cat]==groups[1]][num].dropna()
                        u, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                        c1, c2 = st.columns(2)
                        c1.metric("U-Statistic", f"{u:.1f}")
                        c2.metric("P-Value", f"{p:.4e}")
                        if p < 0.05:
                            st.success("✅ Significant difference between groups.")
                        else:
                            st.warning("⚠️ No significant difference found.")

        elif test_type == "Kruskal-Wallis":
            if cat_cols and num_cols:
                cat = st.selectbox("Grouping Variable", cat_cols, key="kw_cat")
                num = st.selectbox("Measurement Variable", num_cols, key="kw_num")
                if st.button("Run Kruskal-Wallis"):
                    groups_data = [group[num].dropna().values for name, group in df.groupby(cat)]
                    if len(groups_data) >= 2:
                        h, p = stats.kruskal(*groups_data)
                        c1, c2 = st.columns(2)
                        c1.metric("H-Statistic", f"{h:.4f}")
                        c2.metric("P-Value", f"{p:.4e}")
                        if p < 0.05:
                            st.success("✅ Significant difference across groups.")
                        else:
                            st.warning("⚠️ No significant difference found.")

        elif test_type == "Paired T-Test":
            if len(num_cols) >= 2:
                v1 = st.selectbox("Before/Group 1", num_cols, key="pt_v1")
                v2 = st.selectbox("After/Group 2", num_cols, index=1, key="pt_v2")
                if st.button("Run Paired T-Test"):
                    clean = df[[v1, v2]].dropna()
                    t, p = stats.ttest_rel(clean[v1], clean[v2])
                    c1, c2 = st.columns(2)
                    c1.metric("T-Statistic", f"{t:.4f}")
                    c2.metric("P-Value", f"{p:.4e}")
                    if p < 0.05:
                        st.success("✅ Significant difference between paired measurements.")
                    else:
                        st.warning("⚠️ No significant difference.")

    # ═══════════════════════════════════
    # TAB 5: AUTOML & LEADERBOARD (Sections 17-23)
    # ═══════════════════════════════════
    with tabs[4]:
        st.header("AutoML — Multi-Model Training & Evaluation")
        
        # Target suggestion (Section 17)
        suggested = suggest_target(df)
        target = st.selectbox("🎯 Target Variable", df.columns,
            index=list(df.columns).index(suggested) if suggested in df.columns else 0)
        
        feature_cols = [c for c in df.columns if c != target]
        features = st.multiselect("Features (leave empty for all)", feature_cols, key="automl_feats")
        if not features:
            features = feature_cols
        
        # Auto-detect problem type (Section 18)
        problem_type = detect_problem_type(df[target])
        problem_override = st.radio("Problem Type", ["Classification", "Regression"],
            index=0 if problem_type == 'classification' else 1, horizontal=True)
        is_class = problem_override == "Classification"
        
        cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5)
        
        if st.button("🚀 Train All Models", use_container_width=True):
            with st.spinner("Training multiple models with cross-validation..."):
                ml_df = df[features + [target]].dropna()
                if len(ml_df) > 10:
                    X = ml_df[features].copy()
                    y = ml_df[target].copy()
                    
                    # Encode categoricals
                    label_encoders = {}
                    for col in X.columns:
                        if not pd.api.types.is_numeric_dtype(X[col]):
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                            label_encoders[col] = le
                    X = X.apply(pd.to_numeric)
                    
                    target_le = None
                    if is_class:
                        target_le = LabelEncoder()
                        y = target_le.fit_transform(y.astype(str))
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Load XGBoost only when AutoML training is requested.
                    try:
                        import xgboost as xgb
                        HAS_XGB = True
                    except ImportError:
                        HAS_XGB = False

                    # Models (Section 20)
                    if is_class:
                        models = {
                            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
                            "KNN": KNeighborsClassifier(),
                            "Decision Tree": DecisionTreeClassifier(random_state=42),
                            "Random Forest": RandomForestClassifier(random_state=42),
                            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                        }
                        if HAS_XGB:
                            models["XGBoost"] = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
                        scoring = 'accuracy'
                    else:
                        models = {
                            "Linear Regression": LinearRegression(),
                            "Ridge": Ridge(random_state=42),
                            "Lasso": Lasso(random_state=42),
                            "Decision Tree": DecisionTreeRegressor(random_state=42),
                            "Random Forest": RandomForestRegressor(random_state=42),
                            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                        }
                        if HAS_XGB:
                            models["XGBoost"] = xgb.XGBRegressor(random_state=42, verbosity=0)
                        scoring = 'r2'
                    
                    # Train & Evaluate (Section 21-22)
                    results = []
                    best_score = -np.inf
                    best_model = None
                    best_name = ""
                    
                    progress = st.progress(0)
                    for i, (name, model) in enumerate(models.items()):
                        try:
                            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
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
                            st.warning(f"⚠️ {name} failed: {e}")
                        
                        progress.progress((i + 1) / len(models))
                    
                    # Leaderboard
                    if results:
                        st.subheader("🏆 Model Leaderboard")
                        lb = pd.DataFrame(results).sort_values("_score", ascending=False).drop("_score", axis=1)
                        st.dataframe(lb, use_container_width=True, hide_index=True)
                        
                        st.success(f"🥇 **Best Model: {best_name}** — {'Accuracy' if is_class else 'R²'}: {best_score:.4f}")
                        
                        # Save best model
                        st.session_state.trained_model = best_model
                        st.session_state.model_info = {
                            "name": best_name, "features": features,
                            "label_encoders": label_encoders, "target_le": target_le,
                            "target": target, "is_class": is_class,
                            "X_test": X_test, "y_test": y_test
                        }
                        
                        # Confusion Matrix / Residuals (Section 21)
                        preds = best_model.predict(X_test)
                        if is_class:
                            cm_col1, cm_col2 = st.columns(2)
                            with cm_col1:
                                st.subheader("Confusion Matrix")
                                cm = confusion_matrix(y_test, preds)
                                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                    title=f"Confusion Matrix — {best_name}")
                                st.plotly_chart(fig, use_container_width=True)
                            with cm_col2:
                                # ROC Curve
                                try:
                                    if hasattr(best_model, 'predict_proba'):
                                        proba = best_model.predict_proba(X_test)
                                        if proba.shape[1] == 2:
                                            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
                                            auc = roc_auc_score(y_test, proba[:, 1])
                                            fig = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC={auc:.3f})",
                                                labels={'x': 'FPR', 'y': 'TPR'})
                                            fig.add_shape(type='line', x0=0, x1=1, y0=0, y1=1,
                                                line=dict(dash='dash', color='gray'))
                                            st.plotly_chart(fig, use_container_width=True)
                                except:
                                    pass
                        else:
                            st.subheader("Actual vs Predicted")
                            fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'},
                                title=f"Actual vs Predicted — {best_name}",
                                color_discrete_sequence=["#818cf8"])
                            fig.add_shape(type='line', x0=min(y_test), x1=max(y_test),
                                y0=min(y_test), y1=max(y_test), line=dict(dash='dash', color='red'))
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature Importance / SHAP (Section 23)
                        st.subheader("🔍 Feature Importance (SHAP)")
                        try:
                            import shap
                            import matplotlib
                            matplotlib.use('Agg')
                            import matplotlib.pyplot as plt
                            explainer = shap.TreeExplainer(best_model)
                            shap_values = explainer.shap_values(X_test)
                            fig, ax = plt.subplots(figsize=(8, 5))
                            if is_class and isinstance(shap_values, list):
                                shap.summary_plot(shap_values[1], X_test, show=False)
                            else:
                                shap.summary_plot(shap_values, X_test, show=False)
                            st.pyplot(fig)
                        except:
                            # Fallback: built-in feature importance
                            if hasattr(best_model, 'feature_importances_'):
                                imp = pd.DataFrame({
                                    'Feature': features, 
                                    'Importance': best_model.feature_importances_
                                }).sort_values('Importance', ascending=True)
                                fig = px.bar(imp, x='Importance', y='Feature', orientation='h',
                                    title="Feature Importance", color_discrete_sequence=["#818cf8"])
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Not enough data after dropping missing values.")

    # ═══════════════════════════════════
    # TAB 6: PREDICTION PLAYGROUND (Section 24)
    # ═══════════════════════════════════
    with tabs[5]:
        st.header("🔮 Prediction Playground")
        
        if st.session_state.trained_model is None:
            st.info("👆 Train a model in the AutoML tab first, then come here to make predictions!")
        else:
            info = st.session_state.model_info
            st.success(f"Using trained **{info['name']}** model on target: **{info['target']}**")
            
            st.subheader("Enter Feature Values")
            input_data = {}
            cols_per_row = 3
            feature_list = info['features']
            for i in range(0, len(feature_list), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j, col_name in enumerate(feature_list[i:i+cols_per_row]):
                    with row_cols[j]:
                        if col_name in info['label_encoders']:
                            le = info['label_encoders'][col_name]
                            options = list(le.classes_)
                            val = st.selectbox(col_name, options, key=f"pred_{col_name}")
                            input_data[col_name] = le.transform([val])[0]
                        else:
                            default = float(df[col_name].median()) if pd.api.types.is_numeric_dtype(df[col_name]) else 0.0
                            input_data[col_name] = st.number_input(col_name, value=default, key=f"pred_{col_name}")
            
            if st.button("🎯 Predict", use_container_width=True):
                input_df = pd.DataFrame([input_data])
                pred = st.session_state.trained_model.predict(input_df)[0]
                
                if info['is_class'] and info['target_le']:
                    pred_label = info['target_le'].inverse_transform([int(pred)])[0]
                    st.markdown(f"### Prediction: **{pred_label}**")
                    
                    if hasattr(st.session_state.trained_model, 'predict_proba'):
                        proba = st.session_state.trained_model.predict_proba(input_df)[0]
                        classes = info['target_le'].classes_ if info['target_le'] else range(len(proba))
                        prob_df = pd.DataFrame({'Class': classes, 'Probability': proba})
                        fig = px.bar(prob_df, x='Class', y='Probability', title="Prediction Confidence",
                            color_discrete_sequence=["#818cf8"])
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown(f"### Predicted Value: **{pred:.4f}**")

    # ═══════════════════════════════════
    # TAB 7: ANOMALY DETECTION (Sections 11, 25)
    # ═══════════════════════════════════
    with tabs[6]:
        st.header("Anomaly & Outlier Detection")
        
        method = st.selectbox("Detection Method", ["IQR (Interquartile Range)", "Z-Score", "Isolation Forest (Multivariate)"])
        
        if method == "IQR (Interquartile Range)":
            col = st.selectbox("Column", num_cols_list, key="iqr_col")
            if st.button("Detect IQR Outliers"):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                st.error(f"Found **{len(outliers)}** outliers ({len(outliers)/len(df)*100:.1f}%)")
                
                fig = px.box(df, y=col, title=f"Box Plot with Outliers — {col}", points="outliers",
                    color_discrete_sequence=["#f87171"])
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(outliers, use_container_width=True)
                
        elif method == "Z-Score":
            col = st.selectbox("Column", num_cols_list, key="zscore_col")
            threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)
            if st.button("Detect Z-Score Outliers"):
                z = np.abs(stats.zscore(df[col].dropna()))
                outlier_mask = z > threshold
                outliers = df.loc[df[col].dropna().index[outlier_mask]]
                st.error(f"Found **{len(outliers)}** outliers ({len(outliers)/len(df)*100:.1f}%)")
                st.dataframe(outliers, use_container_width=True)
                
        elif method == "Isolation Forest (Multivariate)":
            if len(num_cols_list) >= 2:
                iso_cols = st.multiselect("Select Columns", num_cols_list, default=num_cols_list[:min(4, len(num_cols_list))], key="iso_cols")
                contamination = st.slider("Contamination", 0.01, 0.15, 0.05)
                if st.button("Run Isolation Forest") and len(iso_cols) >= 2:
                    clean = df[iso_cols].dropna()
                    iso = IsolationForest(contamination=contamination, random_state=42)
                    preds = iso.fit_predict(clean)
                    clean['Anomaly'] = ['Anomaly' if p == -1 else 'Normal' for p in preds]
                    anomalies = clean[clean['Anomaly'] == 'Anomaly']
                    st.error(f"Found **{len(anomalies)}** anomalies ({len(anomalies)/len(clean)*100:.1f}%)")
                    
                    if len(iso_cols) >= 2:
                        fig = px.scatter(clean, x=iso_cols[0], y=iso_cols[1], color='Anomaly',
                            color_discrete_map={'Normal': '#34d399', 'Anomaly': '#f87171'},
                            title="Anomaly Scatter Plot")
                        st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(anomalies.drop('Anomaly', axis=1), use_container_width=True)
            else:
                st.info("Need at least 2 numerical columns for Isolation Forest.")

    # ═══════════════════════════════════
    # TAB 8: AI INSIGHTS (Section 28)
    # ═══════════════════════════════════
    with tabs[7]:
        st.header("💡 AI-Generated Insights & Recommendations")
        
        insights = []
        
        # Data Quality Insights
        q = compute_quality_score(df)
        if q < 60:
            insights.append(("🔴", "Critical Data Quality Issue", f"Data quality score is only {q}/100. Heavy cleaning is needed before analysis."))
        elif q < 80:
            insights.append(("🟡", "Moderate Data Quality", f"Data quality score is {q}/100. Some cleaning recommended."))
        else:
            insights.append(("🟢", "Good Data Quality", f"Data quality score is {q}/100. Dataset is in good shape!"))
        
        # Missing value insights
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            worst = df[missing_cols].isnull().sum().idxmax()
            worst_pct = df[worst].isnull().sum() / len(df) * 100
            insights.append(("⚠️", "Missing Values", f"'{worst}' has the most missing values ({worst_pct:.1f}%). Consider imputation or dropping."))
        
        # Correlation insights
        if len(num_cols_list) >= 2:
            corr = df[num_cols_list].corr()
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.85:
                        insights.append(("🔗", "High Correlation", f"'{corr.columns[i]}' and '{corr.columns[j]}' are highly correlated (r={val:.2f}). Consider removing one to reduce multicollinearity."))
        
        # Skewness insights
        for col in num_cols_list:
            skew = df[col].skew()
            if abs(skew) > 2:
                insights.append(("📐", "Extreme Skewness", f"'{col}' is extremely skewed (skew={skew:.2f}). Apply log/sqrt transformation."))
        
        # Class imbalance (Section 36)
        potential_targets = [c for c in df.columns if df[c].nunique() < 10 and df[c].nunique() > 1]
        for col in potential_targets[:3]:
            vc = df[col].value_counts(normalize=True)
            if vc.min() < 0.1:
                insights.append(("⚖️", "Class Imbalance", f"'{col}' has severe class imbalance (minority class: {vc.min()*100:.1f}%). Consider SMOTE or class weights."))
        
        # Constant columns
        const_cols = [c for c in df.columns if df[c].nunique() <= 1]
        if const_cols:
            insights.append(("🗑️", "Constant Columns", f"Columns {const_cols} have only 1 unique value and provide no information. Drop them."))
        
        # Display insights
        for icon, title, desc in insights:
            st.markdown(f"""<div class='insight-card'>
                <strong>{icon} {title}</strong><br>{desc}
            </div>""", unsafe_allow_html=True)
        
        if not insights:
            st.success("✅ No significant issues detected. Your dataset looks great!")

    # ═══════════════════════════════════
    # TAB 9: AI CHATBOT (Section 27)
    # ═══════════════════════════════════
    with tabs[8]:
        st.markdown("""<div class='chat-hero'>
            <div class='chat-hero-icon'>✦</div>
            <div class='chat-hero-copy'>
                <p class='chat-hero-kicker'>Private, local AI</p>
                <h2 class='chat-hero-title'>Chat with your dataset</h2>
                <p class='chat-hero-subtitle'>Ask a question in plain language and explore your data with confidence.</p>
            </div>
            <span class='chat-model-chip'>● LLAMA 3.2</span>
        </div>""", unsafe_allow_html=True)
        
        import requests as req
        import json
        
        OLLAMA_URL = "http://localhost:11434"
        MODEL_NAME = "llama3.2"
        
        def get_models():
            try:
                r = req.get(f"{OLLAMA_URL}/api/tags", timeout=3)
                if r.status_code == 200:
                    return [m["name"] for m in r.json().get("models", [])]
            except Exception:
                return None
            return None
        
        # One local request gives both server status and installed model names.
        models = get_models()
        ollama_ok = models is not None
        models = models or []
        
        if not ollama_ok:
            st.markdown("""<div class='chat-status chat-status-offline'>
                <strong>Ollama is offline.</strong> Start Ollama to enable local chat.
            </div>""", unsafe_allow_html=True)
            st.code("ollama serve\n# Then in another tab:\nollama pull llama3.2", language="bash")
        elif MODEL_NAME not in [m.split(":")[0] for m in models]:
            st.markdown(f"""<div class='chat-status chat-status-warn'>
                <strong>{MODEL_NAME} is not installed.</strong> Add it with <code>ollama pull {MODEL_NAME}</code>.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='chat-status chat-status-ready'>
                <strong>Ready to chat</strong> · {MODEL_NAME} is running locally. Your dataset stays on this device.
            </div>""", unsafe_allow_html=True)
            
            chat_container = st.container()
            
            with chat_container:
                if len(st.session_state.messages) == 0:
                    st.markdown("""<div class='chat-welcome'>
                        <span class='chat-welcome-icon'>🤖</span>
                        <div style='font-size: 1.1rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 8px;'>Ask me about your data</div>
                        <div style='font-size: 0.88rem; color: var(--text-muted);'>Try “What is the average age?” or “Which column has the most missing values?”</div>
                    </div>""", unsafe_allow_html=True)
                for msg in st.session_state.messages:
                    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.markdown(msg["content"])

            prompt = st.chat_input("Ask anything about your dataset...")
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user", avatar="🧑‍💻"):
                        st.write(prompt)
                        
                    with st.chat_message("assistant", avatar="🤖"):
                        # Format stats dynamically for both numeric and categorical columns
                        stats_text = []
                        for col in df.columns:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                stats_text.append(f"- '{col}' (Numeric): Average={df[col].mean():.2f}, Median={df[col].median():.2f}, Min={df[col].min():.2f}, Max={df[col].max():.2f}, Skewness={df[col].skew():.2f}")
                            else:
                                vc = df[col].value_counts().to_dict()
                                vc_str = ", ".join([f"{str(k)}: {v}" for k, v in list(vc.items())[:5]])
                                stats_text.append(f"- '{col}' (Categorical/Text): {vc_str}")
                        
                        missing_stats = ", ".join([f"{col}: {df[col].isnull().sum()}" for col in df.columns if df[col].isnull().sum() > 0])
                        if not missing_stats: missing_stats = "None"
                        
                        system = f"""You are a strict, factual Data Assistant. 
You must ONLY use the exact statistics provided below to answer the question.
DO NOT perform any calculations. DO NOT guess.
If the answer is not in the data below, you MUST exactly say: "I cannot determine this from the summary."
Keep your answer very short (1 sentence).

DATASET STATISTICS:
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Missing values: {missing_stats}
{chr(10).join(stats_text)}
"""
                        
                        # We only pass the system prompt and the immediate user question to avoid small model context drift
                        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
                        
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
                            response = ""
                            with req.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=60) as r:
                                for line in r.iter_lines():
                                    if line:
                                        chunk = json.loads(line)
                                        token = chunk.get("message", {}).get("content", "")
                                        response += token
                                        container.markdown(response + "▌")
                                        if chunk.get("done"):
                                            break
                            container.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Error: {e}")
