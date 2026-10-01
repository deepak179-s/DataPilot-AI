import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
import scipy.stats as stats
import shap
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ─── Page Config ───
st.set_page_config(page_title="DataPilot AI", page_icon="🚀", layout="wide")

# ─── Custom CSS for Premium Look ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-card h3 { color: #94a3b8; font-size: 0.85rem; margin: 0; font-weight: 500; }
    .metric-card h1 { font-size: 2rem; margin: 5px 0 0 0; font-weight: 700; }
    .metric-blue h1 { color: #38bdf8; }
    .metric-purple h1 { color: #a78bfa; }
    .metric-amber h1 { color: #fbbf24; }
    .metric-green h1 { color: #34d399; }
    .metric-red h1 { color: #f87171; }
    
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
        background: #1e293b;
        border-left: 4px solid #818cf8;
        border-radius: 0 12px 12px 0;
        padding: 15px;
        margin: 8px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 15px;
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
        <div style='background: linear-gradient(135deg, #1E293B, #0F172A); padding: 30px; border-radius: 20px; border: 1px solid #334155; box-shadow: 0 20px 40px rgba(0,0,0,0.4);'>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload your CSV, Excel, or JSON dataset", type=["csv", "xlsx", "xls", "json"])
        if uploaded_file is not None:
            load_data(uploaded_file)
            
        st.divider()
        st.markdown("<p style='text-align: center; color: #64748B;'>Or try it instantly:</p>", unsafe_allow_html=True)
        if st.button("🚢 Load Titanic Sample Dataset", use_container_width=True):
            st.session_state.df = sns.load_dataset('titanic')
            st.session_state.filename = "titanic.csv"
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
            col.markdown(f"""<div style='text-align:center; background:#1e293b; border-radius:12px; padding:15px; border:1px solid #334155;'>
                <div style='font-size:2rem;'>{icon}</div>
                <div style='color:#e2e8f0; font-weight:600; margin:5px 0;'>{title}</div>
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
