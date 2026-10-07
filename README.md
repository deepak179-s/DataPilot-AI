# 🚀 DataPilot AI

![DataPilot AI Header](https://img.shields.io/badge/DataPilot_AI-100%25_Offline_Data_Science-2ea44f?style=for-the-badge&logo=rocket)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Machine_Learning-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn)
![Ollama](https://img.shields.io/badge/Local_AI-Ollama-black?style=for-the-badge&logo=ollama)

DataPilot AI is an end-to-end, **100% offline and highly secure** AI-powered data science platform. It automates the entire data pipeline from initial dataset upload and cleaning to complex exploratory data analysis (EDA), rigorous statistical testing, and predictive machine learning (AutoML)—all entirely within your local environment.

**No API keys. No data leaves your machine. Full data privacy.**

---

## ✨ Comprehensive Feature Suite

DataPilot AI is split into 9 dedicated workspaces (Tabs) designed to handle every stage of the Data Science lifecycle:

### 1. 📊 Overview & Quality
- Instantly generates metadata profiles of your dataset.
- Displays Data Types, Missing Value percentages, Unique Values, and Memory Usage.
- Detects severe class imbalances, completely empty columns, or constant variables.

### 2. 🧹 Smart Data Cleaning
- State-preserving data cleaning engine.
- Impute missing values smartly using Mean (for normal data), Median (for skewed data), or Mode (for categorical data).
- One-click duplicate row removal.
- Real-time audit trail logs exactly what was cleaned and when.

### 3. 📉 EDA & Auto-Charts
- **Smart Charting Engine:** Automatically determines the best Plotly visualization based on data types.
- Numeric vs Numeric: Interactive Scatter Plots.
- Numeric vs Categorical: Interactive Box Plots and Bar Charts.
- Categorical: Pie Charts and count plots.
- Includes correlation heatmaps for numeric variables.

### 4. 🧮 Advanced Statistical Tests
A robust, academic-grade statistical testing suite powered by `SciPy`:
- **Normality Tests:** Shapiro-Wilk test to check Gaussian distributions.
- **T-Tests (1-Sample & 2-Sample):** Compare means of continuous variables.
- **ANOVA:** Compare means across 3+ groups.
- **Chi-Square Test:** Determine relationships between categorical variables.

### 5. 🤖 AutoML & Leaderboard
- Automatically preprocesses your data (Label Encoding for categories, Standard Scaling for numerics).
- Trains **up to 8 models simultaneously** depending on whether your target is Classification or Regression (Random Forest, XGBoost, Gradient Boosting, SVM, Logistic Regression, Naive Bayes, etc.).
- Evaluates models using K-Fold Cross Validation.
- Generates a competitive **Leaderboard** ranking models by Accuracy (or R²).

### 6. 🎯 Prediction Playground
- Dynamically generates a beautiful, interactive input form based on your dataset's columns.
- Uses the *best performing model* from the AutoML leaderboard to instantly predict outcomes for new, hypothetical data you input on the fly.

### 7. 🔍 Explainable AI (XAI)
- Calculates **SHAP (SHapley Additive exPlanations)** values to demystify the black-box models.
- Generates Feature Importance charts so you know exactly *which* columns are driving the predictions.

### 8. 🚨 Anomaly Detection
- Unsupervised anomaly detection algorithms built-in.
- Supports **Isolation Forests**, **Z-Score analysis**, and **IQR (Interquartile Range)** outlier detection.
- Highlights anomalous rows directly in the dataset.

### 9. 💬 AI Chatbot (Powered by Ollama)
- A conversational interface built directly into your dashboard.
- Powered by a local **TinyLlama** model running via Ollama.
- Context-aware: The AI is fed the precise statistical summaries of your dataset and can answer questions about the data structure deterministically without making up numbers (zero hallucinations).

---

## 🛠️ Technology Stack

*   **Frontend Framework:** Streamlit (with Custom Premium CSS for dynamic hover effects, metric cards, and glassmorphism)
*   **Data Processing:** Pandas, NumPy
*   **Statistics:** SciPy
*   **Machine Learning:** Scikit-Learn, XGBoost
*   **Explainability:** SHAP
*   **Visualization:** Plotly, Seaborn, Matplotlib
*   **Local AI Engine:** Ollama API (`requests` based integration)

---

## 🚀 Getting Started

### 1. Prerequisites
You must have Python 3.9+ installed and [Ollama](https://ollama.com/) running locally for the AI features to work.

Pull the lightweight local LLM model:
```bash
ollama pull tinyllama
```

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/deepak179-s/DataPilot-AI.git
cd DataPilot-AI

# macOS/Linux: Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Windows: Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt
```

### 3. Run the Application
Launch the Streamlit dashboard from inside your activated virtual environment:

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`.

---

## 🔒 Security & Privacy Notice
DataPilot AI operates entirely locally. Your datasets (`.csv`, `.xlsx`) are parsed in your system's RAM, and the AI chatbot communicates strictly with your local `localhost:11434` Ollama server. **No data is transmitted over the internet.**
