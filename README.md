# DataPilot AI 🚀

DataPilot AI is an end-to-end, **100% offline and secure** AI-powered data science platform. It automates Data Cleaning, Exploratory Data Analysis (EDA), Statistical Testing, and Machine Learning (AutoML), entirely within your local environment without sending any sensitive data to external APIs.

## 🌟 Key Features

*   **100% Offline & Private:** Uses local Apple Silicon (MPS) acceleration and local LLMs via Ollama. No OpenAI keys or internet connection required for data processing.
*   **Intelligent Data Cleaning:** Automatically handles missing values (Mean/Median/Mode imputation), drops duplicates, and detects outliers (Z-Score, IQR, Isolation Forest).
*   **Auto-Charting (EDA):** Smartly selects the best charts (Histograms, Box Plots, Bar Charts, Scatter Plots) based on your data types using Plotly.
*   **Advanced Statistical Tests:** Built-in suite of 9 rigorous statistical tests (T-Test, ANOVA, Chi-Square, Shapiro-Wilk, Correlation matrices, etc.).
*   **AutoML & Leaderboard:** Train multiple models (Random Forest, XGBoost, Gradient Boosting, SVM, etc.) simultaneously with K-Fold cross-validation. View the best performing model on the leaderboard.
*   **Prediction Playground:** Dynamically generates prediction forms based on your trained models and dataset features.
*   **Explainable AI (XAI):** Built-in SHAP (SHapley Additive exPlanations) values and feature importance to understand *why* your model makes certain predictions.
*   **AI Chatbot:** Talk to your dataset! Powered by a local `tinyllama` model, the AI strictly analyzes your summary statistics to answer questions about the data structure.

## 🛠️ Tech Stack

*   **Frontend:** Streamlit (Custom Premium UI/CSS)
*   **Data Processing:** Pandas, NumPy, SciPy
*   **Machine Learning:** Scikit-Learn, XGBoost
*   **Explainability:** SHAP
*   **Visualization:** Plotly, Seaborn, Matplotlib
*   **Local AI Integration:** Ollama (`tinyllama` model)

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have Python 3.9+ installed and [Ollama](https://ollama.com/) running locally.

Pull the local LLM model for the AI chatbot feature:
```bash
ollama pull tinyllama
```

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/deepak179-s/DataPilot-AI.git
cd DataPilot-AI

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

### 3. Run the App
Launch the Streamlit dashboard from inside your virtual environment:

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`.
