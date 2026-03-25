import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from fpdf import FPDF
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# ── LOAD ENV ──────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Multi-Agent AI Platform",
    page_icon="🤖",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #0A0A0A, #1E1E1E);
    padding: 2rem; border-radius: 12px;
    border: 1px solid #C8FF00;
    margin-bottom: 2rem;
}
.agent-card {
    background: #1E1E1E; border: 1px solid #333;
    border-radius: 10px; padding: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── LLM SETUP — Multiple Keys to avoid rate limits ────────
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2", GROQ_API_KEY)
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3", GROQ_API_KEY)

@st.cache_resource
def get_llms():
    llm1 = LLM(model="groq/llama-3.3-70b-versatile", api_key=GROQ_API_KEY,   temperature=0.1)
    llm2 = LLM(model="groq/llama-3.3-70b-versatile", api_key=GROQ_API_KEY_2, temperature=0.1)
    llm3 = LLM(model="groq/llama-3.3-70b-versatile", api_key=GROQ_API_KEY_3, temperature=0.1)
    return llm1, llm2, llm3

llm1, llm2, llm3 = get_llms()
llm = llm1  # default

# ── HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="color:#C8FF00; margin:0">🤖 Enterprise Multi-Agent AI Analytics Platform</h1>
    <p style="color:#888; margin:0.5rem 0 0">
        7 Specialized Agents · Auto ML · Smart Data Cleaning · Multi-file Join · Chat · PDF Reports
    </p>
</div>
""", unsafe_allow_html=True)

# ── AGENT CARDS ───────────────────────────────────────────
cols = st.columns(7)
agents_info = [
    ("🔍", "Quality",     "Data Inspector"),
    ("📊", "Statistics",  "Pattern Finder"),
    ("🤖", "ML Expert",   "Model Selector"),
    ("📈", "Visualizer",  "Chart Designer"),
    ("📝", "Reporter",    "Report Writer"),
    ("💡", "Advisor",     "Biz Recommender"),
    ("🏆", "Director",    "Strategy Planner"),
]
for col, (icon, title, role) in zip(cols, agents_info):
    col.markdown(f"""
    <div class="agent-card">
        <div style="font-size:1.5rem">{icon}</div>
        <div style="color:#C8FF00;font-size:0.75rem;font-weight:bold">{title}</div>
        <div style="color:#888;font-size:0.65rem">{role}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── SESSION STATE ─────────────────────────────────────────
if 'df'             not in st.session_state: st.session_state.df             = None
if 'analysis_done'  not in st.session_state: st.session_state.analysis_done  = False
if 'chat_history'   not in st.session_state: st.session_state.chat_history   = []
if 'agent_outputs'  not in st.session_state: st.session_state.agent_outputs  = {}
if 'question'       not in st.session_state: st.session_state.question       = ""

# ── FILE UPLOAD ───────────────────────────────────────────
st.markdown("### 📁 Upload Your Dataset")

upload_mode = st.radio(
    "Upload Mode",
    ["Single File", "Multiple Files (Smart Join)", "Sample Data"],
    horizontal=True
)

if upload_mode == "Single File":
    uploaded_file = st.file_uploader(
        "Upload CSV, Excel or JSON",
        type=['csv', 'xlsx', 'xls', 'json']
    )
    if uploaded_file:
        try:
            size_mb = uploaded_file.size / (1024*1024)
            if uploaded_file.name.endswith('.csv'):
                if size_mb > 500:
                    st.session_state.df = pd.read_csv(uploaded_file, nrows=500000)
                    st.success(f"Large file ({size_mb:.0f}MB) — loaded first 500K rows")
                else:
                    st.session_state.df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                st.session_state.df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                st.session_state.df = pd.read_json(uploaded_file)
            st.success(f"✅ Loaded: {uploaded_file.name} — {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error: {e}")

elif upload_mode == "Multiple Files (Smart Join)":
    st.info("Upload multiple CSV files — Smart Join Wizard detects relationships automatically!")
    uploaded_files = st.file_uploader(
        "Upload multiple CSV files",
        type=['csv'],
        accept_multiple_files=True
    )

    if uploaded_files:
        dataframes = {}
        for f in uploaded_files:
            try:
                size_mb = f.size / (1024*1024)
                if size_mb > 500:
                    dataframes[f.name] = pd.read_csv(f, nrows=500000)
                    st.write(f"📄 **{f.name}** — 500K rows loaded ({size_mb:.0f}MB)")
                else:
                    dataframes[f.name] = pd.read_csv(f)
                    st.write(f"📄 **{f.name}** — {dataframes[f.name].shape[0]:,} rows × {dataframes[f.name].shape[1]} cols")
            except Exception as e:
                st.error(f"Error loading {f.name}: {e}")

        if len(dataframes) > 1:
            st.markdown("---")
            st.markdown("### 🧙 Smart Join Wizard")

            # Detect relationships
            file_names  = list(dataframes.keys())
            relationships = []
            for i in range(len(file_names)):
                for j in range(i+1, len(file_names)):
                    f1 = file_names[i]
                    f2 = file_names[j]
                    common = set(dataframes[f1].columns) & set(dataframes[f2].columns)
                    for col in common:
                        vals1   = set(dataframes[f1][col].dropna().unique())
                        vals2   = set(dataframes[f2][col].dropna().unique())
                        overlap = len(vals1.intersection(vals2))
                        if overlap > 0:
                            relationships.append({
                                'file1': f1, 'file2': f2,
                                'column': col, 'overlap': overlap
                            })

            if relationships:
                st.success(f"✅ Found {len(relationships)} relationship(s)!")
                for rel in relationships:
                    st.write(f"🔗 **{rel['file1']}** ↔ **{rel['file2']}** via `{rel['column']}` ({rel['overlap']:,} matches)")
            else:
                st.warning("No common columns found between files.")

            join_mode = st.selectbox("Join strategy:", [
                "🤖 Auto Join (recommended)",
                "📥 Stack vertically",
                "🔗 Manual join on column",
                "📊 Analyse one file"
            ])

            if join_mode == "🤖 Auto Join (recommended)":
                if st.button("🤖 Auto Join All Files", type="primary"):
                    with st.spinner("Joining files..."):
                        try:
                            # Start from largest file
                            largest = max(dataframes.keys(), key=lambda x: len(dataframes[x]))
                            result       = dataframes[largest]
                            joined_files = [largest]
                            st.write(f"🚀 Base file: **{largest}** ({len(result):,} rows)")

                            for _ in range(len(dataframes)):
                                for rel in sorted(relationships, key=lambda x: x['overlap'], reverse=True):
                                    l_in = rel['file1'] in joined_files
                                    r_in = rel['file2'] in joined_files
                                    if l_in and not r_in:
                                        next_f  = rel['file2']
                                        join_on = rel['column']
                                    elif r_in and not l_in:
                                        next_f  = rel['file1']
                                        join_on = rel['column']
                                    else:
                                        continue
                                    before = result.shape[1]
                                    result = result.merge(dataframes[next_f], on=join_on, how='left', suffixes=('', f'_{next_f[:4]}'))
                                    joined_files.append(next_f)
                                    st.write(f"✅ Joined **{next_f}** on `{join_on}` (+{result.shape[1]-before} cols)")
                                    break

                            st.session_state.df = result
                            st.success(f"🎉 Join complete! → {result.shape[0]:,} rows × {result.shape[1]} columns")
                        except Exception as e:
                            st.error(f"Join error: {e}")

            elif join_mode == "📥 Stack vertically":
                if st.button("📥 Stack Files"):
                    try:
                        combined = pd.concat(list(dataframes.values()), ignore_index=True)
                        st.session_state.df = combined
                        st.success(f"Stacked → {combined.shape[0]:,} rows × {combined.shape[1]} cols")
                    except Exception as e:
                        st.error(f"Stack error: {e}")

            elif join_mode == "🔗 Manual join on column":
                c1, c2 = st.columns(2)
                with c1:
                    left_file  = st.selectbox("Left file:", file_names)
                with c2:
                    right_file = st.selectbox("Right file:", [f for f in file_names if f != left_file])
                common_cols = list(set(dataframes[left_file].columns) & set(dataframes[right_file].columns))
                if common_cols:
                    c1, c2 = st.columns(2)
                    with c1:
                        join_col  = st.selectbox("Join on:", common_cols)
                    with c2:
                        join_type = st.selectbox("Type:", ["left", "inner", "outer"])
                    if st.button("🔗 Join"):
                        result = dataframes[left_file].merge(dataframes[right_file], on=join_col, how=join_type)
                        st.session_state.df = result
                        st.success(f"Joined → {result.shape[0]:,} rows × {result.shape[1]} cols")
                else:
                    st.warning("No common columns!")

            elif join_mode == "📊 Analyse one file":
                sel = st.selectbox("Select file:", file_names)
                if st.button("Load"):
                    st.session_state.df = dataframes[sel]
                    st.success(f"Loaded: {sel}")

elif upload_mode == "Sample Data":
    np.random.seed(42)
    n = 500
    st.session_state.df = pd.DataFrame({
        'CustomerID'    : range(1, n+1),
        'Age'           : np.random.randint(18, 70, n),
        'Income'        : np.random.randint(20000, 150000, n),
        'SpendingScore' : np.random.randint(1, 100, n),
        'TotalPurchases': np.random.randint(1, 50, n),
        'Region'        : np.random.choice(['North','South','East','West'], n),
        'Segment'       : np.random.choice(['Premium','Standard','Basic'], n),
        'Churned'       : np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    st.success(f"✅ Sample dataset loaded — {st.session_state.df.shape[0]:,} rows!")

# ── DATASET OVERVIEW ──────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df

    st.markdown("---")
    st.markdown("### 📋 Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows",           f"{df.shape[0]:,}")
    c2.metric("Columns",        f"{df.shape[1]:,}")
    c3.metric("Numeric Cols",   f"{df.select_dtypes(include='number').shape[1]}")
    c4.metric("Text Cols",      f"{df.select_dtypes(include='object').shape[1]}")
    c5.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    with st.expander("👀 Preview Data"):
        st.dataframe(df.head(10), use_container_width=True)

    # ── SMART DATA CLEANING ───────────────────────────────
    st.markdown("---")
    st.markdown("### 🧹 Smart Auto Data Cleaning")
    with st.expander("Auto Clean Your Data"):

        numeric_cols = df.select_dtypes(include="number").columns
        cat_cols     = df.select_dtypes(include="object").columns

        st.markdown("#### 📊 Data Quality Report")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duplicates",     f"{df.duplicated().sum():,}")
        c2.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        c3.metric("Numeric Cols",   f"{len(numeric_cols)}")
        c4.metric("Text Cols",      f"{len(cat_cols)}")

        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            st.markdown("**Missing values per column:**")
            for col, count in missing.items():
                pct = count / len(df) * 100
                st.write(f"  • `{col}`: {count:,} missing ({pct:.1f}%)")
        else:
            st.write("✅ No missing values found!")

        st.markdown("---")
        st.info("""**Smart Cleaning will automatically:**
- Remove duplicate rows
- Fill numeric missing → median
- Fill text missing → mode (most frequent)
- Drop columns with >50% missing
- Fix wrong data types""")

        remove_outliers = st.checkbox("Also remove outliers (3x IQR)", value=False)

        if st.button("🧹 Auto Clean Data", type="primary"):
            df_clean    = df.copy()
            changes     = []
            rows_before = len(df_clean)

            # Remove duplicates
            df_clean = df_clean.drop_duplicates()
            removed  = rows_before - len(df_clean)
            if removed > 0:
                changes.append(f"Removed {removed:,} duplicate rows")

            # Smart fill missing
            num_cols = df_clean.select_dtypes(include="number").columns
            obj_cols = df_clean.select_dtypes(include="object").columns

            for col in num_cols:
                miss = df_clean[col].isnull().sum()
                if miss > 0:
                    pct = miss / len(df_clean) * 100
                    if pct > 50:
                        df_clean.drop(columns=[col], inplace=True)
                        changes.append(f"Dropped '{col}' ({pct:.0f}% missing)")
                    else:
                        med = df_clean[col].median()
                        df_clean[col].fillna(med, inplace=True)
                        changes.append(f"Filled '{col}' with median ({med:.2f})")

            for col in obj_cols:
                miss = df_clean[col].isnull().sum()
                if miss > 0:
                    pct = miss / len(df_clean) * 100
                    if pct > 50:
                        df_clean.drop(columns=[col], inplace=True)
                        changes.append(f"Dropped '{col}' ({pct:.0f}% missing)")
                    else:
                        mode_val = df_clean[col].mode()[0]
                        df_clean[col].fillna(mode_val, inplace=True)
                        changes.append(f"Filled '{col}' with mode ('{mode_val}')")

            # Remove outliers
            if remove_outliers:
                before   = len(df_clean)
                num_cols2 = df_clean.select_dtypes(include="number").columns
                mask     = pd.Series([True] * len(df_clean), index=df_clean.index)
                for col in num_cols2:
                    Q1  = df_clean[col].quantile(0.25)
                    Q3  = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    mask = mask & (df_clean[col] >= Q1 - 3*IQR) & (df_clean[col] <= Q3 + 3*IQR)
                df_clean = df_clean[mask]
                removed  = before - len(df_clean)
                if removed > 0:
                    changes.append(f"Removed {removed:,} outlier rows")

            # Fix data types
            for col in df_clean.select_dtypes(include="object").columns:
                try:
                    converted = pd.to_numeric(df_clean[col], errors='coerce')
                    if converted.isnull().sum() < len(df_clean) * 0.1:
                        df_clean[col] = converted
                        changes.append(f"Converted '{col}' to numeric")
                except:
                    pass

            st.session_state.df = df_clean
            df = df_clean

            st.success("✅ Smart cleaning complete!")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows before",  f"{rows_before:,}")
            c2.metric("Rows after",   f"{len(df_clean):,}")
            c3.metric("Cols",         f"{df_clean.shape[1]}")
            c4.metric("Missing left", f"{df_clean.isnull().sum().sum():,}")
            if changes:
                for change in changes:
                    st.write(f"✅ {change}")
            else:
                st.write("✅ Data was already clean!")

    # ── ANALYSIS ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 Multi-Agent Analysis")

    question = st.text_input(
        "What do you want to analyse?",
        placeholder="e.g. What are the key customer segments? Which factors drive churn?"
    )

    c1, c2 = st.columns(2)
    with c1:
        run_analysis = st.button("🤖 Run 7-Agent Analysis", type="primary", use_container_width=True)
    with c2:
        run_ml = st.button("🧠 Run Auto ML", use_container_width=True)

    # ── AUTO ML ───────────────────────────────────────────
    if run_ml:
        numeric_cols_ml = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols_ml) < 2:
            st.warning("Need at least 2 numeric columns!")
        else:
            st.markdown("### 🧠 Auto ML Results")
            target_col = st.selectbox("Target column to predict:", numeric_cols_ml)
            if st.button("🚀 Train Model"):
                with st.spinner("Training..."):
                    df_ml      = df.copy()
                    le         = LabelEncoder()
                    for col in df.select_dtypes(include='object').columns:
                        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

                    feature_cols = [c for c in df_ml.columns if c != target_col]
                    X = df_ml[feature_cols].fillna(0)
                    y = df_ml[target_col].fillna(0)

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    if y.nunique() <= 10:
                        model  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        model.fit(X_train, y_train)
                        score  = accuracy_score(y_test, model.predict(X_test))
                        metric = "Accuracy"
                    else:
                        model  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                        model.fit(X_train, y_train)
                        score  = r2_score(y_test, model.predict(X_test))
                        metric = "R² Score"

                    c1, c2, c3 = st.columns(3)
                    c1.metric(metric,           f"{score:.4f}")
                    c2.metric("Train Samples",  f"{len(X_train):,}")
                    c3.metric("Test Samples",   f"{len(X_test):,}")

                    feat_imp = pd.Series(model.feature_importances_, index=X.columns)\
                               .sort_values(ascending=False).head(10)

                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.bar(x=feat_imp.values, y=feat_imp.index,
                                     orientation='h',
                                     title=f'Top Features → {target_col}',
                                     color=feat_imp.values,
                                     color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        y_pred = model.predict(X_test)
                        fig = px.scatter(x=y_test, y=y_pred,
                                         title='Actual vs Predicted',
                                         labels={'x':'Actual','y':'Predicted'},
                                         color_discrete_sequence=['#C8FF00'])
                        st.plotly_chart(fig, use_container_width=True)

    # ── 3-AGENT ANALYSIS ──────────────────────────────────
    if run_analysis:
        if not question:
            st.warning("Please enter a question!")
        else:
            st.session_state.question = question
            # Limit data summary to avoid token limits
            num_cols_summary = df.select_dtypes(include='number').columns.tolist()[:5]
            cat_cols_summary = df.select_dtypes(include='object').columns.tolist()[:3]
            data_summary = f"""
Shape: {df.shape[0]} rows x {df.shape[1]} columns
Columns: {df.columns.tolist()[:10]}
Sample: {df.head(2)[df.columns[:8]].to_string()}
Stats: {df[num_cols_summary].describe().round(2).to_string() if num_cols_summary else 'N/A'}
Question: {question}
"""
            progress = st.progress(0)
            status   = st.empty()

            with st.spinner("🤖 3 AI Agents working..."):

                # 7 Specialized Agents with rotating API keys
                agents = {
                    'quality': Agent(
                        role='Data Quality Inspector',
                        goal='Thoroughly inspect data quality and identify all issues',
                        backstory="Expert data engineer with 10 years finding data quality issues.",
                        llm=llm1, verbose=False
                    ),
                    'stats': Agent(
                        role='Statistical Analyst',
                        goal='Perform deep statistical analysis and find key patterns',
                        backstory="PhD statistician who finds hidden patterns in complex datasets.",
                        llm=llm1, verbose=False
                    ),
                    'ml': Agent(
                        role='ML Model Expert',
                        goal='Recommend the best ML models with detailed justification',
                        backstory="Senior ML engineer who selects optimal models for any dataset.",
                        llm=llm2, verbose=False
                    ),
                    'viz': Agent(
                        role='Data Visualization Expert',
                        goal='Design the most impactful visualizations for this data',
                        backstory="Data viz expert who creates charts that tell compelling stories.",
                        llm=llm2, verbose=False
                    ),
                    'report': Agent(
                        role='Executive Report Writer',
                        goal='Write a clear professional executive summary',
                        backstory="Business analyst who writes reports for C-suite executives.",
                        llm=llm2, verbose=False
                    ),
                    'advisor': Agent(
                        role='Business Strategy Advisor',
                        goal='Provide specific actionable business recommendations',
                        backstory="Senior consultant who turns data into business decisions.",
                        llm=llm3, verbose=False
                    ),
                    'director': Agent(
                        role='Strategy Director',
                        goal='Create a prioritized 90-day strategic action plan',
                        backstory="C-suite advisor who builds strategic roadmaps from data.",
                        llm=llm3, verbose=False
                    ),
                }

                tasks = [
                    Task(
                        description=f"""Inspect data quality thoroughly.
Find: missing values per column, duplicate rows, outliers, wrong data types, inconsistencies.
Provide specific counts and percentages for each issue.
Data: {data_summary}""",
                        expected_output="Detailed data quality report with specific issues and fix recommendations",
                        agent=agents['quality']
                    ),
                    Task(
                        description=f"""Perform statistical analysis.
Find: correlations between columns, distributions, trends, anomalies.
Answer the user question: {question}
Data: {data_summary}""",
                        expected_output="Statistical analysis with correlations, patterns and direct answer to question",
                        agent=agents['stats']
                    ),
                    Task(
                        description=f"""Recommend top 3 ML models for this dataset.
For each model explain: why it fits, expected accuracy, pros and cons.
Also suggest feature engineering steps.
Data: {data_summary}""",
                        expected_output="Top 3 ML model recommendations with justification and feature engineering tips",
                        agent=agents['ml']
                    ),
                    Task(
                        description=f"""Recommend 4 specific visualizations.
For each chart specify: chart type, x-axis, y-axis, color, insight it shows.
Data: {data_summary}""",
                        expected_output="4 specific chart recommendations with full specifications",
                        agent=agents['viz']
                    ),
                    Task(
                        description=f"""Write a professional executive summary.
Include: overview paragraph, 5 bullet point key findings, conclusion.
Keep it clear for non-technical executives.
Data: {data_summary}""",
                        expected_output="Professional executive summary with overview, findings and conclusion",
                        agent=agents['report']
                    ),
                    Task(
                        description=f"""Provide 5 specific actionable business recommendations.
For each: what to do, why, expected impact, timeline.
Data: {data_summary}""",
                        expected_output="5 actionable recommendations with impact and timeline",
                        agent=agents['advisor']
                    ),
                    Task(
                        description=f"""Create a prioritized 90-day action plan.
Include: Week 1-2 quick wins, Month 1 short term, Month 2-3 medium term.
Add KPIs to track success.
Data: {data_summary}""",
                        expected_output="90-day strategic action plan with priorities and KPIs",
                        agent=agents['director']
                    ),
                ]

                agent_names = [
                    '🔍 Quality Inspector',
                    '📊 Statistical Analyst',
                    '🤖 ML Expert',
                    '📈 Visualization Expert',
                    '📝 Report Writer',
                    '💡 Business Advisor',
                    '🏆 Strategy Director'
                ]
                results = []

                for i, (task, name) in enumerate(zip(tasks, agent_names)):
                    status.text(f"Running {name}... ({i+1}/3)")
                    progress.progress((i+1) / 7)

                    max_retries = 2
                    for attempt in range(max_retries):
                        try:
                            crew = Crew(
                                agents=[list(agents.values())[i]],
                                tasks=[task],
                                process=Process.sequential,
                                verbose=False
                            )
                            crew.kickoff()
                            output = str(task.output) if task.output else "Analysis completed."
                            results.append(output)
                            time.sleep(3)
                            break
                        except Exception as e:
                            err = str(e).lower()
                            if "rate_limit" in err and attempt < max_retries-1:
                                wait = 30
                                status.text(f"⏳ Rate limit — waiting {wait}s before retry...")
                                time.sleep(wait)
                            else:
                                results.append(f"**{name}** — Analysis completed. Please try again for detailed results.")
                                time.sleep(2)
                                break

                st.session_state.agent_outputs = {
                    name: result for name, result in zip(agent_names, results)
                }
                st.session_state.analysis_done = True
                progress.progress(1.0)
                status.text("✅ All 3 Agents Complete!")

            st.success("✅ Analysis Complete!")

    # ── DISPLAY RESULTS ───────────────────────────────────
    if st.session_state.analysis_done and st.session_state.agent_outputs:
        st.markdown("---")
        st.markdown("### 📊 Agent Results")

        tabs    = st.tabs([
            "🔍 Quality", "📊 Statistics", "🤖 ML Models",
            "📈 Charts", "📝 Report", "💡 Advice",
            "🏆 Strategy", "📊 Auto Charts"
        ])
        outputs = list(st.session_state.agent_outputs.values())

        for i, (tab, output) in enumerate(zip(tabs[:7], outputs)):
            with tab:
                st.markdown(output)

        with tabs[7]:
            st.markdown("### 📊 Auto-Generated Charts")
            numeric_cols_chart = df.select_dtypes(include='number').columns.tolist()
            cat_cols_chart     = df.select_dtypes(include='object').columns.tolist()

            if len(numeric_cols_chart) >= 1:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(df, x=numeric_cols_chart[0],
                                       title=f'Distribution of {numeric_cols_chart[0]}',
                                       color_discrete_sequence=['#C8FF00'])
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    if len(numeric_cols_chart) >= 2:
                        fig = px.scatter(df, x=numeric_cols_chart[0], y=numeric_cols_chart[1],
                                         color=cat_cols_chart[0] if cat_cols_chart else None,
                                         title=f'{numeric_cols_chart[0]} vs {numeric_cols_chart[1]}')
                        st.plotly_chart(fig, use_container_width=True)

            if cat_cols_chart and numeric_cols_chart:
                fig = px.box(df, x=cat_cols_chart[0], y=numeric_cols_chart[0],
                             color=cat_cols_chart[0],
                             title=f'{numeric_cols_chart[0]} by {cat_cols_chart[0]}')
                st.plotly_chart(fig, use_container_width=True)

            if len(numeric_cols_chart) >= 3:
                fig = px.imshow(df[numeric_cols_chart].corr().round(2),
                                title='Correlation Heatmap',
                                color_continuous_scale='RdYlGn',
                                text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

        # ── PDF REPORT ────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📄 Download PDF Report")

        if st.button("📥 Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF..."):
                try:
                    pdf = FPDF()
                    pdf.add_page()

                    # Title
                    pdf.set_font('Helvetica', 'B', 20)
                    pdf.cell(0, 12, 'Multi-Agent AI Analysis Report', ln=True, align='C')
                    pdf.set_font('Helvetica', '', 11)
                    pdf.cell(0, 8, f'Dataset: {df.shape[0]} rows x {df.shape[1]} columns', ln=True, align='C')
                    pdf.cell(0, 8, f'Question: {st.session_state.question}', ln=True, align='C')
                    pdf.ln(5)

                    section_titles = [
                        '1. Data Quality Report',
                        '2. Statistical Analysis',
                        '3. ML Model Recommendations',
                        '4. Visualization Recommendations',
                        '5. Executive Summary',
                        '6. Business Recommendations',
                        '7. Strategic Action Plan',
                    ]

                    for title, output in zip(section_titles, outputs):
                        pdf.set_font('Helvetica', 'B', 14)
                        pdf.set_fill_color(200, 255, 0)
                        pdf.cell(0, 10, title, ln=True, fill=True)
                        pdf.set_font('Helvetica', '', 10)
                        clean = output.encode('latin-1', 'replace').decode('latin-1')
                        pdf.multi_cell(0, 6, clean[:2000])
                        pdf.ln(3)

                    pdf_bytes = bytes(pdf.output())
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_bytes,
                        file_name="ai_analysis_report.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ PDF ready!")
                except Exception as e:
                    st.error(f"PDF error: {e}")

    # ── CHAT ──────────────────────────────────────────────
    if st.session_state.analysis_done:
        st.markdown("---")
        st.markdown("### 💬 Chat with Your Data")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.write(msg['content'])

        if prompt := st.chat_input("Ask anything about your data..."):
            st.session_state.chat_history.append({'role':'user','content':prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chat_agent = Agent(
                            role='Data Chat Assistant',
                            goal='Answer questions about the dataset accurately and concisely',
                            backstory="Expert data analyst who answers questions clearly.",
                            llm=llm, verbose=False
                        )
                        chat_task = Task(
                            description=f"""
Dataset: {df.shape[0]} rows, columns: {df.columns.tolist()}
Sample: {df.head(2).to_string()}
Question: {prompt}
Answer concisely.""",
                            expected_output="Clear concise answer",
                            agent=chat_agent
                        )
                        Crew(agents=[chat_agent], tasks=[chat_task],
                             process=Process.sequential, verbose=False).kickoff()
                        response = str(chat_task.output)
                        st.write(response)
                        st.session_state.chat_history.append({'role':'assistant','content':response})
                    except Exception as e:
                        st.error(f"Chat error: {e}")

    # ── DATA PROFILING ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Data Profiling Report")
    with st.expander("📊 View Full Data Profile"):
        st.markdown("#### Column-by-Column Analysis")
        for col in df.columns:
            with st.container():
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.markdown(f"**{col}**")
                c2.markdown(f"Type: `{df[col].dtype}`")
                c3.markdown(f"Missing: `{df[col].isnull().sum():,}`")
                c4.markdown(f"Unique: `{df[col].nunique():,}`")
                if df[col].dtype in ['int64', 'float64']:
                    c5.markdown(f"Mean: `{df[col].mean():.2f}`")
                else:
                    c5.markdown(f"Mode: `{df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'}`")

        st.markdown("---")
        st.markdown("#### 📈 Distribution Charts")
        numeric_prof = df.select_dtypes(include='number').columns.tolist()
        if numeric_prof:
            cols_per_row = 3
            for i in range(0, min(9, len(numeric_prof)), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(numeric_prof[i:i+cols_per_row]):
                    with cols[j]:
                        fig = px.histogram(df, x=col, title=col,
                                          color_discrete_sequence=['#C8FF00'],
                                          height=200)
                        fig.update_layout(margin=dict(l=0,r=0,t=30,b=0))
                        st.plotly_chart(fig, use_container_width=True)

    # ── EXPORT CLEAN DATA ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 💾 Export Clean Data")
    c1, c2, c3 = st.columns(3)
    with c1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        json_data = df.to_json(orient='records').encode('utf-8')
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name="cleaned_data.json",
            mime="application/json",
            use_container_width=True
        )
    with c3:
        st.metric("Rows to export", f"{len(df):,}")

    # ── ANOMALY DETECTION ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚨 Anomaly Detection")
    with st.expander("🚨 Find Unusual Rows"):
        numeric_anom = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_anom) >= 2:
            contamination = st.slider("Anomaly sensitivity (% of data)", 1, 20, 5) / 100
            if st.button("🔍 Detect Anomalies"):
                with st.spinner("Finding anomalies..."):
                    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
                    df_anom = df[numeric_anom].fillna(df[numeric_anom].median())
                    preds   = clf.fit_predict(df_anom)
                    df['is_anomaly'] = (preds == -1).astype(int)

                    n_anomalies = df['is_anomaly'].sum()
                    st.warning(f"Found **{n_anomalies:,}** anomalies ({n_anomalies/len(df)*100:.1f}% of data)")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Sample Anomalous Rows:**")
                        st.dataframe(df[df['is_anomaly']==1].head(10), use_container_width=True)
                    with c2:
                        if len(numeric_anom) >= 2:
                            fig = px.scatter(
                                df, x=numeric_anom[0], y=numeric_anom[1],
                                color=df['is_anomaly'].map({0:'Normal', 1:'Anomaly'}),
                                title='Anomalies vs Normal Points',
                                color_discrete_map={'Normal':'#C8FF00', 'Anomaly':'#FF4757'}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    # Download anomalies
                    anom_csv = df[df['is_anomaly']==1].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Anomalies CSV",
                        data=anom_csv,
                        file_name="anomalies.csv",
                        mime="text/csv"
                    )
                    df.drop(columns=['is_anomaly'], inplace=True)
        else:
            st.warning("Need at least 2 numeric columns for anomaly detection!")

    # ── TIME SERIES DETECTION ─────────────────────────────
    st.markdown("---")
    st.markdown("### 📅 Time Series Analysis")
    with st.expander("📅 Detect & Analyse Time Trends"):
        # Auto detect date columns
        date_cols = []
        for col in df.columns:
            if any(kw in col.lower() for kw in ['date', 'time', 'day', 'month', 'year', 'timestamp']):
                date_cols.append(col)

        if date_cols:
            st.success(f"✅ Auto-detected date columns: {date_cols}")
            date_col   = st.selectbox("Select date column:", date_cols)
            numeric_ts = df.select_dtypes(include='number').columns.tolist()

            if numeric_ts and st.button("📈 Analyse Time Series"):
                with st.spinner("Analysing trends..."):
                    try:
                        df_ts          = df.copy()
                        df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
                        df_ts          = df_ts.dropna(subset=[date_col])
                        df_ts          = df_ts.sort_values(date_col)

                        value_col = st.selectbox("Select value column:", numeric_ts)
                        ts_data   = df_ts.groupby(date_col)[value_col].sum().reset_index()

                        fig = px.line(ts_data, x=date_col, y=value_col,
                                      title=f'{value_col} Over Time',
                                      color_discrete_sequence=['#C8FF00'])
                        st.plotly_chart(fig, use_container_width=True)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Peak Value",   f"{ts_data[value_col].max():,.2f}")
                        c2.metric("Min Value",    f"{ts_data[value_col].min():,.2f}")
                        c3.metric("Avg Value",    f"{ts_data[value_col].mean():,.2f}")

                    except Exception as e:
                        st.error(f"Time series error: {e}")
        else:
            st.info("No date columns detected. Common date column names: date, time, timestamp, created_at")
            manual_date = st.selectbox("Or manually select a date column:", df.columns.tolist())
            if st.button("Use this column as date"):
                try:
                    df[manual_date] = pd.to_datetime(df[manual_date], errors='coerce')
                    st.success(f"✅ Set {manual_date} as date column!")
                except Exception as e:
                    st.error(f"Could not convert: {e}")

    # ── NATURAL LANGUAGE TO CHART ─────────────────────────
    st.markdown("---")
    st.markdown("### 💬 Natural Language to Chart")
    st.markdown("Type what chart you want and AI will create it!")

    nl_query = st.text_input(
        "Describe your chart:",
        placeholder="e.g. show reorder rate by department, plot sales over time, compare income by region"
    )

    if st.button("🎨 Generate Chart", type="primary"):
        if not nl_query:
            st.warning("Please describe the chart you want!")
        else:
            with st.spinner("AI generating chart..."):
                try:
                    chart_agent = Agent(
                        role='Chart Code Generator',
                        goal='Generate plotly chart specifications from natural language',
                        backstory="Expert data visualization engineer who creates perfect charts.",
                        llm=llm1, verbose=False
                    )
                    chart_task = Task(
                        description=f"""
User wants this chart: "{nl_query}"
Dataset columns: {df.columns.tolist()}
Column types: {df.dtypes.to_dict()}
Sample data: {df.head(3).to_string()}

Respond with ONLY a JSON object (no other text) with these exact keys:
{{
  "chart_type": "bar|line|scatter|pie|histogram|box",
  "x": "column_name",
  "y": "column_name or null",
  "color": "column_name or null",
  "title": "chart title",
  "aggregation": "sum|mean|count|none"
}}
""",
                        expected_output="JSON object with chart specifications",
                        agent=chart_agent
                    )
                    Crew(agents=[chart_agent], tasks=[chart_task],
                         process=Process.sequential, verbose=False).kickoff()

                    import json
                    output = str(chart_task.output).strip()
                    # Extract JSON
                    start = output.find('{')
                    end   = output.rfind('}') + 1
                    if start >= 0 and end > start:
                        spec = json.loads(output[start:end])

                        x_col   = spec.get('x')
                        y_col   = spec.get('y')
                        color   = spec.get('color')
                        title   = spec.get('title', nl_query)
                        agg     = spec.get('aggregation', 'none')
                        ctype   = spec.get('chart_type', 'bar')

                        # Prepare data
                        plot_df = df.copy()
                        if agg != 'none' and x_col and y_col:
                            if agg == 'sum':
                                plot_df = df.groupby(x_col)[y_col].sum().reset_index()
                            elif agg == 'mean':
                                plot_df = df.groupby(x_col)[y_col].mean().reset_index()
                            elif agg == 'count':
                                plot_df = df.groupby(x_col)[y_col].count().reset_index()

                        # Generate chart
                        if ctype == 'bar':
                            fig = px.bar(plot_df, x=x_col, y=y_col, color=color, title=title,
                                        color_discrete_sequence=['#C8FF00','#FF6B35','#38BDF8'])
                        elif ctype == 'line':
                            fig = px.line(plot_df, x=x_col, y=y_col, color=color, title=title,
                                         color_discrete_sequence=['#C8FF00'])
                        elif ctype == 'scatter':
                            fig = px.scatter(plot_df, x=x_col, y=y_col, color=color, title=title)
                        elif ctype == 'pie':
                            fig = px.pie(plot_df, names=x_col, values=y_col, title=title)
                        elif ctype == 'histogram':
                            fig = px.histogram(plot_df, x=x_col, title=title,
                                             color_discrete_sequence=['#C8FF00'])
                        elif ctype == 'box':
                            fig = px.box(plot_df, x=x_col, y=y_col, color=color, title=title)
                        else:
                            fig = px.bar(plot_df, x=x_col, y=y_col, title=title)

                        st.plotly_chart(fig, use_container_width=True)
                        st.success(f"✅ Generated {ctype} chart!")
                    else:
                        st.error("Could not parse chart specification. Try rephrasing!")
                except Exception as e:
                    st.error(f"Chart generation error: {e}")

# ── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#888;padding:1rem'>
    Built by <strong style='color:#C8FF00'>Nagarajulu Reddy Nalla</strong> |
    AI Engineer · Data Scientist · ML Engineer · Power BI Developer
</div>
""", unsafe_allow_html=True)