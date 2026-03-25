import os
import time
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from fpdf import FPDF
import warnings
warnings.filterwarnings("ignore")

# ── LOAD ENV ──────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
))
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY2 = os.getenv("GROQ_API_KEY_2", GROQ_API_KEY)
GROQ_API_KEY3 = os.getenv("GROQ_API_KEY_3", GROQ_API_KEY)
GROQ_KEYS     = [GROQ_API_KEY, GROQ_API_KEY2, GROQ_API_KEY3]

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Multi-Agent AI Platform",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #0A0A0A, #1E1E1E);
    padding: 2rem; border-radius: 12px;
    border: 1px solid #C8FF00; margin-bottom: 2rem;
}
.agent-card {
    background: #1E1E1E; border: 1px solid #333;
    border-radius: 10px; padding: 12px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── GROQ API CALL ─────────────────────────────────────────
def call_groq(prompt, key_index=0, max_retries=3):
    key = GROQ_KEYS[key_index % len(GROQ_KEYS)]
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                    "temperature": 0.1
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                wait = 30 * (attempt + 1)
                time.sleep(wait)
            else:
                return f"API Error: {response.status_code}"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                return f"Error: {str(e)}"
    return "Rate limit reached. Please try again in 1 minute."

# ── HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="color:#C8FF00; margin:0">🤖 Enterprise Multi-Agent AI Analytics Platform</h1>
    <p style="color:#888; margin:0.5rem 0 0">
        7 Specialized AI Agents · Auto ML · Smart Data Cleaning · Multi-file Join · Chat · PDF Reports
    </p>
</div>
""", unsafe_allow_html=True)

# ── AGENT CARDS ───────────────────────────────────────────
cols = st.columns(7)
agents_info = [
    ("🔍", "Quality",    "Inspector"),
    ("📊", "Statistics", "Analyst"),
    ("🤖", "ML Expert",  "Recommender"),
    ("📈", "Visualizer", "Designer"),
    ("📝", "Reporter",   "Writer"),
    ("💡", "Advisor",    "Consultant"),
    ("🏆", "Director",   "Strategist"),
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
for key, val in {
    "df": None, "analysis_done": False,
    "chat_history": [], "agent_outputs": {},
    "question": "", "show_ml": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── FILE UPLOAD ───────────────────────────────────────────
st.markdown("### 📁 Upload Your Dataset")
upload_mode = st.radio("Upload Mode", ["Single File", "Multiple Files (Smart Join)", "Sample Data"], horizontal=True)

if upload_mode == "Single File":
    uploaded_file = st.file_uploader("Upload CSV, Excel or JSON", type=["csv","xlsx","xls","json"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                st.session_state.df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith((".xlsx",".xls")):
                st.session_state.df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".json"):
                st.session_state.df = pd.read_json(uploaded_file)
            st.success(f"✅ Loaded: {uploaded_file.name} — {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} cols")
        except Exception as e:
            st.error(f"Error: {e}")

elif upload_mode == "Multiple Files (Smart Join)":
    st.info("Upload multiple CSV files — Smart Join Wizard detects relationships!")
    uploaded_files = st.file_uploader("Upload multiple CSV files", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        dataframes = {}
        for f in uploaded_files:
            try:
                dataframes[f.name] = pd.read_csv(f)
                st.write(f"📄 **{f.name}** — {dataframes[f.name].shape[0]:,} rows × {dataframes[f.name].shape[1]} cols")
            except Exception as e:
                st.error(f"Error loading {f.name}: {e}")

        if len(dataframes) > 1:
            st.markdown("---")
            st.markdown("### 🧙 Smart Join Wizard")
            file_names     = list(dataframes.keys())
            relationships  = []
            for i in range(len(file_names)):
                for j in range(i+1, len(file_names)):
                    f1, f2 = file_names[i], file_names[j]
                    common = set(dataframes[f1].columns) & set(dataframes[f2].columns)
                    for col in common:
                        overlap = len(set(dataframes[f1][col].dropna().unique()) &
                                      set(dataframes[f2][col].dropna().unique()))
                        if overlap > 0:
                            relationships.append({"file1":f1,"file2":f2,"column":col,"overlap":overlap})

            if relationships:
                st.success(f"✅ Found {len(relationships)} relationship(s)!")
                for rel in relationships:
                    st.write(f"🔗 **{rel['file1']}** ↔ **{rel['file2']}** via `{rel['column']}` ({rel['overlap']:,} matches)")

            join_mode = st.selectbox("Join strategy:", [
                "🤖 Auto Join (recommended)", "📥 Stack vertically",
                "🔗 Manual join", "📊 Analyse one file"
            ])

            if join_mode == "🤖 Auto Join (recommended)":
                if st.button("🤖 Auto Join All Files", type="primary"):
                    with st.spinner("Joining files..."):
                        try:
                            largest      = max(dataframes.keys(), key=lambda x: len(dataframes[x]))
                            result       = dataframes[largest]
                            joined_files = [largest]
                            st.write(f"🚀 Base: **{largest}** ({len(result):,} rows)")
                            for _ in range(len(dataframes)):
                                for rel in sorted(relationships, key=lambda x: x["overlap"], reverse=True):
                                    l_in = rel["file1"] in joined_files
                                    r_in = rel["file2"] in joined_files
                                    if l_in and not r_in:
                                        next_f, join_on = rel["file2"], rel["column"]
                                    elif r_in and not l_in:
                                        next_f, join_on = rel["file1"], rel["column"]
                                    else:
                                        continue
                                    before = result.shape[1]
                                    result = result.merge(dataframes[next_f], on=join_on, how="left", suffixes=("", f"_{next_f[:4]}"))
                                    joined_files.append(next_f)
                                    st.write(f"✅ Joined **{next_f}** on `{join_on}` (+{result.shape[1]-before} cols)")
                                    break
                            if len(result) > 20000000:
                                st.warning(f"Dataset has {len(result):,} rows — sampling 1M rows!")
                                result = result.sample(1000000, random_state=42)
                            st.session_state.df = result
                            st.success(f"🎉 Done! → {result.shape[0]:,} rows × {result.shape[1]} cols")
                        except Exception as e:
                            st.error(f"Join error: {e}")

            elif join_mode == "📥 Stack vertically":
                if st.button("📥 Stack Files"):
                    try:
                        combined = pd.concat(list(dataframes.values()), ignore_index=True)
                        st.session_state.df = combined
                        st.success(f"Stacked → {combined.shape[0]:,} rows")
                    except Exception as e:
                        st.error(f"Error: {e}")

            elif join_mode == "🔗 Manual join":
                c1, c2 = st.columns(2)
                left_file  = c1.selectbox("Left file:", file_names)
                right_file = c2.selectbox("Right file:", [f for f in file_names if f != left_file])
                common_cols = list(set(dataframes[left_file].columns) & set(dataframes[right_file].columns))
                if common_cols:
                    c1, c2 = st.columns(2)
                    join_col  = c1.selectbox("Join on:", common_cols)
                    join_type = c2.selectbox("Type:", ["left","inner","outer"])
                    if st.button("🔗 Join"):
                        result = dataframes[left_file].merge(dataframes[right_file], on=join_col, how=join_type)
                        st.session_state.df = result
                        st.success(f"Joined → {result.shape[0]:,} rows")
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
        "CustomerID"    : range(1, n+1),
        "Age"           : np.random.randint(18, 70, n),
        "Income"        : np.random.randint(20000, 150000, n),
        "SpendingScore" : np.random.randint(1, 100, n),
        "TotalPurchases": np.random.randint(1, 50, n),
        "Region"        : np.random.choice(["North","South","East","West"], n),
        "Segment"       : np.random.choice(["Premium","Standard","Basic"], n),
        "Churned"       : np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    st.success(f"✅ Sample dataset loaded — {st.session_state.df.shape[0]:,} rows!")

# ── DATASET OVERVIEW ──────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df
    st.markdown("---")
    st.markdown("### 📋 Dataset Overview")
    c1,c2,c3,c4,c5 = st.columns(5)
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
        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(include="object").columns
        st.markdown("#### 📊 Data Quality Report")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Duplicates",     f"{df.duplicated().sum():,}")
        c2.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        c3.metric("Numeric Cols",   f"{len(num_cols)}")
        c4.metric("Text Cols",      f"{len(cat_cols)}")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            for col, count in missing.items():
                st.write(f"  • `{col}`: {count:,} missing ({count/len(df)*100:.1f}%)")
        else:
            st.write("✅ No missing values!")
        remove_outliers = st.checkbox("Also remove outliers (3x IQR)", value=False)
        if st.button("🧹 Auto Clean Data", type="primary"):
            df_clean    = df.copy()
            changes     = []
            rows_before = len(df_clean)
            df_clean    = df_clean.drop_duplicates()
            removed     = rows_before - len(df_clean)
            if removed > 0:
                changes.append(f"Removed {removed:,} duplicates")
            for col in df_clean.select_dtypes(include="number").columns:
                miss = df_clean[col].isnull().sum()
                if miss > 0:
                    pct = miss / len(df_clean) * 100
                    if pct > 50:
                        df_clean.drop(columns=[col], inplace=True)
                        changes.append(f"Dropped '{col}' ({pct:.0f}% missing)")
                    else:
                        med = df_clean[col].median()
                        df_clean[col].fillna(med, inplace=True)
                        changes.append(f"Filled '{col}' with median")
            for col in df_clean.select_dtypes(include="object").columns:
                miss = df_clean[col].isnull().sum()
                if miss > 0:
                    pct = miss / len(df_clean) * 100
                    if pct > 50:
                        df_clean.drop(columns=[col], inplace=True)
                        changes.append(f"Dropped '{col}' ({pct:.0f}% missing)")
                    else:
                        mode_val = df_clean[col].mode()[0]
                        df_clean[col].fillna(mode_val, inplace=True)
                        changes.append(f"Filled '{col}' with mode")
            if remove_outliers:
                before    = len(df_clean)
                num_cols2 = df_clean.select_dtypes(include="number").columns
                mask      = pd.Series([True]*len(df_clean), index=df_clean.index)
                for col in num_cols2:
                    Q1  = df_clean[col].quantile(0.25)
                    Q3  = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    mask = mask & (df_clean[col] >= Q1-3*IQR) & (df_clean[col] <= Q3+3*IQR)
                df_clean = df_clean[mask]
                removed  = before - len(df_clean)
                if removed > 0:
                    changes.append(f"Removed {removed:,} outliers")
            st.session_state.df = df_clean
            df = df_clean
            st.success("✅ Cleaning complete!")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Rows before",  f"{rows_before:,}")
            c2.metric("Rows after",   f"{len(df_clean):,}")
            c3.metric("Cols",         f"{df_clean.shape[1]}")
            c4.metric("Missing left", f"{df_clean.isnull().sum().sum():,}")
            for change in changes:
                st.write(f"✅ {change}")

    # ── ANALYSIS ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 7-Agent AI Analysis")
    question = st.text_input("What do you want to analyse?",
                              placeholder="e.g. What are the key customer segments?")
    c1, c2 = st.columns(2)
    with c1:
        run_analysis = st.button("🤖 Run 7-Agent Analysis", type="primary", use_container_width=True)
    with c2:
        if st.button("🧠 Run Auto ML", use_container_width=True):
            st.session_state.show_ml = True

    # ── AUTO ML ───────────────────────────────────────────
    if st.session_state.show_ml:
        st.markdown("---")
        st.markdown("### 🧠 Auto ML Predictions")
        numeric_cols_ml = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols_ml) < 2:
            st.warning("Need at least 2 numeric columns!")
        else:
            target_keywords = ["churn","churned","reorder","reordered","target","label",
                               "y","outcome","result","class","attrition","fraud","price","sales"]
            auto_target = next((col for col in df.columns
                                if any(kw in col.lower() for kw in target_keywords)), numeric_cols_ml[-1])
            st.info(f"🎯 Auto-detected target: **{auto_target}**")
            default_idx = numeric_cols_ml.index(auto_target) if auto_target in numeric_cols_ml else 0
            target_col  = st.selectbox("Target column:", numeric_cols_ml, index=default_idx)
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training..."):
                    df_ml = df.copy()
                    le    = LabelEncoder()
                    for col in df.select_dtypes(include="object").columns:
                        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
                    X = df_ml[[c for c in df_ml.columns if c != target_col]].fillna(0)
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
                    c1,c2,c3 = st.columns(3)
                    c1.metric(metric, f"{score:.4f}")
                    c2.metric("Train Samples", f"{len(X_train):,}")
                    c3.metric("Test Samples",  f"{len(X_test):,}")
                    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.bar(x=feat_imp.values, y=feat_imp.index, orientation="h",
                                     title=f"Top Features → {target_col}",
                                     color=feat_imp.values, color_continuous_scale="Viridis")
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        y_pred = model.predict(X_test)
                        fig = px.scatter(x=y_test, y=y_pred, title="Actual vs Predicted",
                                         labels={"x":"Actual","y":"Predicted"},
                                         color_discrete_sequence=["#C8FF00"])
                        st.plotly_chart(fig, use_container_width=True)

    # ── 7-AGENT ANALYSIS ──────────────────────────────────
    if run_analysis:
        if not question:
            st.warning("Please enter a question!")
        else:
            st.session_state.question = question
            num_s = df.select_dtypes(include="number").columns.tolist()[:5]
            data_summary = f"""Shape: {df.shape[0]:,} rows x {df.shape[1]} cols
Columns: {df.columns.tolist()[:10]}
Sample: {df.head(3)[df.columns[:6]].to_string()}
Stats: {df[num_s].describe().round(2).to_string() if num_s else "N/A"}
Question: {question}"""

            agent_prompts = [
                ("🔍 Quality Inspector",
                 f"You are a Data Quality Inspector. Analyse this dataset.\n{data_summary}\nReport: missing values per column, duplicates, outliers, data type issues. Give specific counts."),
                ("📊 Statistical Analyst",
                 f"You are a Statistical Analyst. Find key patterns and answer the question.\n{data_summary}\nProvide: correlations, distributions, trends, anomalies, and direct answer to the question."),
                ("🤖 ML Expert",
                 f"You are an ML Model Expert. Recommend the best ML models for this dataset.\n{data_summary}\nRecommend top 3 models with: why they fit, expected accuracy, pros/cons, feature engineering tips."),
                ("📈 Visualization Expert",
                 f"You are a Data Visualization Expert. Recommend 4 specific charts.\n{data_summary}\nFor each chart specify: chart type, x-axis, y-axis, color, what insight it reveals."),
                ("📝 Report Writer",
                 f"You are a Business Report Writer. Write an executive summary.\n{data_summary}\nInclude: overview paragraph, 5 bullet key findings, conclusion. Keep it clear for executives."),
                ("💡 Business Advisor",
                 f"You are a Business Strategy Advisor. Provide 5 actionable recommendations.\n{data_summary}\nFor each: what to do, why, expected impact, timeline."),
                ("🏆 Strategy Director",
                 f"You are a Strategy Director. Create a 90-day action plan.\n{data_summary}\nInclude: Week 1-2 quick wins, Month 1, Month 2-3. Add KPIs to track success."),
            ]

            progress = st.progress(0)
            status   = st.empty()
            results  = {}

            for i, (name, prompt) in enumerate(agent_prompts):
                status.text(f"Running {name}... ({i+1}/7)")
                progress.progress((i+1) / 7)
                result = call_groq(prompt, key_index=i)
                results[name] = result
                time.sleep(3)

            st.session_state.agent_outputs  = results
            st.session_state.analysis_done  = True
            progress.progress(1.0)
            status.text("✅ All 7 Agents Complete!")
            st.success("✅ Analysis Complete!")

    # ── DISPLAY RESULTS ───────────────────────────────────
    if st.session_state.analysis_done and st.session_state.agent_outputs:
        st.markdown("---")
        st.markdown("### 📊 Agent Results")
        tabs    = st.tabs(["🔍 Quality","📊 Statistics","🤖 ML Models","📈 Charts","📝 Report","💡 Advice","🏆 Strategy","📊 Auto Charts"])
        outputs = list(st.session_state.agent_outputs.values())

        for i, (tab, output) in enumerate(zip(tabs[:7], outputs)):
            with tab:
                st.markdown(output)

        with tabs[7]:
            st.markdown("### 📊 Auto-Generated Charts")
            num_c = df.select_dtypes(include="number").columns.tolist()
            cat_c = df.select_dtypes(include="object").columns.tolist()
            if len(num_c) >= 1:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(df, x=num_c[0], title=f"Distribution of {num_c[0]}",
                                       color_discrete_sequence=["#C8FF00"])
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    if len(num_c) >= 2:
                        fig = px.scatter(df, x=num_c[0], y=num_c[1],
                                         color=cat_c[0] if cat_c else None,
                                         title=f"{num_c[0]} vs {num_c[1]}")
                        st.plotly_chart(fig, use_container_width=True)
            if cat_c and num_c:
                fig = px.box(df, x=cat_c[0], y=num_c[0], color=cat_c[0],
                             title=f"{num_c[0]} by {cat_c[0]}")
                st.plotly_chart(fig, use_container_width=True)
            if len(num_c) >= 3:
                fig = px.imshow(df[num_c].corr().round(2), title="Correlation Heatmap",
                                color_continuous_scale="RdYlGn", text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

        # ── PDF REPORT ────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📄 Download PDF Report")
        if st.button("📥 Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF..."):
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Helvetica", "B", 20)
                    pdf.cell(0, 12, "Multi-Agent AI Analysis Report", ln=True, align="C")
                    pdf.set_font("Helvetica", "", 11)
                    pdf.cell(0, 8, f"Dataset: {df.shape[0]:,} rows x {df.shape[1]} cols", ln=True, align="C")
                    pdf.cell(0, 8, f"Question: {st.session_state.question}", ln=True, align="C")
                    pdf.ln(5)
                    titles = ["1. Data Quality","2. Statistical Analysis","3. ML Models",
                              "4. Visualizations","5. Executive Summary","6. Recommendations","7. Action Plan"]
                    for title, output in zip(titles, outputs):
                        pdf.set_font("Helvetica", "B", 14)
                        pdf.set_fill_color(200, 255, 0)
                        pdf.cell(0, 10, title, ln=True, fill=True)
                        pdf.set_font("Helvetica", "", 10)
                        clean = output.encode("latin-1","replace").decode("latin-1")
                        pdf.multi_cell(0, 6, clean[:2000])
                        pdf.ln(3)
                    pdf_bytes = bytes(pdf.output())
                    st.download_button("📥 Download PDF", data=pdf_bytes,
                                       file_name="ai_report.pdf", mime="application/pdf")
                    st.success("✅ PDF ready!")
                except Exception as e:
                    st.error(f"PDF error: {e}")

    # ── DATA PROFILING ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Data Profiling Report")
    with st.expander("📊 View Full Data Profile"):
        for col in df.columns:
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.markdown(f"**{col}**")
            c2.markdown(f"`{df[col].dtype}`")
            c3.markdown(f"Missing: `{df[col].isnull().sum():,}`")
            c4.markdown(f"Unique: `{df[col].nunique():,}`")
            if df[col].dtype in ["int64","float64"]:
                c5.markdown(f"Mean: `{df[col].mean():.2f}`")
            else:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
                c5.markdown(f"Mode: `{mode_val}`")
        st.markdown("---")
        st.markdown("#### Distribution Charts")
        num_prof = df.select_dtypes(include="number").columns.tolist()
        for i in range(0, min(9, len(num_prof)), 3):
            cols = st.columns(3)
            for j, col in enumerate(num_prof[i:i+3]):
                with cols[j]:
                    fig = px.histogram(df, x=col, title=col,
                                       color_discrete_sequence=["#C8FF00"], height=200)
                    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig, use_container_width=True)

    # ── EXPORT DATA ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💾 Export Clean Data")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("📥 Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="clean_data.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("📥 Download JSON", data=df.to_json(orient="records").encode("utf-8"),
                           file_name="clean_data.json", mime="application/json", use_container_width=True)
    with c3:
        st.metric("Rows to export", f"{len(df):,}")

    # ── ANOMALY DETECTION ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚨 Anomaly Detection")
    with st.expander("🚨 Find Unusual Rows"):
        num_anom = df.select_dtypes(include="number").columns.tolist()
        if len(num_anom) >= 2:
            contamination = st.slider("Sensitivity (%)", 1, 20, 5) / 100
            if st.button("🔍 Detect Anomalies"):
                with st.spinner("Finding anomalies..."):
                    clf    = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
                    df_a   = df[num_anom].fillna(df[num_anom].median())
                    preds  = clf.fit_predict(df_a)
                    df["is_anomaly"] = (preds == -1).astype(int)
                    n_anom = df["is_anomaly"].sum()
                    st.warning(f"Found **{n_anom:,}** anomalies ({n_anom/len(df)*100:.1f}%)")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.dataframe(df[df["is_anomaly"]==1].head(10), use_container_width=True)
                    with c2:
                        fig = px.scatter(df, x=num_anom[0], y=num_anom[1],
                                         color=df["is_anomaly"].map({0:"Normal",1:"Anomaly"}),
                                         title="Anomalies vs Normal",
                                         color_discrete_map={"Normal":"#C8FF00","Anomaly":"#FF4757"})
                        st.plotly_chart(fig, use_container_width=True)
                    st.download_button("📥 Download Anomalies",
                                       data=df[df["is_anomaly"]==1].to_csv(index=False).encode("utf-8"),
                                       file_name="anomalies.csv", mime="text/csv")
                    df.drop(columns=["is_anomaly"], inplace=True)
        else:
            st.warning("Need at least 2 numeric columns!")

    # ── TIME SERIES ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📅 Time Series Analysis")
    with st.expander("📅 Analyse Time Trends"):
        date_cols = [col for col in df.columns
                     if any(kw in col.lower() for kw in ["date","time","day","month","year","timestamp"])]
        if date_cols:
            st.success(f"✅ Detected date columns: {date_cols}")
            date_col   = st.selectbox("Select date column:", date_cols)
            num_ts     = df.select_dtypes(include="number").columns.tolist()
            if num_ts and st.button("📈 Analyse Trends"):
                with st.spinner("Analysing..."):
                    try:
                        df_ts            = df.copy()
                        df_ts[date_col]  = pd.to_datetime(df_ts[date_col], errors="coerce")
                        df_ts            = df_ts.dropna(subset=[date_col]).sort_values(date_col)
                        value_col        = st.selectbox("Value column:", num_ts)
                        ts_data          = df_ts.groupby(date_col)[value_col].sum().reset_index()
                        fig = px.line(ts_data, x=date_col, y=value_col,
                                      title=f"{value_col} Over Time",
                                      color_discrete_sequence=["#C8FF00"])
                        st.plotly_chart(fig, use_container_width=True)
                        c1,c2,c3 = st.columns(3)
                        c1.metric("Peak",    f"{ts_data[value_col].max():,.2f}")
                        c2.metric("Min",     f"{ts_data[value_col].min():,.2f}")
                        c3.metric("Average", f"{ts_data[value_col].mean():,.2f}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("No date columns detected automatically.")
            manual = st.selectbox("Select date column manually:", df.columns.tolist())
            if st.button("Use as date column"):
                try:
                    df[manual] = pd.to_datetime(df[manual], errors="coerce")
                    st.success(f"✅ Set {manual} as date!")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── NATURAL LANGUAGE TO CHART ─────────────────────────
    st.markdown("---")
    st.markdown("### 💬 Natural Language to Chart")
    nl_query = st.text_input("Describe your chart:",
                              placeholder="e.g. show reorder rate by department, plot sales over time")
    if st.button("🎨 Generate Chart", type="primary"):
        if not nl_query:
            st.warning("Please describe the chart!")
        else:
            with st.spinner("AI generating chart..."):
                prompt = f"""User wants this chart: "{nl_query}"
Dataset columns: {df.columns.tolist()}
Column types: {df.dtypes.to_dict()}
Sample: {df.head(2).to_string()}

Respond with ONLY valid JSON (no other text):
{{"chart_type": "bar|line|scatter|pie|histogram|box", "x": "column_name", "y": "column_name or null", "color": "column_name or null", "title": "chart title", "aggregation": "sum|mean|count|none"}}"""
                response = call_groq(prompt, key_index=0)
                try:
                    start = response.find("{")
                    end   = response.rfind("}") + 1
                    spec  = json.loads(response[start:end])
                    x_col = spec.get("x")
                    y_col = spec.get("y")
                    color = spec.get("color")
                    title = spec.get("title", nl_query)
                    agg   = spec.get("aggregation","none")
                    ctype = spec.get("chart_type","bar")
                    plot_df = df.copy()
                    if agg != "none" and x_col and y_col and x_col in df.columns and y_col in df.columns:
                        if agg == "sum":   plot_df = df.groupby(x_col)[y_col].sum().reset_index()
                        elif agg == "mean": plot_df = df.groupby(x_col)[y_col].mean().reset_index()
                        elif agg == "count": plot_df = df.groupby(x_col)[y_col].count().reset_index()
                    if ctype == "bar":       fig = px.bar(plot_df,x=x_col,y=y_col,color=color,title=title)
                    elif ctype == "line":    fig = px.line(plot_df,x=x_col,y=y_col,color=color,title=title)
                    elif ctype == "scatter": fig = px.scatter(plot_df,x=x_col,y=y_col,color=color,title=title)
                    elif ctype == "pie":     fig = px.pie(plot_df,names=x_col,values=y_col,title=title)
                    elif ctype == "histogram": fig = px.histogram(plot_df,x=x_col,title=title)
                    elif ctype == "box":     fig = px.box(plot_df,x=x_col,y=y_col,color=color,title=title)
                    else:                    fig = px.bar(plot_df,x=x_col,y=y_col,title=title)
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"✅ Generated {ctype} chart!")
                except Exception as e:
                    st.error(f"Could not generate chart: {e}. Try rephrasing!")

    # ── CHAT ──────────────────────────────────────────────
    if st.session_state.analysis_done:
        st.markdown("---")
        st.markdown("### 💬 Chat with Your Data")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if prompt := st.chat_input("Ask anything about your data..."):
            st.session_state.chat_history.append({"role":"user","content":prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chat_prompt = f"""Dataset: {df.shape[0]:,} rows, columns: {df.columns.tolist()}
Sample: {df.head(2).to_string()}
Question: {prompt}
Answer concisely and accurately."""
                    response = call_groq(chat_prompt)
                    st.write(response)
                    st.session_state.chat_history.append({"role":"assistant","content":response})

# ── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#888;padding:1rem'>
    Built by <strong style='color:#C8FF00'>Nagarajulu Reddy Nalla</strong> |
    AI Engineer · Data Scientist · ML Engineer · Power BI Developer
</div>
""", unsafe_allow_html=True) 