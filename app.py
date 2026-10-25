"""
Loan Approval Predictor - Streamlit app
Run:  streamlit run app.py
Needs: pip install streamlit pandas numpy plotly joblib scikit-learn xgboost
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# 1. CONFIG  (edit this section to match YOUR notebook)
# =====================================================================
st.set_page_config(page_title="Loan Lends | Loan Approval Predictor", page_icon="🏦", layout="wide")
st.caption("👨‍💻 Built by [Zakir Ahmed](https://www.linkedin.com/in/zakir00/)")

# Save your models in the notebook like:  joblib.dump(logistic_model, "logistic_model.pkl")
MODEL_FILES = {
    "Logistic Regression": "logistic_model.pkl",
    "Decision Tree": "Decision_model.pkl",
    "Random Forest": "Random_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "KNN": "Knn_model.pkl",
}

# Optional: if you scaled data (StandardScaler/MinMaxScaler) before Logistic/KNN,
# save it with joblib.dump(scaler, "scaler.pkl") and list those models here.
SCALER_FILE = "scaler.pkl"
SCALED_MODELS = {"Logistic Regression", "KNN"}

# Exact column order used for training (X.columns)
FEATURES = [
    "person_age", "person_education", "person_income", "person_emp_exp",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file",
    "person_gender_male", "person_home_ownership_OTHER", "person_home_ownership_OWN",
    "person_home_ownership_RENT", "loan_intent_EDUCATION", "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL", "loan_intent_PERSONAL", "loan_intent_VENTURE",
]

# How person_education was encoded in your notebook. CHANGE if you used something else
# (e.g. LabelEncoder is alphabetical: Associate 0, Bachelor 1, Doctorate 2, High School 3, Master 4)
EDUCATION_MAP = {"Associate": 1, "High School": 2, "Bachelor": 3, "Master": 4, "Doctorate": 5}

# In this dataset loan_status 1 = Approved, 0 = Rejected. Flip if yours is reversed.
APPROVED_CLASS = 1

# Metrics you measured on the test set
METRICS = {
    "Logistic Regression": {"Accuracy": 0.82, "Precision": 0.67, "Recall": 0.39, "F1 Score": 0.50},
    "Decision Tree":       {"Accuracy": 0.90, "Precision": 0.77, "Recall": 0.78, "F1 Score": 0.78},
    "Random Forest":       {"Accuracy": 0.92, "Precision": 0.89, "Recall": 0.77, "F1 Score": 0.83},
    "XGBoost":             {"Accuracy": 0.93, "Precision": 0.89, "Recall": 0.80, "F1 Score": 0.844},
    "KNN":                 {"Accuracy": 0.82, "Precision": 0.62, "Recall": 0.53, "F1 Score": 0.57},
}

MODEL_INFO = {
    "Logistic Regression": ("📈", "Draws a straight boundary between approved and rejected. Fast and easy to explain, but misses complex patterns."),
    "Decision Tree": ("🌳", "Asks a chain of yes/no questions (e.g. 'credit score > 650?'). Very readable, can overfit."),
    "Random Forest": ("🌲", "Hundreds of decision trees voting together. Stable and accurate."),
    "XGBoost": ("⚡", "Builds trees one after another, each fixing the previous one's mistakes. Our top performer."),
    "KNN": ("👥", "Finds the most similar past applicants and copies their outcome."),
}

# =====================================================================
# 2. STYLE
# =====================================================================
st.markdown("""
<style>
.hero{padding:28px 32px;border-radius:18px;color:white;margin-bottom:18px;
      background:linear-gradient(120deg,#1e3a8a 0%,#6d28d9 55%,#db2777 100%);}
.hero h1{margin:0;font-size:2.2rem;color:white}
.hero p{margin:6px 0 0 0;opacity:.92;font-size:1.05rem}
.card{padding:16px 18px;border-radius:14px;border:1px solid rgba(128,128,128,.25);
      background:rgba(128,128,128,.07);height:100%}
.verdict-ok{padding:18px;border-radius:14px;background:rgba(16,185,129,.15);border:2px solid #10b981;font-size:1.3rem;font-weight:700;text-align:center}
.verdict-no{padding:18px;border-radius:14px;background:rgba(239,68,68,.15);border:2px solid #ef4444;font-size:1.3rem;font-weight:700;text-align:center}
.step{padding:14px 18px;border-left:5px solid #6d28d9;border-radius:8px;background:rgba(109,40,217,.08);margin-bottom:10px}
div.stButton > button{border-radius:12px;font-weight:600}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 3. HELPERS
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    loaded, missing = {}, []
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
        else:
            missing.append(path)
    scaler = joblib.load(SCALER_FILE) if os.path.exists(SCALER_FILE) else None
    return loaded, missing, scaler


MODELS, MISSING, SCALER = load_models()


def build_row(p: dict) -> pd.DataFrame:
    """Turn friendly form inputs into the exact 19 columns the models were trained on."""
    row = {
        "person_age": p["age"],
        "person_education": EDUCATION_MAP[p["education"]],
        "person_income": p["income"],
        "person_emp_exp": p["emp_exp"],
        "loan_amnt": p["loan_amnt"],
        "loan_int_rate": p["int_rate"],
        "loan_percent_income": round(p["loan_amnt"] / max(p["income"], 1), 4),
        "cb_person_cred_hist_length": p["cred_hist"],
        "credit_score": p["credit_score"],
        "previous_loan_defaults_on_file": 1 if p["prev_default"] == "Yes" else 0,
        "person_gender_male": 1 if p["gender"] == "Male" else 0,
        "person_home_ownership_OTHER": int(p["home"] == "Other"),
        "person_home_ownership_OWN": int(p["home"] == "Own"),
        "person_home_ownership_RENT": int(p["home"] == "Rent"),
        "loan_intent_EDUCATION": int(p["intent"] == "Education"),
        "loan_intent_HOMEIMPROVEMENT": int(p["intent"] == "Home Improvement"),
        "loan_intent_MEDICAL": int(p["intent"] == "Medical"),
        "loan_intent_PERSONAL": int(p["intent"] == "Personal"),
        "loan_intent_VENTURE": int(p["intent"] == "Venture"),
    }
    return pd.DataFrame([row])[FEATURES]


def prepare(df: pd.DataFrame, model_name: str):
    """Apply the scaler only to models that were trained on scaled data."""
    if SCALER is not None and model_name in SCALED_MODELS:
        out = df.copy()
        if hasattr(SCALER, "feature_names_in_"):
            cols = list(SCALER.feature_names_in_)
            out[cols] = SCALER.transform(out[cols])
            return out
        return pd.DataFrame(SCALER.transform(out.values), columns=out.columns)
    return df


def approval_probability(model_name: str, df: pd.DataFrame) -> float:
    model = MODELS[model_name]
    proba = model.predict_proba(prepare(df, model_name))[0]
    idx = list(model.classes_).index(APPROVED_CLASS)
    return float(proba[idx])


def gauge(prob: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%"},
        title={"text": "Approval Probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#6d28d9"},
               "steps": [{"range": [0, 40], "color": "rgba(239,68,68,.35)"},
                         {"range": [40, 60], "color": "rgba(245,158,11,.35)"},
                         {"range": [60, 100], "color": "rgba(16,185,129,.35)"}],
               "threshold": {"line": {"color": "black", "width": 4}, "value": 50}}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def feature_strength(model_name: str):
    m = MODELS[model_name]
    if hasattr(m, "feature_importances_"):
        return pd.Series(m.feature_importances_, index=FEATURES), "Feature importance"
    if hasattr(m, "coef_"):
        return pd.Series(np.abs(m.coef_[0]), index=FEATURES), "Absolute coefficient (inputs were scaled)"
    return None, None


def advice(p: dict, row: pd.DataFrame) -> list:
    tips = []
    lpi = float(row["loan_percent_income"].iloc[0])
    if p["prev_default"] == "Yes":
        tips.append("❗ A previous default is usually the strongest negative signal for lenders.")
    if lpi > 0.30:
        tips.append(f"⚠️ The loan is {lpi:.0%} of your income. Under ~25% usually looks safer.")
    if p["credit_score"] < 640:
        tips.append("⚠️ A credit score below ~640 lowers approval chances. Improving it helps most.")
    if p["int_rate"] > 14:
        tips.append("⚠️ A high interest rate often reflects a risky profile in the training data.")
    if p["cred_hist"] < 3:
        tips.append("ℹ️ A short credit history gives the model less evidence of reliability.")
    if not tips:
        tips.append("✅ Nothing stands out as a major risk factor. Nice profile!")
    return tips


# =====================================================================
# 4. HEADER + SIDEBAR
# =====================================================================
st.markdown("""
<div class="hero"><h1>🏦 Loan Lends</h1>
<p>Pick an AI model, enter your details, and see if your loan would be approved, and why.<br>
<small>Built by <b>Zakir Ahmed</b></small></p></div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

if not MODELS:
    st.error("No model files found. Save your trained models from the notebook first:")
    st.code("import joblib\n" + "\n".join(
        f'joblib.dump({v.split(".")[0]}, "{v}")' for v in MODEL_FILES.values()), language="python")
    st.stop()

if MISSING:
    st.warning("Some model files were not found and are hidden: " + ", ".join(MISSING))

with st.sidebar:
    st.header("🤖 Choose your model")
    model_name = st.radio("Model", list(MODELS.keys()),
                          format_func=lambda n: f"{MODEL_INFO[n][0]} {n}", label_visibility="collapsed")
    st.caption(MODEL_INFO[model_name][1])
    m = METRICS[model_name]
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{m['Accuracy']:.0%}")
    c2.metric("F1 Score", f"{m['F1 Score']:.2f}")
    c1.metric("Precision", f"{m['Precision']:.0%}")
    c2.metric("Recall", f"{m['Recall']:.0%}")
    best = max(METRICS, key=lambda k: METRICS[k]["F1 Score"])
    if model_name != best and best in MODELS:
        st.info(f"💡 Best overall model: **{best}**")
    st.divider()
    if st.button("🗑️ Clear prediction history", use_container_width=True):
        st.session_state.history = []
    st.caption("👨‍💻 Built by **Zakir Ahmed**")

tab_predict, tab_compare, tab_how, tab_about = st.tabs(
    ["🔮 Predict", "📊 Model Comparison", "🧠 How It Works", "📘 About the Data"])

# =====================================================================
# 5. PREDICT TAB
# =====================================================================
with tab_predict:
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.subheader("👤 Applicant details")
        with st.expander("Personal", expanded=True):
            a, b = st.columns(2)
            age = a.slider("Age", 18, 100, 30)
            gender = b.radio("Gender", ["Male", "Female"], horizontal=True)
            education = a.selectbox("Education", list(EDUCATION_MAP.keys()), index=2)
            home = b.selectbox("Home ownership", ["Mortgage", "Own", "Rent", "Other"], index=2)
            emp_exp = a.slider("Work experience (years)", 0, 60, 5)
            income = b.number_input("Annual income", 5000, 5_000_000, 60000, step=1000)

        with st.expander("Loan", expanded=True):
            a, b = st.columns(2)
            intent = a.selectbox("Loan purpose", ["Personal", "Education", "Medical", "Venture",
                                                  "Home Improvement", "Debt Consolidation"])
            loan_amnt = b.number_input("Loan amount", 500, 1_000_000, 10000, step=500)
            int_rate = a.slider("Interest rate (%)", 5.0, 25.0, 11.0, 0.1)
            lpi = loan_amnt / max(income, 1)
            b.metric("Loan as % of income (auto)", f"{lpi:.1%}")

        with st.expander("Credit profile", expanded=True):
            a, b = st.columns(2)
            credit_score = a.slider("Credit score", 300, 850, 650)
            cred_hist = b.slider("Credit history length (years)", 0, 40, 5)
            prev_default = a.radio("Previous loan default on file?", ["No", "Yes"], horizontal=True)

        go_btn = st.button("🚀 Predict my loan status", type="primary", use_container_width=True)

    inputs = dict(age=age, gender=gender, education=education, home=home, emp_exp=emp_exp,
                  income=income, intent=intent, loan_amnt=loan_amnt, int_rate=int_rate,
                  credit_score=credit_score, cred_hist=cred_hist, prev_default=prev_default)

    with right:
        st.subheader("🎯 Result")
        if go_btn:
            row = build_row(inputs)
            with st.spinner(f"{model_name} is thinking..."):
                prob = approval_probability(model_name, row)
            approved = prob >= 0.5
            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"), "Model": model_name,
                "Income": income, "Loan": loan_amnt, "Credit score": credit_score,
                "Approval %": round(prob * 100, 1), "Result": "Approved" if approved else "Rejected"})

            st.plotly_chart(gauge(prob), use_container_width=True)
            if approved:
                st.markdown('<div class="verdict-ok">✅ Likely APPROVED</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown('<div class="verdict-no">❌ Likely REJECTED</div>', unsafe_allow_html=True)

            st.markdown("#### 💡 Personalised tips")
            for t in advice(inputs, row):
                st.write(t)

            st.markdown("#### 🗳️ What do all models say?")
            votes = {n: approval_probability(n, row) * 100 for n in MODELS}
            fig = go.Figure(go.Bar(x=list(votes.values()), y=list(votes.keys()), orientation="h",
                                   marker_color=["#10b981" if v >= 50 else "#ef4444" for v in votes.values()],
                                   text=[f"{v:.0f}%" for v in votes.values()], textposition="outside"))
            fig.update_layout(height=280, xaxis=dict(range=[0, 115], title="Approval probability (%)"),
                              margin=dict(l=10, r=10, t=10, b=10))
            fig.add_vline(x=50, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)
            n_yes = sum(v >= 50 for v in votes.values())
            st.caption(f"{n_yes} of {len(votes)} models approve this application.")

            st.session_state.last = (inputs, row, model_name)
        else:
            st.info("Fill in the details on the left and press **Predict**.")

    # ---- Extras shown after a prediction ----
    if "last" in st.session_state:
        l_inputs, l_row, l_model = st.session_state.last
        st.divider()
        w1, w2 = st.columns(2, gap="large")

        with w1:
            st.subheader("🧪 What-if: change the loan amount")
            st.caption(f"See how {l_model} reacts if you borrow less or more (everything else unchanged).")
            factor = st.slider("Loan amount multiplier", 0.2, 2.0, 1.0, 0.1)
            amounts = np.linspace(l_inputs["loan_amnt"] * 0.2, l_inputs["loan_amnt"] * 2, 25)
            probs = []
            for amt in amounts:
                tmp = dict(l_inputs, loan_amnt=float(amt))
                probs.append(approval_probability(l_model, build_row(tmp)) * 100)
            cur = approval_probability(l_model, build_row(dict(l_inputs, loan_amnt=l_inputs["loan_amnt"] * factor))) * 100
            fig = go.Figure(go.Scatter(x=amounts, y=probs, mode="lines", line=dict(color="#6d28d9", width=3)))
            fig.add_trace(go.Scatter(x=[l_inputs["loan_amnt"] * factor], y=[cur], mode="markers",
                                     marker=dict(size=14, color="#db2777"), name="Selected"))
            fig.update_layout(height=300, xaxis_title="Loan amount", yaxis_title="Approval %",
                              showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.metric(f"Approval at {l_inputs['loan_amnt'] * factor:,.0f}", f"{cur:.1f}%")

        with w2:
            st.subheader("🔍 What drives this model?")
            strength, label = feature_strength(l_model)
            if strength is None:
                st.info("KNN has no built-in feature importance because it compares whole applicants, not individual columns.")
            else:
                top = strength.sort_values().tail(10)
                fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color="#6d28d9"))
                fig.update_layout(height=340, title=f"Top 10: {label}", margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

        d1, d2 = st.columns(2)
        with d1:
            with st.expander("📋 Exact data sent to the model"):
                st.dataframe(l_row.T.rename(columns={0: "value"}), use_container_width=True)
        with d2:
            if st.session_state.history:
                hist = pd.DataFrame(st.session_state.history)
                with st.expander("🕘 Prediction history", expanded=True):
                    st.dataframe(hist, use_container_width=True, hide_index=True)
                    st.download_button("⬇️ Download history (CSV)", hist.to_csv(index=False).encode(),
                                       "loan_predictions.csv", "text/csv")

# =====================================================================
# 6. COMPARISON TAB
# =====================================================================
with tab_compare:
    st.subheader("📊 How do the models compare?")
    mdf = pd.DataFrame(METRICS).T
    st.dataframe(mdf.style.highlight_max(axis=0, color="#86efac").format("{:.3f}"), use_container_width=True)

    metric_cols = list(mdf.columns)
    fig = go.Figure()
    for name in mdf.index:
        fig.add_trace(go.Bar(name=name, x=metric_cols, y=mdf.loc[name].values))
    fig.update_layout(barmode="group", height=400, yaxis=dict(range=[0, 1]), title="Metrics side by side")
    st.plotly_chart(fig, use_container_width=True)

    pick = st.multiselect("Radar view: choose models", list(mdf.index), default=["Random Forest", "XGBoost"])
    radar = go.Figure()
    for name in pick:
        vals = mdf.loc[name].tolist()
        radar.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=metric_cols + metric_cols[:1], fill="toself", name=name))
    radar.update_layout(height=420, polar=dict(radialaxis=dict(range=[0, 1])))
    st.plotly_chart(radar, use_container_width=True)

    st.success("🏆 **XGBoost** is the strongest overall (highest accuracy, recall and F1). "
               "**Logistic Regression** has the lowest recall: it misses many people who should be approved.")

# =====================================================================
# 7. HOW IT WORKS TAB
# =====================================================================
with tab_how:
    st.subheader("🧠 How this app works (in plain English)")
    steps = [
        ("1️⃣ You enter your details", "Age, income, loan amount, credit score and so on. The app calculates 'loan as % of income' for you."),
        ("2️⃣ The app translates them into numbers", "Models only understand numbers. 'Rent' becomes `person_home_ownership_RENT = 1`, 'Yes' for a past default becomes `1`, and education becomes a level from 0 up."),
        ("3️⃣ The chosen model reads the numbers", "Each model was trained on thousands of past applications where the real outcome (`loan_status`) was known. It learned which patterns usually lead to approval."),
        ("4️⃣ It returns a probability", "Instead of just yes/no you get a percentage. Above 50% means **Approved**, below means **Rejected**."),
        ("5️⃣ Extra tools help you understand", "Compare all models, try 'what-if' loan amounts, and see which columns matter most."),
    ]
    for t, d in steps:
        st.markdown(f'<div class="step"><b>{t}</b><br>{d}</div>', unsafe_allow_html=True)

    st.markdown("### 🤖 The five models explained")
    cols = st.columns(3)
    for i, (n, (icon, desc)) in enumerate(MODEL_INFO.items()):
        with cols[i % 3]:
            st.markdown(f'<div class="card"><h4>{icon} {n}</h4><p>{desc}</p></div>', unsafe_allow_html=True)
            st.write("")

    st.markdown("### 📏 What do the scores mean?")
    st.markdown("""
- **Accuracy**: out of all predictions, how many were right.
- **Precision**: when the model says *Approved*, how often is it correct. (High = few wrong approvals.)
- **Recall**: of all people who really were approved, how many did the model find. (High = few missed approvals.)
- **F1 Score**: a balance of precision and recall. Good when the classes are uneven.
""")
    st.info("⚠️ This is a learning/demo tool. Real banks use many more factors, and a model's output is not a guarantee of any real loan decision.")

# =====================================================================
# 8. ABOUT TAB
# =====================================================================
with tab_about:
    st.subheader("📘 Columns used by the models")
    desc = pd.DataFrame({
        "Column": FEATURES,
        "Meaning": [
            "Applicant age", "Education level (encoded 1 to 5)", "Yearly income", "Years of work experience",
            "Loan amount requested", "Interest rate (%)", "Loan amount ÷ income", "Years of credit history",
            "Credit score", "Past default on file (1 = yes)", "1 = male, 0 = female",
            "Home ownership: Other", "Home ownership: Own", "Home ownership: Rent (all 0 = Mortgage)",
            "Loan purpose: Education", "Loan purpose: Home improvement", "Loan purpose: Medical",
            "Loan purpose: Personal", "Loan purpose: Venture (all 0 = Debt consolidation)"],
    })
    st.dataframe(desc, use_container_width=True, hide_index=True)
    st.caption("Target column: `loan_status` (1 = Approved, 0 = Rejected).")

st.divider()
st.caption("Built with ❤️ by Zakir Ahmed | Models: Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN")
