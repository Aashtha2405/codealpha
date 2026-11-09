import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Default Predictor", page_icon="💳", layout="wide")

MODEL_PATH = "credit_default_model.joblib"
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
STATUS_COLS = [f"PAY_{i}" for i in range(1, 7)]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning + feature engineering as the notebook (Steps 4 and 6)."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    df["AVG_BILL"] = df[BILL_COLS].mean(axis=1)
    df["AVG_PAY_AMT"] = df[PAY_AMT_COLS].mean(axis=1)
    df["PAY_RATIO"] = df[PAY_AMT_COLS].sum(axis=1) / (df[BILL_COLS].clip(lower=0).sum(axis=1) + 1)
    df["UTILIZATION"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
    df["MAX_DELAY"] = df[STATUS_COLS].max(axis=1)
    df["MONTHS_DELAYED"] = (df[STATUS_COLS] > 0).sum(axis=1)
    return df


def predict(model, raw: pd.DataFrame) -> np.ndarray:
    X = add_features(raw)
    if hasattr(model, "feature_names_in_"):
        X = X[list(model.feature_names_in_)]
    return model.predict_proba(X)[:, 1]


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load `{MODEL_PATH}`. Keep it in the same folder as this app.\n\n{e}")
    st.stop()

st.title("💳 Credit Card Default Predictor")
st.caption("Predicts the probability that a client defaults on next month's payment.")

threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.26, 0.01,
                              help="Use the best_t value from your notebook (about 0.26 for gradient boosting).")
st.sidebar.markdown("Probability at or above the threshold is flagged as **high risk**.")

tab1, tab2 = st.tabs(["Single client", "Batch (CSV)"])

# ---------- Single client ----------
with tab1:
    with st.form("client_form"):
        st.subheader("Client profile")
        c1, c2, c3, c4, c5 = st.columns(5)
        limit_bal = c1.number_input("Credit limit", 10000, 1000000, 100000, 10000)
        sex = c2.selectbox("Sex", [1, 2], format_func=lambda x: {1: "Male", 2: "Female"}[x])
        edu = c3.selectbox("Education", [1, 2, 3, 4],
                           format_func=lambda x: {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}[x], index=1)
        mar = c4.selectbox("Marital status", [1, 2, 3],
                           format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x], index=1)
        age = c5.number_input("Age", 18, 100, 35)

        st.subheader("Repayment status (-2/-1 = paid early or on time, 0 = revolving credit, 1-8 = months of delay)")
        cols = st.columns(6)
        labels = ["Last month", "2 months ago", "3 months ago", "4 months ago", "5 months ago", "6 months ago"]
        pay = [cols[i].number_input(labels[i], -2, 8, 0, key=f"pay{i}") for i in range(6)]

        st.subheader("Bill amounts")
        cols = st.columns(6)
        bills = [cols[i].number_input(labels[i], value=20000, step=1000, key=f"bill{i}") for i in range(6)]

        st.subheader("Payment amounts")
        cols = st.columns(6)
        pays = [cols[i].number_input(labels[i], min_value=0, value=2000, step=500, key=f"pamt{i}") for i in range(6)]

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = {"LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": edu, "MARRIAGE": mar, "AGE": age}
        row.update({f"PAY_{i+1}": pay[i] for i in range(6)})
        row.update({f"BILL_AMT{i+1}": bills[i] for i in range(6)})
        row.update({f"PAY_AMT{i+1}": pays[i] for i in range(6)})
        p = float(predict(model, pd.DataFrame([row]))[0])

        m1, m2 = st.columns(2)
        m1.metric("Default probability", f"{p:.1%}")
        if p >= threshold:
            m2.error("⚠️ High risk of default")
        else:
            m2.success("✅ Low risk of default")
        st.progress(min(p, 1.0))

# ---------- Batch ----------
with tab2:
    st.write("Upload a CSV with the original dataset columns: `LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, "
             "PAY_1 (or PAY_0), PAY_2..PAY_6, BILL_AMT1..6, PAY_AMT1..6`.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up).rename(columns={"PAY_0": "PAY_1"})
        needed = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + STATUS_COLS + BILL_COLS + PAY_AMT_COLS
        missing = [c for c in needed if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            probs = predict(model, data[needed])
            out = data.copy()
            out["default_probability"] = probs.round(4)
            out["high_risk"] = (probs >= threshold).astype(int)
            st.metric("Clients flagged high risk", f"{out['high_risk'].sum()} of {len(out)}")
            st.dataframe(out, use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")