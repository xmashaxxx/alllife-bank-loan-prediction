"""
AllLife Bank Personal Loan Predictor
Decision Tree model + Claude-powered plain-English explanations
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AllLife Bank Loan Predictor",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def train_model(df):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split

    drop_cols = [c for c in ["ID", "ZIPCode", "Experience"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    if "Education" in df.columns:
        df["Education"] = df["Education"].astype(int)

    target = "Personal_Loan"
    features = [c for c in df.columns if c != target]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=7, min_samples_leaf=10,
        random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Recall":   round(recall_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "F1":       round(f1_score(y_test, y_pred), 4),
    }

    return model, features, metrics


def get_claude_explanation(customer, prediction, probability, feature_importances):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "_Claude explanation requires ANTHROPIC_API_KEY to be set._"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = "\n".join([f"  - {k}: {v:.1%} importance" for k, v in top_features])
        outcome = "ACCEPT the personal loan offer" if prediction == 1 else "DECLINE the personal loan offer"

        prompt = f"""You are a senior loan analyst at AllLife Bank explaining a model prediction to a relationship manager.

Customer profile:
- Age: {customer.get('Age')} years
- Annual Income: ${customer.get('Income', 0) * 1000:,.0f}
- Family size: {customer.get('Family')} people
- Education: {['Undergrad', 'Graduate', 'Advanced/Professional'][int(customer.get('Education', 1)) - 1]}
- Monthly credit card spending: ${customer.get('CCAvg', 0) * 1000:,.0f}
- Mortgage: ${customer.get('Mortgage', 0) * 1000:,.0f}
- Has CD account: {'Yes' if customer.get('CD_Account') == 1 else 'No'}
- Has securities account: {'Yes' if customer.get('Securities_Account') == 1 else 'No'}
- Uses online banking: {'Yes' if customer.get('Online') == 1 else 'No'}
- Has AllLife credit card: {'Yes' if customer.get('CreditCard') == 1 else 'No'}

Prediction: This customer is likely to {outcome}
Confidence: {probability:.1%}

Top model features by importance:
{top_str}

Write 3-4 sentences for the relationship manager explaining:
1. What the model predicted and why (reference specific customer values)
2. Which 2-3 factors were most influential for THIS customer
3. One practical recommendation for approaching this customer

Be specific with numbers. No jargon. Write as if briefing a colleague before a call."""

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text.strip()

    except Exception as e:
        return f"_Could not generate explanation: {e}_"


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🏦 AllLife Bank — Personal Loan Predictor")
st.caption("Decision tree model trained on 5,000 customer records · Claude-powered explanation layer")

st.sidebar.header("Setup")
uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
else:
    import os
    default_path = os.path.join(os.path.dirname(__file__), "Loan_Modelling.csv")
    if os.path.exists(default_path):
        df_raw = pd.read_csv(default_path)
    else:
        st.info(
            "**To get started:** upload `Loan_Modelling.csv` using the sidebar.\n\n"
            "See `data/README.md` for the expected column structure."
        )
        st.stop()
with st.spinner("Training decision tree model..."):
    model, feature_cols, metrics = train_model(df_raw.copy())

st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance")
for k, v in metrics.items():
    st.sidebar.metric(k, f"{v:.4f}")
st.sidebar.caption("Pre-pruned Decision Tree · Test set · 80/20 split")

st.subheader("Enter Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    age = st.slider("Age", 20, 70, 35)
    family = st.selectbox("Family Size", [1, 2, 3, 4], index=1)
    education = st.selectbox(
        "Education", options=[1, 2, 3],
        format_func=lambda x: {1: "Undergraduate", 2: "Graduate", 3: "Advanced/Professional"}[x],
        index=1
    )

with col2:
    st.markdown("**Financials**")
    income = st.slider("Annual Income ($000s)", 8, 224, 75)
    ccavg = st.slider("Monthly CC Spending ($000s)", 0.0, 10.0, 2.0, step=0.1)
    mortgage = st.slider("Mortgage Value ($000s)", 0, 635, 0)

with col3:
    st.markdown("**Banking Relationship**")
    securities = st.selectbox("Securities Account", [0, 1], format_func=lambda x: "Yes" if x else "No")
    cd_account = st.selectbox("CD Account", [0, 1], format_func=lambda x: "Yes" if x else "No")
    online = st.selectbox("Online Banking", [0, 1], format_func=lambda x: "Yes" if x else "No", index=1)
    creditcard = st.selectbox("AllLife Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")

st.markdown("---")

if st.button("🔍 Predict & Explain", type="primary"):
    customer = {
        "Age": age, "Income": income, "Family": family,
        "CCAvg": ccavg, "Education": education, "Mortgage": mortgage,
        "Securities_Account": securities, "CD_Account": cd_account,
        "Online": online, "CreditCard": creditcard
    }

    input_df = pd.DataFrame([customer])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    importances = dict(zip(feature_cols, model.feature_importances_))

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if prediction == 1:
            st.success("✅ **Likely to ACCEPT loan offer**")
        else:
            st.error("❌ **Likely to DECLINE loan offer**")

        st.metric("Acceptance Probability", f"{probability:.1%}")

        import plotly.express as px
        top_imp = pd.DataFrame(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)[:6],
            columns=["Feature", "Importance"]
        )
        fig = px.bar(top_imp, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues",
                     title="Top Feature Importances")
        fig.update_layout(coloraxis_showscale=False, height=300,
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with res_col2:
        st.subheader("🤖 Claude's Explanation")
        with st.spinner("Generating explanation..."):
            explanation = get_claude_explanation(customer, prediction, probability, importances)
        st.markdown(explanation)
        st.caption(
            "Generated by Claude (Haiku) based on feature importances and customer profile. "
            "Decision-support tool only — not a credit determination."
        )
