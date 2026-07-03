import pickle

import pandas as pd
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("models/model.h5")

    with open("models/label_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)

    with open("models/onehot_encoder_geo.pkl", "rb") as file:
        onehot_encoder_geo = pickle.load(file)

    with open("models/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


def build_input_frame(
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_of_products,
    has_cr_card,
    is_active_member,
    estimated_salary,
    label_encoder_gender,
    onehot_encoder_geo,
):
    base_frame = pd.DataFrame(
        {
            "CreditScore": [credit_score],
            "Gender": [label_encoder_gender.transform([gender])[0]],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
        }
    )

    geo_frame = pd.DataFrame(
        onehot_encoder_geo.transform(pd.DataFrame({"Geography": [geography]})).toarray(),
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"]),
    )

    return pd.concat([base_frame.reset_index(drop=True), geo_frame], axis=1)


def get_risk_band(probability):
    if probability >= 0.75:
        return "Very high risk", "high"
    if probability >= 0.5:
        return "High risk", "high"
    if probability >= 0.25:
        return "Moderate risk", "medium"
    return "Low risk", "low"


st.markdown(
    """
    <style>
        :root {
            color-scheme: dark;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.22) 0%, rgba(9, 11, 18, 0) 36%),
                radial-gradient(circle at bottom right, rgba(16, 185, 129, 0.14) 0%, rgba(9, 11, 18, 0) 30%),
                linear-gradient(180deg, #070b14 0%, #0b1120 100%);
            color: #e5e7eb;
        }
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 2rem 2.2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(8, 15, 31, 0.96), rgba(15, 23, 42, 0.94));
            color: white;
            border: 1px solid rgba(148, 163, 184, 0.16);
            box-shadow: 0 30px 70px rgba(2, 6, 23, 0.45);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin: 0.75rem 0 0;
            color: rgba(226, 232, 240, 0.82);
            font-size: 1.02rem;
            max-width: 60ch;
        }
        .card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(17, 24, 39, 0.94));
            border: 1px solid rgba(71, 85, 105, 0.45);
            border-radius: 22px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 18px 45px rgba(2, 6, 23, 0.34);
        }
        .stat-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #94a3b8;
            margin-bottom: 0.35rem;
        }
        .stat-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.2rem;
        }
        .stat-help {
            color: #cbd5e1;
            font-size: 0.92rem;
        }
        .result-pill {
            display: inline-block;
            padding: 0.38rem 0.8rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .result-pill.high {
            background: rgba(239, 68, 68, 0.18);
            color: #fca5a5;
        }
        .result-pill.medium {
            background: rgba(245, 158, 11, 0.18);
            color: #fdba74;
        }
        .result-pill.low {
            background: rgba(34, 197, 94, 0.16);
            color: #86efac;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(8, 15, 31, 0.98), rgba(6, 10, 20, 0.98));
            border-right: 1px solid rgba(71, 85, 105, 0.36);
        }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            color: #dbeafe;
        }
        div[data-testid="stForm"] {
            background: rgba(15, 23, 42, 0.62);
            border: 1px solid rgba(71, 85, 105, 0.42);
            border-radius: 24px;
            padding: 1.1rem;
            box-shadow: 0 18px 45px rgba(2, 6, 23, 0.28);
        }
        .feature-list li {
            margin-bottom: 0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()


st.markdown(
    """
    <section class="hero">
        <h1>Customer Churn Prediction</h1>
        <p>
            Estimate churn risk for a bank customer using the trained ANN model from the notebook workflow.
            The app applies the same encoding and scaling steps used during training, then returns a simple risk interpretation.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.write("")

with st.sidebar:
    st.markdown("### Model Overview")
    st.caption(
        "The prediction pipeline mirrors the training notebook: label encoding for gender, one-hot encoding for geography, and standard scaling before ANN inference."
    )
    st.markdown("### Input Features")
    st.markdown(
        """
        <ul class="feature-list">
            <li><strong>Customer profile:</strong> Geography, gender, age, tenure</li>
            <li><strong>Account state:</strong> Balance, credit score, salary</li>
            <li><strong>Engagement:</strong> Number of products, credit card ownership, active member flag</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Project Files")
    st.caption("`app.py` powers the Streamlit interface. `dl-3-simple-ann.ipynb` trains the model. `prediction.ipynb` demonstrates the inference pipeline.")


st.markdown("## Run a Prediction")
st.caption("Use the form below to evaluate a single customer profile.")

with st.form("churn_prediction_form"):
    left_col, right_col = st.columns(2)

    with left_col:
        geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0], index=0)
        gender = st.selectbox("Gender", label_encoder_gender.classes_, index=0)
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure", 0, 10, 3)
        num_of_products = st.slider("Number of Products", 1, 4, 2)

    with right_col:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600, step=1)
        balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=1000.0, format="%.2f")
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            format="%.2f",
        )
        has_cr_card = st.selectbox("Has Credit Card", [0, 1], index=1, format_func=lambda value: "Yes" if value else "No")
        is_active_member = st.selectbox("Is Active Member", [0, 1], index=1, format_func=lambda value: "Yes" if value else "No")

    submitted = st.form_submit_button("Predict churn risk")


if submitted:
    input_frame = build_input_frame(
        credit_score=credit_score,
        geography=geography,
        gender=gender,
        age=age,
        tenure=tenure,
        balance=balance,
        num_of_products=num_of_products,
        has_cr_card=has_cr_card,
        is_active_member=is_active_member,
        estimated_salary=estimated_salary,
        label_encoder_gender=label_encoder_gender,
        onehot_encoder_geo=onehot_encoder_geo,
    )

    scaled_input = scaler.transform(input_frame)
    prediction_probability = float(model.predict(scaled_input, verbose=0)[0][0])
    risk_label, risk_level = get_risk_band(prediction_probability)

    st.markdown("## Prediction Result")
    result_col_1, result_col_2, result_col_3 = st.columns(3)

    with result_col_1:
        st.markdown(
            f"""
            <div class="card">
                <div class="stat-label">Churn probability</div>
                <div class="stat-value">{prediction_probability:.1%}</div>
                <div class="stat-help">Model score for the submitted profile.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_col_2:
        st.markdown(
            f"""
            <div class="card">
                <div class="stat-label">Risk band</div>
                <div class="stat-value">{risk_label}</div>
                <div class="stat-help">A simplified interpretation of the probability score.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_col_3:
        recommendation = "Prioritize retention outreach" if prediction_probability >= 0.5 else "Customer appears stable"
        st.markdown(
            f"""
            <div class="card">
                <div class="stat-label">Action</div>
                <div class="stat-value">{recommendation}</div>
                <div class="stat-help">Use this as a starting point for customer success review.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(min(prediction_probability, 1.0))

    st.markdown(
        f"<span class='result-pill {risk_level}'>{risk_label}</span>",
        unsafe_allow_html=True,
    )

    if prediction_probability >= 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")

    with st.expander("See the transformed input passed to the model"):
        st.dataframe(input_frame, use_container_width=True)

else:
    overview_col_1, overview_col_2 = st.columns(2)

    with overview_col_1:
        st.markdown(
            """
            <div class="card">
                <div class="stat-label">What this app does</div>
                <div class="stat-value">Interactive churn scoring</div>
                <div class="stat-help">Pick a customer profile, submit it once, and review the churn probability with a clear interpretation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with overview_col_2:
        st.markdown(
            """
            <div class="card">
                <div class="stat-label">Training flow</div>
                <div class="stat-value">Notebook-backed</div>
                <div class="stat-help">The model, encoders, and scaler were exported from the training notebook and reused here for inference.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
