import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor | Acknobit",
    page_icon="🔮",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background — dark navy so white text is visible */
    .stApp { background-color: #0d1b2e; }

    /* Force all default Streamlit text to be light */
    .stApp, .stApp p, .stApp span, .stApp label,
    .stApp div, .stApp li, .stApp h1, .stApp h2,
    .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #e2e8f0;
    }

    /* Streamlit widget labels */
    .stSelectbox label, .stSlider label,
    .stRadio label, .stCheckbox label,
    .stTextInput label, .stNumberInput label {
        color: #cbd5e1 !important;
    }

    /* Selectbox / input boxes */
    .stSelectbox > div > div,
    .stTextInput > div > div input,
    .stNumberInput > div > div input {
        background-color: #1e2d42 !important;
        color: #e2e8f0 !important;
        border: 1px solid #2d4a6b !important;
    }

    /* Slider text */
    .stSlider .stMarkdown p { color: #cbd5e1 !important; }

    /* Horizontal rule */
    hr { border-color: #2d4a6b; }

    /* Section headings written with st.markdown */
    h3 { color: #00C9A7 !important; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0A1628 0%, #1a3a5c 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        border: 1px solid #2d4a6b;
    }
    .main-header h1 {
        color: #00C9A7;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
    }
    .main-header p {
        color: #a0b4c8;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    /* Cards — dark background so text is readable */
    .card {
        background: #1a2d45;
        border: 1px solid #2d4a6b;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
        color: #e2e8f0;
    }
    .card p, .card span, .card div { color: #e2e8f0; }
    .card-title {
        color: #00C9A7;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #00C9A7;
    }

    /* Result boxes */
    .result-high {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(231,76,60,0.4);
    }
    .result-low {
        background: linear-gradient(135deg, #00C9A7, #00a98a);
        color: white;
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,201,167,0.4);
    }
    .result-number {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0;
        line-height: 1;
        color: white;
    }
    .result-label {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.9;
        color: white;
    }

    /* Probability bar label text */
    .prob-label-text { color: #94a3b8 !important; }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: #1e2d42;
        border: 1px solid #2d4a6b;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        flex: 1;
        text-align: center;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 800;
        color: #00C9A7;
    }
    .metric-lbl {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Sidebar */
    .css-1d391kg { background-color: #0A1628; }
    section[data-testid="stSidebar"] {
        background-color: #0A1628;
        border-right: 1px solid #2d4a6b;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #00C9A7, #00a98a);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,201,167,0.4);
    }

    /* Progress bar */
    .prob-bar-container {
        background: #1e2d42;
        border-radius: 50px;
        height: 18px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .prob-bar-fill-high {
        background: linear-gradient(90deg, #f39c12, #e74c3c);
        height: 100%;
        border-radius: 50px;
        transition: width 0.5s ease;
    }
    .prob-bar-fill-low {
        background: linear-gradient(90deg, #00C9A7, #27ae60);
        height: 100%;
        border-radius: 50px;
    }

    /* st.error / st.info / st.success boxes */
    .stAlert { background-color: #1a2d45 !important; color: #e2e8f0 !important; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Colab mein Drive se load hoga
    model_path  = '/content/drive/My Drive/Capstone Project/Churn Prediction/models/best_model.pkl'
    scaler_path = '/content/drive/My Drive/Capstone Project/Churn Prediction/models/scaler.pkl'

    # Local testing ke liye fallback
    if not os.path.exists(model_path):
        model_path  = 'models/best_model.pkl'
        scaler_path = 'models/scaler.pkl'

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# ── Feature Engineering Function ─────────────────────────────
def engineer_features(data):
    """Same feature engineering jo Step 3 mein kiya tha"""
    internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies']

    data['total_services']      = data[internet_cols].sum(axis=1)
    data['charge_per_tenure']   = data['MonthlyCharges'] / (data['tenure'] + 1)
    data['is_new_customer']     = (data['tenure'] <= 12).astype(int)
    data['has_no_protection']   = ((data['OnlineSecurity'] == 0) & (data['TechSupport'] == 0)).astype(int)
    data['payment_consistency'] = data['TotalCharges'] / (data['tenure'] * data['MonthlyCharges'] + 1)
    return data


# ── Preprocessing Function ────────────────────────────────────
def preprocess_input(inputs, scaler):
    """User input ko model-ready format mein convert karo"""

    # Binary mappings
    yes_no = {'Yes': 1, 'No': 0}

    row = {
        'gender':           1 if inputs['gender'] == 'Male' else 0,
        'SeniorCitizen':    inputs['SeniorCitizen'],
        'Partner':          yes_no[inputs['Partner']],
        'Dependents':       yes_no[inputs['Dependents']],
        'tenure':           inputs['tenure'],
        'PhoneService':     yes_no[inputs['PhoneService']],
        'OnlineSecurity':   1 if inputs['OnlineSecurity'] == 'Yes' else 0,
        'OnlineBackup':     1 if inputs['OnlineBackup'] == 'Yes' else 0,
        'DeviceProtection': 1 if inputs['DeviceProtection'] == 'Yes' else 0,
        'TechSupport':      1 if inputs['TechSupport'] == 'Yes' else 0,
        'StreamingTV':      1 if inputs['StreamingTV'] == 'Yes' else 0,
        'StreamingMovies':  1 if inputs['StreamingMovies'] == 'Yes' else 0,
        'PaperlessBilling': yes_no[inputs['PaperlessBilling']],
        'MonthlyCharges':   inputs['MonthlyCharges'],
        'TotalCharges':     inputs['tenure'] * inputs['MonthlyCharges'],

        # MultipleLines OHE (drop_first=True → base = 'No')
        'MultipleLines_No phone service': 1 if inputs['MultipleLines'] == 'No phone service' else 0,
        'MultipleLines_Yes':              1 if inputs['MultipleLines'] == 'Yes' else 0,

        # InternetService OHE (base = 'DSL')
        'InternetService_Fiber optic': 1 if inputs['InternetService'] == 'Fiber optic' else 0,
        'InternetService_No':          1 if inputs['InternetService'] == 'No' else 0,

        # Contract OHE (base = 'Month-to-month')
        'Contract_One year': 1 if inputs['Contract'] == 'One year' else 0,
        'Contract_Two year': 1 if inputs['Contract'] == 'Two year' else 0,

        # PaymentMethod OHE (base = 'Bank transfer (automatic)')
        'PaymentMethod_Credit card (automatic)': 1 if inputs['PaymentMethod'] == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check':        1 if inputs['PaymentMethod'] == 'Electronic check' else 0,
        'PaymentMethod_Mailed check':            1 if inputs['PaymentMethod'] == 'Mailed check' else 0,
    }

    df = pd.DataFrame([row])

    # Feature engineering
    df = engineer_features(df)

    # Scale numerical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                      'charge_per_tenure', 'payment_consistency']
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔮 Customer Churn Predictor</h1>
    <p>Acknobit Capstone Project — Telco Customer Churn Prediction</p>
    <p style="color:#00C9A7; font-size:0.85rem;">Powered by Logistic Regression | AUC-ROC: 83.81%</p>
</div>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────
try:
    model, scaler = load_model()
    st.sidebar.success("Model loaded!")
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.info("Pehle 04_Model_Building.ipynb run karo aur models/ folder mein best_model.pkl save karo.")
    st.stop()


# ── Sidebar: About ────────────────────────────────────────────
st.sidebar.markdown("## About This App")
st.sidebar.markdown("""
**Project:** Telco Customer Churn Prediction

**Model:** Logistic Regression

**Performance:**
- Accuracy: 80.03%
- F1-Score: 58.74%
- AUC-ROC: 83.81%

**Dataset:** 7,032 customers, 29 features

**Built by:** Acknobit
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### How to Use")
st.sidebar.markdown("""
1. Customer ka data form mein bharo
2. **Predict Churn** button dabao
3. Result dekhlo!
""")


# ── Main Form ─────────────────────────────────────────────────
st.markdown("### Customer Information")
st.markdown("Neeche customer ka data bharo aur predict karo ki woh churn karega ya nahi.")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # ── Personal Info ──────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Personal Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    gender         = c1.selectbox("Gender", ["Male", "Female"])
    senior         = c2.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner        = c1.selectbox("Partner", ["Yes", "No"])
    dependents     = c2.selectbox("Dependents", ["Yes", "No"])
    tenure         = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12,
                                help="Kitne mahine se customer hai?")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Billing Info ───────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Billing Information</div>', unsafe_allow_html=True)
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0,
                                  value=65.0, step=0.5)
    contract        = st.selectbox("Contract Type",
                                    ["Month-to-month", "One year", "Two year"])
    payment_method  = st.selectbox("Payment Method",
                                    ["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
    st.markdown('</div>', unsafe_allow_html=True)


with col_right:
    # ── Phone Services ─────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Phone Services</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    phone_service  = c1.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = c2.selectbox("Multiple Lines",
                                   ["Yes", "No", "No phone service"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Internet Services ──────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Internet Services</div>', unsafe_allow_html=True)
    internet_service = st.selectbox("Internet Service",
                                     ["DSL", "Fiber optic", "No"])
    c1, c2 = st.columns(2)
    online_security   = c1.selectbox("Online Security",   ["Yes", "No"])
    online_backup     = c2.selectbox("Online Backup",     ["Yes", "No"])
    device_protection = c1.selectbox("Device Protection", ["Yes", "No"])
    tech_support      = c2.selectbox("Tech Support",      ["Yes", "No"])
    streaming_tv      = c1.selectbox("Streaming TV",      ["Yes", "No"])
    streaming_movies  = c2.selectbox("Streaming Movies",  ["Yes", "No"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict Button ─────────────────────────────────────
    predict_btn = st.button("🔮 Predict Churn", use_container_width=True)


# ── Prediction ────────────────────────────────────────────────
if predict_btn:
    inputs = {
        'gender': gender, 'SeniorCitizen': senior,
        'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
    }

    try:
        processed = preprocess_input(inputs, scaler)
        prob        = model.predict_proba(processed)[0][1]
        prediction  = model.predict(processed)[0]
        prob_pct    = round(prob * 100, 1)

        st.markdown("---")
        st.markdown("### Prediction Result")

        res_col, insight_col = st.columns([1, 1], gap="large")

        with res_col:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-high">
                    <p class="result-number">{prob_pct}%</p>
                    <p class="result-label">⚠️ HIGH CHURN RISK</p>
                    <p style="font-size:0.85rem; margin-top:0.5rem; opacity:0.85;">
                        Ye customer churn kar sakta hai!<br>
                        Retention offer bhejo immediately.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <p class="result-number">{prob_pct}%</p>
                    <p class="result-label">✅ LOW CHURN RISK</p>
                    <p style="font-size:0.85rem; margin-top:0.5rem; opacity:0.85;">
                        Customer loyal rehne ki zyada probability.<br>
                        Normal engagement continue karo.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Probability bar
            bar_color = "prob-bar-fill-high" if prediction == 1 else "prob-bar-fill-low"
            st.markdown(f"""
            <div style="margin-top:1rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#94a3b8;">
                    <span>0% (Loyal)</span><span>Churn Probability</span><span>100% (Will Churn)</span>
                </div>
                <div class="prob-bar-container">
                    <div class="{bar_color}" style="width:{prob_pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with insight_col:
            st.markdown('<div class="card"><div class="card-title">Why This Prediction?</div>', unsafe_allow_html=True)

            # Key risk factors
            risk_factors = []
            positive_factors = []

            if tenure <= 12:
                risk_factors.append("Naya customer hai (tenure ≤ 12 months)")
            else:
                positive_factors.append(f"Purana customer hai ({tenure} months)")

            if contract == "Month-to-month":
                risk_factors.append("Month-to-month contract = no commitment")
            elif contract == "Two year":
                positive_factors.append("2-year contract = high loyalty")

            if monthly_charges > 80:
                risk_factors.append(f"High monthly charges (${monthly_charges})")
            else:
                positive_factors.append(f"Reasonable charges (${monthly_charges})")

            total_svcs = sum([
                1 if online_security == 'Yes' else 0,
                1 if online_backup == 'Yes' else 0,
                1 if device_protection == 'Yes' else 0,
                1 if tech_support == 'Yes' else 0,
                1 if streaming_tv == 'Yes' else 0,
                1 if streaming_movies == 'Yes' else 0,
            ])
            if total_svcs <= 1:
                risk_factors.append(f"Kam services subscribed ({total_svcs}/6)")
            else:
                positive_factors.append(f"Multiple services ({total_svcs}/6)")

            if online_security == 'No' and tech_support == 'No':
                risk_factors.append("No security/support = less attached")

            if payment_method == 'Electronic check':
                risk_factors.append("Electronic check = highest churn rate payment method")

            if risk_factors:
                st.markdown("**Risk Factors:**")
                for f in risk_factors:
                    st.markdown(f"🔴 {f}")

            if positive_factors:
                st.markdown("**Positive Factors:**")
                for f in positive_factors:
                    st.markdown(f"🟢 {f}")

            st.markdown('</div>', unsafe_allow_html=True)

        # Business Recommendation
        st.markdown('<div class="card"><div class="card-title">Business Recommendation</div>', unsafe_allow_html=True)
        if prediction == 1:
            r1, r2, r3 = st.columns(3)
            r1.markdown("**Immediate Action**\n\nRetention call karo. Discount ya upgrade offer karo.")
            r2.markdown("**Short Term**\n\nContract upgrade offer karo (Month-to-month → 1 year).")
            r3.markdown("**Long Term**\n\nService bundling suggest karo. Zyada services = kam churn.")
        else:
            r1, r2, r3 = st.columns(3)
            r1.markdown("**Maintain**\n\nCurrent service quality maintain karo.")
            r2.markdown("**Upsell**\n\nAdditional services offer karo to increase stickiness.")
            r3.markdown("**Reward**\n\nLoyalty rewards program mein enroll karo.")
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.exception(e)


# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#94a3b8; font-size:0.85rem;'>"
    "Acknobit Capstone Project | Telco Customer Churn Prediction | "
    "Built with Streamlit + Scikit-learn"
    "</p>",
    unsafe_allow_html=True
)
