import streamlit as st
import pickle
import pandas as pd

scaler = pickle.load(open("standard_scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
kmeans_model = pickle.load(open("kmeans_model.pkl", "rb"))

st.sidebar.header("Customer Data Input")


BALANCE = st.sidebar.number_input("BALANCE", min_value=0.0)
BALANCE_FREQUENCY = st.sidebar.slider("BALANCE_FREQUENCY", min_value=0.0, max_value=1.0, step=0.01)
PURCHASES = st.sidebar.number_input("PURCHASES", min_value=0.0)
ONEOFF_PURCHASES = st.sidebar.number_input("ONEOFF_PURCHASES", min_value=0.0)
INSTALLMENTS_PURCHASES = st.sidebar.number_input("INSTALLMENTS_PURCHASES", min_value=0.0)
CASH_ADVANCE = st.sidebar.number_input("CASH_ADVANCE", min_value=0.0)
PURCHASES_FREQUENCY = st.sidebar.slider("PURCHASES_FREQUENCY", min_value=0.0, max_value=1.0, step=0.01)
ONEOFF_PURCHASES_FREQUENCY = st.sidebar.slider("ONEOFF_PURCHASES_FREQUENCY", min_value=0.0, max_value=1.0, step=0.01)
PURCHASES_INSTALLMENTS_FREQUENCY = st.sidebar.slider("PURCHASES_INSTALLMENTS_FREQUENCY", min_value=0.0, max_value=1.0, step=0.01)
CASH_ADVANCE_FREQUENCY = st.sidebar.slider("CASH_ADVANCE_FREQUENCY", min_value=0.0, max_value=1.0, step=0.01)
CASH_ADVANCE_TRX = st.sidebar.number_input("CASH_ADVANCE_TRX", min_value=0)
PURCHASES_TRX = st.sidebar.number_input("PURCHASES_TRX", min_value=0)
CREDIT_LIMIT = st.sidebar.number_input("CREDIT_LIMIT", min_value=0.0)
PAYMENTS = st.sidebar.number_input("PAYMENTS", min_value=0.0)
MINIMUM_PAYMENTS = st.sidebar.number_input("MINIMUM_PAYMENTS", min_value=0.0)
PRC_FULL_PAYMENT = st.sidebar.slider("PRC_FULL_PAYMENT", min_value=0.0, max_value=1.0, step=0.01)
TENURE = st.sidebar.number_input("TENURE", min_value=0)


input_data = pd.DataFrame({
    "BALANCE": [BALANCE],
    "BALANCE_FREQUENCY": [BALANCE_FREQUENCY],
    "PURCHASES": [PURCHASES],
    "ONEOFF_PURCHASES": [ONEOFF_PURCHASES],
    "INSTALLMENTS_PURCHASES": [INSTALLMENTS_PURCHASES],
    "CASH_ADVANCE": [CASH_ADVANCE],
    "PURCHASES_FREQUENCY": [PURCHASES_FREQUENCY],
    "ONEOFF_PURCHASES_FREQUENCY": [ONEOFF_PURCHASES_FREQUENCY],
    "PURCHASES_INSTALLMENTS_FREQUENCY": [PURCHASES_INSTALLMENTS_FREQUENCY],
    "CASH_ADVANCE_FREQUENCY": [CASH_ADVANCE_FREQUENCY],
    "CASH_ADVANCE_TRX": [CASH_ADVANCE_TRX],
    "PURCHASES_TRX": [PURCHASES_TRX],
    "CREDIT_LIMIT": [CREDIT_LIMIT],
    "PAYMENTS": [PAYMENTS],
    "MINIMUM_PAYMENTS": [MINIMUM_PAYMENTS],
    "PRC_FULL_PAYMENT": [PRC_FULL_PAYMENT],
    "TENURE": [TENURE]
})

st.write("### Entered Customer Data")
st.write(input_data)


if st.button("Submit"):
    # Apply transformations
    st.write("### Transforming data...")
    scaled_data = scaler.transform(input_data)
    pca_data = pca.transform(scaled_data)


    predicted_cluster = kmeans_model.predict(pca_data)

    st.write(f"### Predicted Cluster: {predicted_cluster[0]}")


    cluster_descriptions = {
        0: "Cluster 0 (Moderate Usage, Balanced Payment Behavior):\n"
           "Customers in this cluster maintain moderate balances and exhibit a balanced spending pattern. "
           "They tend to make installment purchases rather than one-off purchases, showing a conservative approach to credit usage. "
           "Their reliance on cash advances is low, and their credit limits are relatively modest. Payment behavior is stable, "
           "with some users paying off their balances in full, reflecting responsible financial habits.",
        
        1: "Cluster 1 (Heavy Cash Advance Usage, Financial Strain):\n"
           "Customers in this cluster heavily rely on cash advances but make minimal purchases through regular channels. "
           "Despite holding moderate balances, their payment behavior suggests that most payments go toward repaying cash advances. "
           "Few users pay off their balances in full, indicating potential financial stress.",
        
        2: "Cluster 2 (Balanced Usage, Moderate Financial Behavior):\n"
           "Customers maintain moderate balances and exhibit a balanced usage pattern between one-off and installment purchases. "
           "Cash advance usage is low, and credit limits are average. Payment levels are consistent, with a moderate portion paying "
           "off their balances in full, indicating relatively sound financial management.",
        
        3: "Cluster 3 (High Balances, Financial Vulnerability):\n"
           "Customers in this cluster hold high balances and exhibit high levels of spending, including frequent one-off and installment purchases. "
           "There is also significant reliance on cash advances. Payments are substantial but insufficient to cover their high expenditures, "
           "leading to financial strain. Very few users pay off their balances in full, signaling potential debt management issues.",
        
        4: "Cluster 4 (High Spending, Financially Disciplined):\n"
           "Cluster 4 customers are the highest spenders, with large balances and high credit limits. However, they demonstrate strong financial "
           "responsibility by making large payments and paying off balances in full at a higher rate than other clusters. Their credit usage is healthy, "
           "suggesting they are financially well-off."
    }


    marketing_strategies = {
        0: "Marketing Strategy: Offer incentives for increased credit usage, such as cashback or rewards on installment purchases. "
           "Consider raising credit limits for those with a strong repayment history.",
        
        1: "Marketing Strategy: Provide financial counseling or offer products with lower interest rates on cash advances. Promote balance "
           "transfer options or lower interest loans to alleviate financial stress.",
        
        2: "Marketing Strategy: Encourage increased usage by offering rewards for consistent spending. Consider promoting rewards programs "
           "or increasing credit limits slightly to stimulate higher engagement.",
        
        3: "Marketing Strategy: Provide debt consolidation options, financial planning tools, or help restructure their credit lines. "
           "Encourage them to reduce reliance on cash advances to improve financial health.",
        
        4: "Marketing Strategy: Retain these high-value customers by offering premium rewards programs, credit line increases, or special privileges. "
           "Target them with luxury or high-tier products to increase loyalty and engagement."
    }


    st.write("### Cluster Description")
    st.write(cluster_descriptions[predicted_cluster[0]])

    st.write("### Recommended Marketing Strategy")
    st.write(marketing_strategies[predicted_cluster[0]])
